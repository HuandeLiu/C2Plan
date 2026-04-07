from skimage import io
from skimage import morphology, feature, transform, measure
from pathlib import Path
from scipy import stats
from scipy import ndimage
from shapely import geometry
import numpy as np
from skimage.segmentation import watershed
import cv2
from utils import collide2d, point_box_relation, door_room_relation
from scipy.ndimage import grey_dilation

class Floorplan():

    @property
    def boundary(self):
        return self.image[..., 0]  # 外墙127，前门255，其他0

    @property
    def category(self):
        return self.image[..., 1]  # 13是外部区域 14是外墙，16是内墙 15是前门，17是内门吗，[0,1,2,...]这个是房间的类型

    @property
    def instance(self):
        # TODO 重置编号
        im = self.image[..., 2]  # 房子内的房间的索引编号mask，用来区分第1个通道里面相同的房间类型导致编号相同问题，外部区域和墙、门统一用0表示
        regions = measure.regionprops(im)
        new_instance = np.zeros_like(im)
        ord = 1
        for region in regions:
            new_instance[region.coords[:, 0], region.coords[:, 1]] = ord
            ord += 1
        # im = new_instance

        inside = self.image[..., 3] > 0
        filled = new_instance.copy()
        # 迭代传播
        max_iter = 7
        for _ in range(max_iter):
            # 在邻域里传播房间ID
            dilated = grey_dilation(filled, size=(3, 3))

            # 只更新原来是墙的像素
            update_mask = (filled == 0) & (dilated > 0)
            filled[update_mask] = dilated[update_mask]

            if np.all(filled > 0):
                break
        filled = filled * inside
        return filled

    @property
    def inside(self):
        return self.image[..., 3]  # 内部区域+内墙为255，其他为0

    def __init__(self, file_path):
        self.path = file_path
        self.name = Path(self.path).stem
        self.image = io.imread(self.path)
        self.h, self.w, self.c = self.image.shape

        self.front_door = None # 前门坐标左上右下
        self.exterior_boundary = None # [x,y,dir,flag] xy坐标dir是方向、flag是是否为前门，从前门开始的转向
        self.boundary_mask = self.boundary != 0
        self.rooms = None # [y0, x0, y1, x1, c]
        self.order = None
        self.room_mask = None # 第o个房间的类别是c，掩码是[256*256]
        self.edges = None # [u, v, relation]
        self.corners = []
        self.corners_mask = []

        self.archs = None # 内门的矩形框 左上右下
        self.graph = None # u,v,e,t,d = graph[i] # uv房间的位置关系是e，t表示有没有门连接，d表示第几扇门，没有门则为None
        self.door_pos = None  # N*2 房间u的门是第d个，在房间的位置是dpos

        self._get_front_door()
        self._get_exterior_boundary()
        self._get_rooms()
        self._get_edges()
        self._get_archs()
        self._get_graph()

    def __repr__(self):
        return f'{self.name},({self.h},{self.w},{self.c})'

    # TODO 获取前门坐标[左上，右下]
    def _get_front_door(self):
        front_door_mask = self.boundary == 255  # 获取前门的掩码
        # fast bbox
        # min_h,max_h = np.where(np.any(front_door_mask,axis=1))[0][[0,-1]]
        # min_w,max_w = np.where(np.any(front_door_mask,axis=0))[0][[0,-1]]
        # self.front_door = np.array([min_h,min_w,max_h,max_w],dtype=int)
        region = measure.regionprops(front_door_mask.astype(int))[0]
        self.front_door = np.array(region.bbox, dtype=int)  # 前门坐标，左上右下

    # TODO # [x,y,dir,flag] xy坐标dir是方向、flag是是否为前门，从前门开始的转向,最后会调整顺序，让门成为起点
    def _get_exterior_boundary(self):
        if self.front_door is None: self._get_front_door()
        self.exterior_boundary = []

        # 找出包含建筑的最小矩形区域，并适当扩展避免边界截断
        min_h, max_h = np.where(np.any(self.boundary, axis=1))[0][[0, -1]]
        min_w, max_w = np.where(np.any(self.boundary, axis=0))[0][[0, -1]]
        min_h = max(min_h - 10, 0)
        min_w = max(min_w - 10, 0)
        max_h = min(max_h + 10, self.h)
        max_w = min(max_w + 10, self.w)

        # src: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
        # search direction:0(right)/1(down)/2(left)/3(up) 这里的方向是相对于节点搜索方向
        # find the left-top point 轮廓追踪算法，从左上角开始遍历，找到第一个属于建筑内部的点作为起始点
        flag = False
        for h in range(min_h, max_h):
            for w in range(min_w, max_w):
                if self.inside[h, w] == 255:
                    self.exterior_boundary.append((h, w, 0))  # 第一个点的方向肯定是向右的
                    flag = True
                    break
            if flag:
                break

        # left/top edge: inside
        # right/bottom edge: outside
        while (flag):
            if self.exterior_boundary[-1][2] == 0:
                for w in range(self.exterior_boundary[-1][1] + 1, max_w):
                    corner_sum = 0
                    if self.inside[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0] - 1, w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0], w - 1] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0] - 1, w - 1] == 255:
                        corner_sum += 1
                    # 如果是2那么还是同向
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break

            if self.exterior_boundary[-1][2] == 1:
                for h in range(self.exterior_boundary[-1][0] + 1, max_h):
                    corner_sum = 0
                    if self.inside[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h - 1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h, self.exterior_boundary[-1][1] - 1] == 255:
                        corner_sum += 1
                    if self.inside[h - 1, self.exterior_boundary[-1][1] - 1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break

            if self.exterior_boundary[-1][2] == 2:
                for w in range(self.exterior_boundary[-1][1] - 1, min_w, -1):
                    corner_sum = 0
                    if self.inside[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0] - 1, w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0], w - 1] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0] - 1, w - 1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break

            if self.exterior_boundary[-1][2] == 3:
                for h in range(self.exterior_boundary[-1][0] - 1, min_h, -1):
                    corner_sum = 0
                    if self.inside[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h - 1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h, self.exterior_boundary[-1][1] - 1] == 255:
                        corner_sum += 1
                    if self.inside[h - 1, self.exterior_boundary[-1][1] - 1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break

            if new_point != self.exterior_boundary[0]:
                self.exterior_boundary.append(new_point)
            else:  # 回到远点，结束循环
                flag = False
        # 记录的是从左上的第一个点开始的(x,y)坐标，转向，0
        self.exterior_boundary = [[r, c, d, 0] for r, c, d in self.exterior_boundary]  # 记录每个转向点

        door_y1, door_x1, door_y2, door_x2 = self.front_door
        door_h, door_w = door_y2 - door_y1, door_x2 - door_x1
        is_vertical = door_h > door_w or door_h == 1  # 门的朝向

        insert_index = None
        door_index = None
        new_p = []
        th = 3
        # 将前门位置整合到建筑外部边界中
        for i in range(len(self.exterior_boundary)):
            y1, x1, d, _ = self.exterior_boundary[i]
            y2, x2, _, _ = self.exterior_boundary[(i + 1) % len(self.exterior_boundary)]
            if is_vertical != d % 2: continue  # 方向不匹配，跳过这个线段
            # 处理垂直方向的门（如入户门为垂直方向）
            if is_vertical and (x1 - th < door_x1 < x1 + th or x1 - th < door_x2 < x1 + th):  # 1:down 3:up
                # 创建边界线段和门线段的几何对象
                l1 = geometry.LineString([[y1, x1], [y2, x2]])
                l2 = geometry.LineString([[door_y1, x1], [door_y2, x1]])
                l12 = l1.intersection(l2)  # # 计算两线段的交点
                if l12.length > 0:  # 存在交点
                    dy1, dy2 = l12.xy[0]  # (y1>y2)==(dy1>dy2)
                    insert_index = i
                    door_index = i + (y1 != dy1)
                    # 插入新的边界点（标记为门，第四个参数为1）
                    if y1 != dy1: new_p.append([dy1, x1, d, 1])
                    if y2 != dy2: new_p.append([dy2, x1, d, 1])
            elif not is_vertical and (y1 - th < door_y1 < y1 + th or y1 - th < door_y2 < y1 + th):
                l1 = geometry.LineString([[y1, x1], [y2, x2]])
                l2 = geometry.LineString([[y1, door_x1], [y1, door_x2]])
                l12 = l1.intersection(l2)
                if l12.length > 0:
                    dx1, dx2 = l12.xy[1]  # (x1>x2)==(dx1>dx2)
                    insert_index = i
                    door_index = i + (x1 != dx1)
                    if x1 != dx1: new_p.append([y1, dx1, d, 1])
                    if x2 != dx2: new_p.append([y1, dx2, d, 1])
                    # 把门的节点插入到边界中
        if len(new_p) > 0:
            self.exterior_boundary = self.exterior_boundary[:insert_index + 1] + new_p + self.exterior_boundary[
                                                                                         insert_index + 1:]
        # 重新排列列表，使门成为新的起点
        self.exterior_boundary = self.exterior_boundary[door_index:] + self.exterior_boundary[:door_index]
        self.exterior_boundary = np.array(self.exterior_boundary, dtype=int)

    # TODO 矫正门的掩码
    def _adjust_door(self,y0, x0, y1, x1):
        # print("调整门")
        door_center = ((y0 + y1) // 2, (x0 + x1) // 2)
        center = door_center
        ny,nx = door_center[0],door_center[1]
        flag = (y1 - y0)>(x1 - x0) # True 垂直门 False 水平门
        if flag:
            for dx in [0,1,-1,2,-2,3,-3]:
                nx = door_center[1] + dx
                if 0<=nx<256 and self.instance[ny, nx+1]!=self.instance[ny, nx-1]: # 相邻房间不同
                    center=(ny,nx)
                    break
            if (x1 - x0) <=4:
                x1+=2
                x0-=2
        else:
            for dy in [0,1, -1,2,-2,3,-3]:
                ny = door_center[0] + dy
                if 0<=ny<256 and self.instance[ny+1, nx]!=self.instance[ny-1, nx]: # 相邻房间不同
                    center=(ny,nx)
                    break
            if (y1 - y0) <=6:
                y1+=2
                y0-=2
        move = (center[0]-door_center[0],center[1]-door_center[1])
        return y0+move[0], x0+move[1], y1+move[0] , x1+move[1]

    # 正交化
    def orthogonalize_polygon(self,approx, tol_angle=20):
        """
        将多边形近似到水平/竖直
        """
        pts = approx.reshape(-1, 2)
        new_pts = [pts[0]]

        for i in range(1, len(pts)):
            p1 = new_pts[-1]
            p2 = pts[i]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]

            if abs(dx) > abs(dy):  # 更水平
                p2 = (p2[0], p1[1])
            else:  # 更垂直
                p2 = (p1[0], p2[1])
            new_pts.append(p2)

        return np.array(new_pts, dtype=np.int32).reshape(-1, 1, 2)
    # TODO 获取房间的左上右下坐标，可以在这里根据 c 获取到房间的掩码
    def _get_rooms(self):
        if self.archs is None:self._get_archs() # 获取门的种类和坐标
        rooms = []
        order = []
        rooms_mask = []
        regions = measure.regionprops(self.instance) # 根据房间的编号划分出区域
        ord = 1
        for region in regions:
            # 获取区域内的主类别
            c = stats.mode(self.category[region.coords[:, 0], region.coords[:, 1]])[0]  # [0] 类别
            o = stats.mode(self.instance[region.coords[:, 0], region.coords[:, 1]])[0]  # [0] 编号
            order.append(o)
            ord += 1
            # 获取边框
            y0, x0, y1, x1 = np.array(region.bbox)
            rooms.append([y0, x0, y1, x1, c])  # 这里的c是房间类型
            rooms_mask.append(self.instance == o) # 第o个房间的类别是c，掩码是...
        for door in self.archs:
            c = door[-1]
            order.append([ord])
            y0, x0, y1, x1 = np.array(door[:-1])

            rooms.append([y0, x0, y1, x1, c])
            mask = np.zeros_like(self.instance, dtype=np.uint8)
            mask[y0:y1, x0:x1] = 1

            rooms_mask.append(mask)
            ord += 1
        self.rooms = np.array(rooms, dtype=int) # 房间的坐标和类别
        self.order = np.array(np.concatenate(order, axis=0), dtype=int)
        self.rooms_mask = np.array(rooms_mask, dtype=int)
        # corners_count = 0
        for i,(room_mask, room_id,room) in enumerate(zip(self.rooms_mask, self.order,self.rooms)):
            room_mask = room_mask.astype(np.uint8)
            room_mask = cv2.resize(room_mask, (256, 256), interpolation=cv2.INTER_AREA)
            corners_per_room, _ = cv2.findContours(room_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            corners_per_room = corners_per_room[0]  # (N, 1, 2)
            # TODO 多边形简化、正交化\门不做
            if room[-1] not in [15,17]:
                corners_per_room = cv2.approxPolyDP(corners_per_room, 1.0, closed=True)
                # corners_per_room = self.orthogonalize_polygon(approx)

            self.corners.append(corners_per_room[:, 0, :])
            self.corners_mask.extend([room_id] * len(corners_per_room))


            # corners_count += len(corners_per_room)
        # TODO 轮廓简化
        # for i, (room_mask, room_id) in enumerate(zip(self.rooms_mask, self.order)):
        #     room_mask = room_mask.astype(np.uint8)
        #     room_mask = cv2.resize(room_mask, (256, 256), interpolation=cv2.INTER_AREA)
        #     contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #     if len(contours) == 0:continue
        #     # 取最大轮廓（主要房间形状）
        #     main_contour = max(contours, key=cv2.contourArea)
        #     # 2. 使用Douglas-Peucker算法简化轮廓
        #     epsilon = 0.005 * cv2.arcLength(main_contour, True)  # 简化程度参数（可调整）
        #     approx_corners = cv2.approxPolyDP(main_contour, epsilon, True)
        #     # 3. 过滤过近的点（可选）
        #     min_distance = 5  # 像素距离阈值（可根据图像尺寸调整）
        #     filtered_corners = []
        #     prev_point = None
        #     for corner in approx_corners[:, 0, :]:
        #         if prev_point is None or np.linalg.norm(corner - prev_point) > min_distance:
        #             filtered_corners.append(corner)
        #             prev_point = corner
            # 4. 确保多边形闭合（首尾点不同时添加起点）
            # if len(filtered_corners) > 2:
            #     first_point = filtered_corners[0]
            #     last_point = filtered_corners[-1]
            #     if np.linalg.norm(first_point - last_point) > min_distance:
            #         filtered_corners.append(first_point)

            # self.corners.append(np.array(filtered_corners))
            # self.corners_mask.extend([room_id] * len(filtered_corners))

        self.corners = np.concatenate(self.corners, axis=0)

    # TODO 通过碰撞检测，获取房间之间的相邻关系
    def _get_edges(self, th=9):
        if self.rooms is None: self._get_rooms() # 获取房间类别
        edges = []
        for u in range(len(self.rooms)):
            for v in range(u + 1, len(self.rooms)):
                door = [15, 17]
                if self.rooms[u, -1] in door or self.rooms[v, -1] in door:continue
                # collide2d函数检测两个房间的边界框是否在th=9的阈值范围内重叠或相邻
                if not collide2d(self.rooms[u, :4], self.rooms[v, :4], th=th): continue
                uy0, ux0, uy1, ux1, c1 = self.rooms[u]
                vy0, vx0, vy1, vx1, c2 = self.rooms[v]
                uc = (uy0 + uy1) / 2, (ux0 + ux1) / 2
                vc = (vy0 + vy1) / 2, (vx0 + vx1) / 2
                if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                    relation = 5  # 'surrounding' # u完全包围v（如建筑包围房间）
                elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                    relation = 4  # 'inside' # u完全在v内部（如房间内的柱子）
                else:
                    relation = point_box_relation(uc, self.rooms[v, :4])  # 根据坐标边判断两个房间之间的关系
                edges.append([u, v, relation])

        self.edges = np.array(edges, dtype=int)

    # TODO 算出内门的外部矩形区域
    def _get_archs(self):
        '''
        Interior doors
        '''
        archs = []
        # treat archs as instances
        # index = len(self.rooms)+1

        # for category in range(num_category,len(room_label)):
        for category in [17,15]:  # only get doors for building graphs
            mask = (self.category == category).astype(np.uint8)  # 门区域为1，其他为0

            # distance transform -> threshold -> corner detection -> remove corner -> watershed -> label region
            # distance = cv2.distanceTransform(mask,cv2.DIST_C,3) 计算每个门区域像素到最近背景（非门区域）的欧氏距离，生成距离图（离门中心越近值越大）
            distance = ndimage.morphology.distance_transform_cdt(mask)

            # local_maxi = feature.peak_local_max(distance, indices=False) # line with one pixel
            local_maxi = (distance > 1).astype(np.uint8)  # 保留距离图中大于1的像素（过滤靠近门边缘的像素），作为门的候选中心点

            # corner_measurement = feature.corner_shi_tomasi(local_maxi) # short lines will be removed
            corner_measurement = feature.corner_harris(local_maxi)  # 用Harris角点检测算法识别门框角点，并移除这些点（避免角点被误认为门中心）

            local_maxi[corner_measurement > 0] = 0  # 剔除角点

            # 以局部极大值为种子点，结合距离图进行分水岭分割，将相邻门区域精确分离
            markers = measure.label(local_maxi)
            labels = watershed(-distance, markers, mask=mask, connectivity=8)
            # 对分割后的每个独立门区域，计算其最小外接矩形（Bounding Box）并保存
            regions = measure.regionprops(labels)
            for region in regions:
                y0, x0, y1, x1 = np.array(region.bbox)
                if y1 - y0 == 1: y1 += 1
                if x1 - x0 == 1: x1 += 1
                y0, x0, y1, x1 = self._adjust_door(y0, x0, y1, x1)
                archs.append([y0, x0, y1, x1, category])

        self.archs = np.array(archs, dtype=int)

    # TODO 添加有门的连接边，和无门的连接边，以及边的位置关系
    def _get_graph(self, th=9):
        '''
        More detail graph
        '''
        if self.rooms is None: self._get_rooms()
        if self.archs is None: self._get_archs()  # 内部门
        graph = []  # 存储图的边关系
        door_pos = [[None, 0] for i in range(len(self.rooms))]  # 记录每个房间的门位置
        edge_set = set()  # 避免重复边，相连房间的ID

        # add accessible edges
        doors = self.archs[self.archs[:, -1] == 17]  # 内门的数量
        for i in range(len(doors)):
            bbox = doors[i, :4]  # 门的边界框

            # left <-> right
            # 水平门：检测左右房间连通性
            for row in range(bbox[0], bbox[2]):
                u = self.instance[row, bbox[1] - 1] - 1  # 门左侧房间ID
                v = self.instance[row, bbox[3] + 1] - 1  # 门左侧房间ID
                if (u, v) in edge_set or (v, u) in edge_set: continue
                if u >= 0 and v >= 0 and u!=v:  # 如果左右没有房间，则为-1
                    edge_set.add((u, v))
                    graph.append([u, v, None, 1, i])  # type=1表示有门连接

            # up <-> down
            # 垂直门：检测上下房间连通性（逻辑同上）
            for col in range(bbox[1], bbox[3]):
                u = self.instance[bbox[0] - 1, col] - 1
                v = self.instance[bbox[2] + 1, col] - 1
                if (u, v) in edge_set or (v, u) in edge_set: continue
                if u >= 0 and v >= 0 and u!=v:
                    edge_set.add((u, v))
                    graph.append([u, v, None, 1, i])

        # add adjacent edges 两个房间相邻但是没有门连接
        for u in range(len(self.rooms)):
            for v in range(u + 1, len(self.rooms)):
                if (u, v) in edge_set or (v, u) in edge_set: continue

                # collision detection
                door = [15,17]
                if self.rooms[u, -1] not in door and self.rooms[v, -1] not in door:
                    if collide2d(self.rooms[u, :4], self.rooms[v, :4], th=th):
                        edge_set.add((u, v))  # 相邻
                        graph.append([u, v, None, 0, None])  # 但是没有门连接

        # add edge relation
        for i in range(len(graph)):
            u, v, e, t, d = graph[i]  # uv房间的关系是e，t表示有没有门连接，d表示第几扇门，没有门则为None
            uy0, ux0, uy1, ux1 = self.rooms[u, :4]
            vy0, vx0, vy1, vx1 = self.rooms[v, :4]
            uc = (uy0 + uy1) / 2, (ux0 + ux1) / 2
            vc = (vy0 + vy1) / 2, (vx0 + vx1) / 2

            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                relation = 5  # 'surrounding'
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                relation = 4  # 'inside'
            else:
                relation = point_box_relation(uc, self.rooms[v, :4])

            graph[i][2] = relation

            if d is not None:  # 有门连接的时候
                c_u = self.rooms[u, -1]
                c_v = self.rooms[v, -1]
                if c_u > c_v and door_pos[u][0] is None:  # 避免重复
                    room = u
                else:
                    room = v
                door_pos[room][0] = d

                d_center = self.archs[d, :4]
                d_center = (d_center[:2] + d_center[2:]) / 2.0

                dpos = door_room_relation(d_center, self.rooms[room, :4])
                if dpos != 0: door_pos[room][1] = dpos

        self.graph = graph  # u,v,e,t,d = graph[i] # uv房间的关系是e，t表示有没有门连接，d表示第几扇门，没有门则为None
        self.door_pos = door_pos  # N*2 房间u的门是第d个，在房间的位置是dpos

    def to_dict(self, xyxy=True, dtype=int):
        '''
        Compress data, notice:
        !!! int->uint8: a(uint8)+b(uint8) may overflow !!!
        '''
        # print(self.corners,self.corners_mask)
        return {
            'name': self.name,
            'rType': self.rooms[:, -1].astype(dtype),  # 房间类型
            'order': self.order, # 房间id
            # 'boxes': (self.rooms[:, [1, 0, 3, 2]]).astype(dtype)  # 房间框
            # if xyxy else self.rooms[:, :4].astype(dtype),
            'gtBoxNew': (self.rooms[:, [1, 0, 3, 2]]).astype(dtype)  # 房间框
            if xyxy else self.rooms[:, :4].astype(dtype),
            # 'room_mask': self.rooms_mask.astype(dtype),  # 房间掩码
            'corners':self.corners,
            'corners_mask':self.corners_mask,
            # 'door':self.archs.astype(dtype)[:,:-1], # 内门的矩形坐标
            'graph': self.graph, # u,v,e,t,d = graph[i] # uv房间的位置关系是e，t表示有没有门连接，d表示第几扇门，没有门则为None
            'boundary': self.exterior_boundary[:, [1, 0, 2, 3]].astype(dtype)  # 边界+前门
            if xyxy else self.exterior_boundary.astype(dtype),
            # 'boundary_mask':self.boundary_mask.astype(dtype),
            'rEdge': self.edges.astype(dtype),  # [u,v,relation] 房间之间的关系
        }


# self.front_door = None # 前门坐标左上右下
# self.exterior_boundary = None # [x,y,dir,flag] xy坐标dir是方向、flag是是否为前门，从前门开始的转向
# self.boundary_mask = self.boundary != 0
# self.rooms = None # [y0, x0, y1, x1, c]
# self.room_mask = None # 第o个房间的类别是c，掩码是[256*256]
# self.edges = None # [u, v, relation]
#
# self.archs = None # 内门的矩形框 左上右下
# self.graph = None
# self.door_pos = None  # N*2 房间u的门是第d个，在房间的位置是dpos

if __name__ == "__main__":
    RPLAN_DIR = '../data'
    file_path = f'{RPLAN_DIR}/8988.png'
    fp = Floorplan(file_path)
    data = fp.to_dict()
    print(data)