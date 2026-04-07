import math
import random
import torch as th
from scipy.ndimage import distance_transform_edt

from PIL import Image, ImageDraw
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json
import os
import cv2 as cv
from tqdm import tqdm
from shapely import geometry as gm
from shapely.ops import unary_union
from collections import defaultdict
import copy
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# import os
# if not os.path.exists("/dev/random"):
#     os.symlink("/dev/urandom", "/dev/random")
def load_rplanhg_data(
    batch_size,
    analog_bit,
    target_set = 8,
    set_name = 'train',
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of target set {target_set}")
    deterministic = False if set_name=='train' else True
    dataset = RPlanhgDataset(set_name, analog_bit, target_set)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False
        )
    while True:
        yield from loader


def load_demo_to_model_input(
    demo_json_path,
    analog_bit=False,
    target_set=8,
    set_name='eval',
):
    """
    将用户的 demo.json 配置转化为 (arr, cond) 模型输入。
    demo.json 结构参考 model/demo.json。
    """
    with open(demo_json_path) as f:
        spec = json.load(f)

    name = str(spec['name'])
    # 1. 使用 name 在 dataprocess/rplan_json 下定位真实样本，只处理该样本生成真实 arr/cond
    rplan_json_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataprocess", "rplan_json"))
    rplan_json_path = os.path.join(rplan_json_dir, f"{name}.json")
    if not os.path.exists(rplan_json_path):
        raise FileNotFoundError(f"Real sample json not found: {rplan_json_path}")

    arr, cond = _load_single_rplan_json_to_model_input(
        rplan_json_path=rplan_json_path,
        analog_bit=analog_bit,
        set_name=set_name,
    )

    # 2. 根据用户提供的 room_type / room_corner_nums / room_area_rate / room_connections
    #    构造一整套 syn_* 条件，保持和原始 eval 分支相同的尺寸与语义。
    room_types = spec['room_type']

    # 房间类型映射
    type_map = {15: 11, 17: 12, 16: 13}
    # 安全替换，找不到就用原来的值，绝对不崩溃
    room_types = [type_map.get(rt, rt) for rt in room_types]
    print(room_types)
    room_corner_nums = spec['room_corner_nums']
    room_area_rate = spec['room_area_rate']
    room_connections = spec['room_connections']  # 形如 [[u, v], ...]，用来构造 syn_graph 与 syn_door_mask

    assert len(room_types) == len(room_corner_nums) == len(room_area_rate), \
        'room_type / room_corner_nums / room_area_rate 长度必须一致'

    num_rooms = len(room_types)
    max_num_points = 100

    # 角点级别条件张量
    syn_room_types = np.zeros((max_num_points, 25), dtype=np.float32)
    syn_corner_indices = np.zeros((max_num_points, 32), dtype=np.float32)
    syn_room_indices = np.zeros((max_num_points, 32), dtype=np.float32)
    syn_src_key_padding_mask = np.ones((max_num_points,), dtype=np.float32)
    syn_connections = np.zeros((max_num_points, 2), dtype=np.int64)
    syn_room_areas = np.zeros((max_num_points, 1), dtype=np.float32)

    # 图与注意力掩码
    syn_gen_mask = np.ones((max_num_points, max_num_points), dtype=np.float32)
    syn_door_mask = np.ones((max_num_points, max_num_points), dtype=np.float32)
    syn_self_mask = np.ones((max_num_points, max_num_points), dtype=np.float32)

    # 记录每个房间在角点序列中的 [start, end) 范围
    corner_bounds = []
    num_points = 0

    for r_idx in range(num_rooms):
        k = int(room_corner_nums[r_idx]) # 房间拐点数量

        start = num_points
        end = num_points + k
        corner_bounds.append([start, end])
        num_points = end

        rtype_oh = get_one_hot(room_types[r_idx], 25)
        room_index_oh = get_one_hot(r_idx + 1, 32)

        for j in range(k):
            p = start + j
            syn_room_types[p] = rtype_oh
            syn_corner_indices[p] = get_one_hot(j, 32)
            syn_room_indices[p] = room_index_oh
            syn_src_key_padding_mask[p] = 0
            syn_room_areas[p, 0] = float(room_area_rate[r_idx])

            nxt = start + (j + 1) % k
            syn_connections[p] = np.array([p, nxt], dtype=np.int64)

        # self_mask: 同一房间内部角点之间保持 0，其余为 1
        syn_self_mask[start:end, start:end] = 0

    # gen_mask: 有效角点之间为 0，其余为 1
    syn_gen_mask[:num_points, :num_points] = 0

    # 3. 构造房间级 syn_graph（[u, rel, v]），rel=1 表示通过门相连，否则为 -1
    connection_set = {(int(u), int(v)) for u, v in room_connections}
    connection_set |= {(v, u) for u, v in room_connections}

    triples = []
    for u in range(num_rooms):
        for v in range(u + 1, num_rooms):
            rel = 1 if (u, v) in connection_set else -1
            triples.append([u, rel, v])
    syn_graph = np.array(triples, dtype=np.int64)
    if len(syn_graph) < 200:
        pad = np.zeros((200 - len(syn_graph), 3), dtype=np.int64)
        syn_graph = np.concatenate([syn_graph, pad], 0)
    else:
        syn_graph = syn_graph[:200]

    # 4. 根据房间连接关系构造 syn_door_mask
    for u in range(num_rooms):
        for v in range(num_rooms):
            if u == v:
                continue
            if (u, v) in connection_set:
                bu0, bu1 = corner_bounds[u]
                bv0, bv1 = corner_bounds[v]
                syn_door_mask[bu0:bu1, bv0:bv1] = 0.0

    # 5. 将 syn_* 更新进 cond，保持 eval 分支使用的键名
    cond.update({
        'syn_door_mask': syn_door_mask,
        'syn_self_mask': syn_self_mask,
        'syn_gen_mask': syn_gen_mask,
        'syn_room_types': syn_room_types,
        'syn_corner_indices': syn_corner_indices,
        'syn_room_indices': syn_room_indices,
        'syn_src_key_padding_mask': syn_src_key_padding_mask,
        'syn_connections': syn_connections,
        'syn_room_areas': syn_room_areas,
        'syn_graph': syn_graph,
    })

    # 处理 arr
    arr = th.from_numpy(arr).float()
    arr = arr.unsqueeze(0)  # (C, N) → (1, C, N)

    # 处理 cond 里所有值
    for key in cond:
        val = cond[key]
        # numpy → tensor
        if isinstance(val, np.ndarray):
            if val.dtype in [np.int64, np.int32]:
                val = th.from_numpy(val).long()
            else:
                val = th.from_numpy(val).float()
        # 增加 batch 维度
        cond[key] = val.unsqueeze(0)

    return arr, cond

def _load_single_rplan_json_to_model_input(
    rplan_json_path,
    analog_bit=False,
    set_name='eval',
    max_num_points=100,
):
    """
    仅读取并处理单个 rplan_json（raster_to_json.py 输出），生成真实 (arr, cond)。
    该路径用于加速用户自定义输入：避免加载整个 processed_rplan 数据集。
    """
    # 创建一个“轻量”对象来复用类方法（不走 __init__）
    ds = RPlanhgDataset.__new__(RPlanhgDataset)
    ds.set_name = set_name
    ds.analog_bit = analog_bit
    ds.num_coords = 2
    ds.max_num_points = max_num_points

    name, rms_type, fp_eds, rms_bbs, eds_to_rms = reader(rplan_json_path)

    # build input graph
    graph_nodes, graph_edges, rooms_mks = ds.build_graph(rms_type, fp_eds, eds_to_rms)

    # # ✅ 统一房间类型映射（和训练一致）
    # graph_nodes = np.array([
    #     {15: 11, 17: 12, 16: 13}.get(t, t)
    #     for t in graph_nodes
    # ])

    # 根据房间 mask 合并出外轮廓，构造 boundary 输入图
    boundary_mask = np.zeros((256, 256), dtype=np.uint8)
    for room_mask, room_type in zip(rooms_mks, graph_nodes):
        if room_type in [15, 17]:
            continue
        room_mask = room_mask.astype(np.uint8)
        room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
        boundary_mask = cv.bitwise_or(boundary_mask, room_mask)
    contours, _ = cv.findContours(boundary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_boundary = max(contours, key=cv.contourArea)
    input_image = ds.get_input_boundary(contour_boundary[:, 0, :])

    # 构造房间多边形（来自 mask 轮廓）
    house = []
    for room_mask, room_type in zip(rooms_mks, graph_nodes):
        room_mask = room_mask.astype(np.uint8)
        room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
        contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        room_area_rate = cv.contourArea(contours) / (256 * 256)
        house.append([contours[:, 0, :], room_type, room_area_rate])

    # --- 将 house 转成训练/推理一致的 houses[idx] 格式 ---
    corner_bounds = []
    num_points = 0
    house_rows = []

    for i, room in enumerate(house):
        if room[1] > 10:
            room[1] = {15: 11, 17: 12, 16: 13}[room[1]]
        room[0] = np.reshape(room[0], [len(room[0]), 2]) / 256.0 - 0.5
        room[0] = room[0] * 2  # [-1, 1]

        num_room_corners = len(room[0])
        if num_points + num_room_corners > max_num_points:
            raise ValueError(f"Too many corners for max_num_points={max_num_points}")
        # print("房间类型：",room[1])
        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
        area = np.repeat(room[2], num_room_corners)
        area = np.expand_dims(area, 1)
        room_index = np.repeat(np.array([get_one_hot(len(house_rows) + 1, 32)]), num_room_corners, 0)
        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
        padding_mask = np.repeat(1, num_room_corners)
        padding_mask = np.expand_dims(padding_mask, 1)
        connections = np.array([[i, (i + 1) % num_room_corners] for i in range(num_room_corners)])
        connections += num_points

        corner_bounds.append([num_points, num_points + num_room_corners])
        num_points += num_room_corners

        room_row = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections, area), 1)
        house_rows.append(room_row)

    house_layouts = np.concatenate(house_rows, 0)
    padding = np.zeros((max_num_points - len(house_layouts), 94 + 1))
    house_layouts = np.concatenate((house_layouts, padding), 0)

    gen_mask = np.ones((max_num_points, max_num_points))
    gen_mask[:len(house_layouts) - len(padding), :len(house_layouts) - len(padding)] = 0

    door_mask = np.ones((max_num_points, max_num_points))
    self_mask = np.ones((max_num_points, max_num_points))
    for i in range(len(corner_bounds)):
        for j in range(len(corner_bounds)):
            if i == j:
                self_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[j][0]:corner_bounds[j][1]] = 0
            elif any(np.equal([i, 1, j], graph_edges).all(1)) or any(np.equal([j, 1, i], graph_edges).all(1)):
                door_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[j][0]:corner_bounds[j][1]] = 0

    graph = np.concatenate((graph_edges, np.zeros([200 - len(graph_edges), 3])), 0)

    arr = house_layouts[:, :ds.num_coords]
    cond = {
        'door_mask': door_mask,
        'self_mask': self_mask,
        'gen_mask': gen_mask,
        # 'name': name,
        'room_types': house_layouts[:, ds.num_coords:ds.num_coords + 25],
        'corner_indices': house_layouts[:, ds.num_coords + 25:ds.num_coords + 57],
        'room_indices': house_layouts[:, ds.num_coords + 57:ds.num_coords + 89],
        'src_key_padding_mask': 1 - house_layouts[:, ds.num_coords + 89],
        'connections': house_layouts[:, ds.num_coords + 90:ds.num_coords + 92],
        'room_areas': house_layouts[:, ds.num_coords + 92:ds.num_coords + 92 + 1],
        'graph': graph,
        'boundary': input_image,
    }

    if not analog_bit:
        arr = np.transpose(arr, [1, 0])
        return arr.astype(float), cond
    else:
        ONE_HOT_RES = 256
        arr_onehot = np.zeros((ONE_HOT_RES * 2, arr.shape[1])) - 1
        xs = ((arr[:, 0] + 1) * (ONE_HOT_RES / 2)).astype(int)
        ys = ((arr[:, 1] + 1) * (ONE_HOT_RES / 2)).astype(int)
        xs = np.array([get_bin(x, 8) for x in xs])
        ys = np.array([get_bin(x, 8) for x in ys])
        arr_onehot = np.concatenate([xs, ys], 1)
        arr_onehot = np.transpose(arr_onehot, [1, 0])
        arr_onehot[arr_onehot == 0] = -1
        return arr_onehot.astype(float), cond

def make_non_manhattan(poly, polygon, house_poly):
    dist = abs(poly[2]-poly[0])
    direction = np.argmin(dist)
    center = poly.mean(0)
    min = poly.min(0)
    max = poly.max(0)

    tmp = np.random.randint(3, 7)
    new_min_y = center[1]-(max[1]-min[1])/tmp
    new_max_y = center[1]+(max[1]-min[1])/tmp
    if center[0]<128:
        new_min_x = min[0]-(max[0]-min[0])/np.random.randint(2,5)
        new_max_x = center[0]
        poly1=[[min[0], min[1]], [new_min_x, new_min_y], [new_min_x, new_max_y], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]]]
    else:
        new_min_x = center[0]
        new_max_x = max[0]+(max[0]-min[0])/np.random.randint(2,5)
        poly1=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [new_max_x, new_max_y], [new_max_x, new_min_y], [max[0], min[1]]]

    new_min_x = center[0]-(max[0]-min[0])/tmp
    new_max_x = center[0]+(max[0]-min[0])/tmp
    if center[1]<128:
        new_min_y = min[1]-(max[1]-min[1])/np.random.randint(2,5)
        new_max_y = center[1]
        poly2=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]], [new_max_x, new_min_y], [new_min_x, new_min_y]]
    else:
        new_min_y = center[1]
        new_max_y = max[1]+(max[1]-min[1])/np.random.randint(2,5)
        poly2=[[min[0], min[1]], [min[0], max[1]], [new_min_x, new_max_y], [new_max_x, new_max_y], [max[0], max[1]], [max[0], min[1]]]
    p1 = gm.Polygon(poly1)
    iou1 = house_poly.intersection(p1).area/ p1.area
    p2 = gm.Polygon(poly2)
    iou2 = house_poly.intersection(p2).area/ p2.area
    if iou1>0.9 and iou2>0.9:
        return poly
    if iou1<iou2:
        return poly1
    else:
        return poly2

get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]
class RPlanhgDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False):
        super().__init__()
        base_dir = '../dataprocess'
        self.non_manhattan = non_manhattan
        self.set_name = set_name
        self.analog_bit = analog_bit
        self.target_set = target_set
        self.subgraphs = []
        self.org_graphs = []
        self.org_houses = []
        self.names = []
        max_num_points = 100
        if self.set_name == 'eval':
            cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item() # 随机拐点数量
        if os.path.exists(f'processed_rplan/rplan_{set_name}_{target_set}.npz'):
            data = np.load(f'processed_rplan/rplan_{set_name}_{target_set}.npz', allow_pickle=True)
            self.graphs = data['graphs']
            self.houses = data['houses']
            self.door_masks = data['door_masks']
            self.self_masks = data['self_masks']
            self.gen_masks = data['gen_masks']
            self.num_coords = 2
            self.max_num_points = max_num_points
            self.boundarys = data['boundarys']
            self.names = data['names']
            cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
            if self.set_name == 'eval':
                data = np.load(f'processed_rplan/rplan_{set_name}_{target_set}_syn.npz', allow_pickle=True)
                self.syn_graphs = data['graphs']
                self.syn_houses = data['houses']
                self.syn_door_masks = data['door_masks']
                self.syn_self_masks = data['self_masks']
                self.syn_gen_masks = data['gen_masks']
        else:
            with open(f'{base_dir}/list.txt') as f:
                lines = f.readlines()
            cnt=0
            for line in tqdm(lines):
                cnt=cnt+1
                file_name = f'{base_dir}/rplan_json/{line[:-1]}'
                name, rms_type, fp_eds, rms_bbs, eds_to_rms = reader(file_name)
                fp_size = len([x for x in rms_type if x != 15 and x != 17])
                if self.set_name=='train' and fp_size == target_set:
                        continue
                if self.set_name=='eval' and fp_size != target_set:
                        continue
                # a = [rms_type, rms_bbs, fp_eds, eds_to_rms,boundary_cor,input_image]
                a = [name, rms_type, rms_bbs, fp_eds, eds_to_rms]
                self.subgraphs.append(a)
            for graph in tqdm(self.subgraphs):
                name = graph[0]
                rms_type = graph[1]
                rms_bbs = graph[2]
                fp_eds = graph[3]
                eds_to_rms = graph[4]
                # boundary_cor= graph[4]
                # input_image= graph[5]

                rms_bbs = np.array(rms_bbs)
                fp_eds = np.array(fp_eds)
                # boundary_cor = np.array(boundary_cor)

                # extract boundary box and centralize
                tl = np.min(rms_bbs[:, :2], 0)
                br = np.max(rms_bbs[:, 2:], 0)
                shift = (tl+br)/2.0 - 0.5
                rms_bbs[:, :2] -= shift
                rms_bbs[:, 2:] -= shift
                fp_eds[:, :2] -= shift
                fp_eds[:, 2:] -= shift
                # boundary[:, :2] -= shift
                tl -= shift
                br -= shift

                # build input graph 房间类型、房间之间是否连通、每个房间的掩码图
                graph_nodes, graph_edges, rooms_mks = self.build_graph(rms_type, fp_eds, eds_to_rms) # 房间类型、u和v是否连接(1,-1),每个房间的像素掩码

                house = []
                # TODO 遍历房间掩码，根据房间掩码构造边界
                boundary_mask = np.zeros((256, 256), dtype=np.uint8)
                for room_mask, room_type in zip(rooms_mks, graph_nodes):
                    if room_type in [15,17]:continue
                    room_mask = room_mask.astype(np.uint8)
                    room_mask = cv.resize(room_mask, (256, 256), interpolation=cv.INTER_AREA)
                    # 合并到 boundary_mask 中（取并集）
                    boundary_mask = cv.bitwise_or(boundary_mask, room_mask)
                contours, _ = cv.findContours(boundary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                contour_boundary = max(contours, key=cv.contourArea) # 边界拐点坐标
                input_image = self.get_input_boundary(contour_boundary[:,0,:]) # 边界特征 [3,256,256]
                # contour_boundary = boundary_cor[:, :2].reshape(-1, 1, 2)


                boundary_area = cv.contourArea(contour_boundary) # 边界面积
                for room_mask, room_type in zip(rooms_mks, graph_nodes):
                    room_mask = room_mask.astype(np.uint8)
                    room_mask = cv.resize(room_mask, (256, 256), interpolation = cv.INTER_AREA)
                    contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    contours = contours[0]
                    room_area_rate = cv.contourArea(contours)/ (256*256)
                    # print(room_area_rate)
                    house.append([contours[:,0,:], room_type,room_area_rate]) # 房间拐点坐标、房间类型、房间面积占边界面积的比例
                self.org_graphs.append(graph_edges) # 房间之间的连通情况
                self.org_houses.append([name, house, input_image])
                # self.boundarys.append(input_image)
            houses = []
            boundarys = []
            door_masks = []
            self_masks = []
            gen_masks = []
            graphs = []
            names = []
            if self.set_name=='train':
                cnumber_dist = defaultdict(list)

            if self.non_manhattan:
                for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                    # Generating non-manhattan Balconies
                    tmp = []
                    for i, room in enumerate(h[1]):
                        if room[1]>10:
                            continue
                        if len(room[0])!=4: 
                            continue
                        if np.random.randint(2):
                            continue
                        poly = gm.Polygon(room[0])
                        house_polygon = unary_union([gm.Polygon(room[0]) for room in h])
                        room[0] = make_non_manhattan(room[0], poly, house_polygon)

            for oh, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                name = oh[0]
                h = oh[1]
                b = oh[2]
                house = []
                corner_bounds = []
                num_points = 0
                for i, room in enumerate(h):
                    if room[1]>10:
                        room[1] = {15:11, 17:12, 16:13}[room[1]]
                    room[0] = np.reshape(room[0], [len(room[0]), 2])/256. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                    room[0] = room[0] * 2 # map to [-1, 1]
                    # room[2] = int((room[2] * 100 +1.99)/2) # 面积占边界面积的比例，这个需要限制在30内
                    # room[2] = np.clip(room[2], a_min=0, a_max=29)  # 限制在 [0, 30] 范围内
                    if self.set_name=='train':
                        cnumber_dist[room[1]].append(len(room[0]))
                    # Adding conditions
                    num_room_corners = len(room[0])
                    rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)

                    area = np.repeat(room[2], num_room_corners) # 重复拐点长度次
                    area = np.expand_dims(area, 1)
                    room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                    corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                    # Src_key_padding_mask
                    padding_mask = np.repeat(1, num_room_corners)
                    padding_mask = np.expand_dims(padding_mask, 1)
                    # Generating corner bounds for attention masks
                    connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                    connections += num_points
                    corner_bounds.append([num_points, num_points+num_room_corners])
                    num_points += num_room_corners
                    # [100,2]、[100,25]、[100,32]、[100,32]、[100,1]->哪些拐点是有效的、[100,2]->哪些拐点之间是连通的、[100,1]->房间面积占边界面积的比例
                    room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections,area), 1) # 对列进行拼接 2+25+32+32+1+2
                    house.append(room)

                house_layouts = np.concatenate(house, 0)
                if len(house_layouts)>max_num_points:
                    continue
                padding = np.zeros((max_num_points-len(house_layouts), 94+1)) # +30是面积
                gen_mask = np.ones((max_num_points, max_num_points))
                gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                house_layouts = np.concatenate((house_layouts, padding), 0) # 对行进行补全

                door_mask = np.ones((max_num_points, max_num_points))
                self_mask = np.ones((max_num_points, max_num_points))
                for i in range(len(corner_bounds)):
                    for j in range(len(corner_bounds)):
                        if i==j:
                            self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                        elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                houses.append(house_layouts)
                door_masks.append(door_mask)
                self_masks.append(self_mask)
                gen_masks.append(gen_mask)
                graph = np.concatenate((graph, np.zeros([200 - len(graph), 3])), 0)
                graphs.append(graph)
                boundarys.append(b)
                names.append(name)
            self.max_num_points = max_num_points
            self.houses = houses
            self.door_masks = door_masks
            self.self_masks = self_masks
            self.gen_masks = gen_masks
            self.num_coords = 2
            self.graphs = graphs
            self.boundarys = boundarys
            self.names = np.array(names, dtype=object)

            np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}', graphs=self.graphs, houses=self.houses,
                    door_masks=self.door_masks, self_masks=self.self_masks, gen_masks=self.gen_masks, boundarys=self.boundarys, names=self.names)
            if self.set_name=='train':
                np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_cndist', cnumber_dist=cnumber_dist)

            if set_name=='eval':
                houses = []
                graphs = []
                door_masks = []
                self_masks = []
                gen_masks = []
                len_house_layouts = 0
                for oh, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                    h = oh[1]
                    house = []
                    corner_bounds = []
                    num_points = 0
                    num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    # num_room_corners_total = []
                    # for room in h:
                    #     if isinstance(room, (list, tuple)) and len(room) >= 2:
                    #         num_room_corners_total.append(
                    #             cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]]) - 1)])

                    while np.sum(num_room_corners_total)>=max_num_points:
                        num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    for i, room in enumerate(h):
                        # Adding conditions
                        num_room_corners = num_room_corners_total[i]
                        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
                        area = np.repeat(room[2], num_room_corners) # 重复拐点长度次
                        area = np.expand_dims(area, 1)

                        room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                        # Src_key_padding_mask
                        padding_mask = np.repeat(1, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                        # Generating corner bounds for attention masks
                        connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                        connections += num_points
                        corner_bounds.append([num_points, num_points+num_room_corners])
                        num_points += num_room_corners
                        room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index, padding_mask, connections,area), 1)
                        house.append(room)

                    house_layouts = np.concatenate(house, 0)
                    if np.sum([len(room[0]) for room in h])>max_num_points:
                        continue
                    padding = np.zeros((max_num_points-len(house_layouts), 94+1))
                    gen_mask = np.ones((max_num_points, max_num_points))
                    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                    house_layouts = np.concatenate((house_layouts, padding), 0)

                    door_mask = np.ones((max_num_points, max_num_points))
                    self_mask = np.ones((max_num_points, max_num_points))
                    for i, room in enumerate(h):
                        if room[1]==1:
                            living_room_index = i
                            break
                    for i in range(len(corner_bounds)):
                        is_connected = False
                        for j in range(len(corner_bounds)):
                            if i==j:
                                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                            elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                                door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                                is_connected = True
                        if not is_connected:
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0
                    graph = np.concatenate((graph, np.zeros([200 - len(graph), 3])), 0)
                    houses.append(house_layouts)
                    door_masks.append(door_mask)
                    self_masks.append(self_mask)
                    gen_masks.append(gen_mask)
                    graphs.append(graph)
                self.syn_houses = houses
                self.syn_door_masks = door_masks
                self.syn_self_masks = self_masks
                self.syn_gen_masks = gen_masks
                self.syn_graphs = graphs
                np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_syn', graphs=self.syn_graphs, houses=self.syn_houses,
                        door_masks=self.syn_door_masks, self_masks=self.syn_self_masks, gen_masks=self.syn_gen_masks)

    def __len__(self):
        return len(self.houses)

    def _rotate_points_image(self,arr, size=256,rotation=0):
        rotation_matrices = {
            0: np.array([[1, 0],
                         [0, 1]]),  # 0° (单位矩阵)
            1: np.array([[0, -1],
                         [1, 0]]),  # 90° 逆时针
            2: np.array([[-1, 0],
                         [0, -1]]),  # 180°
            3: np.array([[0, 1],
                         [-1, 0]])  # 90° 顺时针
        }
        arr_rot =[]
        for i in range(len(arr)):
            center = (size - 1) / 2.0  # 255→中心在127.5

            # 平移点到中心 → 旋转 → 平移回去
            arr_centered = arr[i] - center
            arr_rot.append((arr_centered @ rotation_matrices[rotation].T) + center)

        for i in range(len(arr_rot)):
            # 保证坐标落回 [0, size-1]
            arr_rot[i] = np.clip(np.round(arr_rot), 0, size - 1).astype(int)
        return arr_rot

    def __getitem__(self, idx):
        # idx = int(idx//20)
        arr = self.houses[idx][:, :self.num_coords]
        boundary = self.boundarys[idx].copy() # 5,256,256
        # print("boundary shape:",boundary.shape)
        graph = np.concatenate((self.graphs[idx], np.zeros([200-len(self.graphs[idx]), 3])), 0)

        cond = {
                'door_mask': self.door_masks[idx],
                'self_mask': self.self_masks[idx],
                'gen_mask': self.gen_masks[idx],
                'room_types': self.houses[idx][:, self.num_coords:self.num_coords+25],
                'corner_indices': self.houses[idx][:, self.num_coords+25:self.num_coords+57],
                'room_indices': self.houses[idx][:, self.num_coords+57:self.num_coords+89],
                'src_key_padding_mask': 1-self.houses[idx][:, self.num_coords+89],
                'connections': self.houses[idx][:, self.num_coords+90:self.num_coords+92],
                'room_areas':self.houses[idx][:, self.num_coords+92:self.num_coords+92+1],
                'graph': graph,
                'boundary': boundary,
                }
        if self.set_name == 'eval':
            syn_graph = np.concatenate((self.syn_graphs[idx], np.zeros([200-len(self.syn_graphs[idx]), 3])), 0)
            assert (graph == syn_graph).all(), idx
            cond.update({
                # Transform 的三个注意力掩码：门和房间、房间和自己、房间和房间
                'syn_door_mask': self.syn_door_masks[idx],
                'syn_self_mask': self.syn_self_masks[idx],
                'syn_gen_mask': self.syn_gen_masks[idx],
                'syn_room_types': self.syn_houses[idx][:, self.num_coords:self.num_coords+25],
                'syn_corner_indices': self.syn_houses[idx][:, self.num_coords+25:self.num_coords+57],
                'syn_room_indices': self.syn_houses[idx][:, self.num_coords+57:self.num_coords+89],
                'syn_src_key_padding_mask': 1-self.syn_houses[idx][:, self.num_coords+89],
                'syn_connections': self.syn_houses[idx][:, self.num_coords+90:self.num_coords+92],
                'syn_room_areas': self.houses[idx][:, self.num_coords + 92:self.num_coords + 92 + 1],
                'syn_graph': syn_graph,
                })

        if not self.analog_bit:
            arr = np.transpose(arr, [1, 0])
            return arr.astype(float), cond
        else:
            ONE_HOT_RES = 256
            arr_onehot = np.zeros((ONE_HOT_RES*2, arr.shape[1])) - 1
            xs = ((arr[:, 0]+1)*(ONE_HOT_RES/2)).astype(int)
            ys = ((arr[:, 1]+1)*(ONE_HOT_RES/2)).astype(int)
            xs = np.array([get_bin(x, 8) for x in xs])
            ys = np.array([get_bin(x, 8) for x in ys])
            arr_onehot = np.concatenate([xs, ys], 1)
            arr_onehot = np.transpose(arr_onehot, [1, 0])
            arr_onehot[arr_onehot==0] = -1
            return arr_onehot.astype(float), cond

    def make_sequence(self, edges):
        polys = []
        v_curr = tuple(edges[0][:2])
        e_ind_curr = 0
        e_visited = [0]
        seq_tracker = [v_curr]
        find_next = False
        while len(e_visited) < len(edges):
            if find_next == False:
                if v_curr == tuple(edges[e_ind_curr][2:]):
                    v_curr = tuple(edges[e_ind_curr][:2])
                else:
                    v_curr = tuple(edges[e_ind_curr][2:])
                find_next = not find_next 
            else:
                # look for next edge
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        if (v_curr == tuple(e[:2])):
                            v_curr = tuple(e[2:])
                            e_ind_curr = k
                            e_visited.append(k)
                            break
                        elif (v_curr == tuple(e[2:])):
                            v_curr = tuple(e[:2])
                            e_ind_curr = k
                            e_visited.append(k)
                            break

            # extract next sequence
            if v_curr == seq_tracker[-1]:
                polys.append(seq_tracker)
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        v_curr = tuple(edges[0][:2])
                        seq_tracker = [v_curr]
                        find_next = False
                        e_ind_curr = k
                        e_visited.append(k)
                        break
            else:
                seq_tracker.append(v_curr)
        polys.append(seq_tracker)

        return polys

    def build_graph(self, rms_type, fp_eds, eds_to_rms, out_size=64):
        # create edges
        triples = []
        nodes = rms_type 
        # encode connections
        for k in range(len(nodes)):
            for l in range(len(nodes)):
                if l > k:
                    is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                    if is_adjacent:
                        if 'train' in self.set_name:
                            triples.append([k, 1, l]) # 这里表示的是房间之间的相邻关系，而不是房间和门之间的关系
                        else:
                            triples.append([k, 1, l])
                    else:
                        if 'train' in self.set_name:
                            triples.append([k, -1, l])
                        else:
                            triples.append([k, -1, l])
        # get rooms masks
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):                  
            eds_to_rms_tmp.append([eds_to_rms[l][0]])
        rms_masks = []
        im_size = 256
        fp_mk = np.zeros((out_size, out_size))
        for k in range(len(nodes)):
            # add rooms and doors
            eds = []
            for l, e_map in enumerate(eds_to_rms_tmp):
                if (k in e_map):
                    eds.append(l)
            # draw rooms
            rm_im = Image.new('L', (im_size, im_size))
            dr = ImageDraw.Draw(rm_im)
            for eds_poly in [eds]:
                poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
                poly = [(im_size*x, im_size*y) for x, y in poly]
                if len(poly) >= 2:
                    dr.polygon(poly, fill='white')
                else:
                    print("Empty room")
                    exit(0)
            rm_im = rm_im.resize((out_size, out_size)) ############## TODO
            rm_arr = np.array(rm_im)
            inds = np.where(rm_arr>0)
            rm_arr[inds] = 1.0
            rms_masks.append(rm_arr)
            if rms_type[k] != 15 and rms_type[k] != 17:
                fp_mk[inds] = k+1
        # trick to remove overlap
        for k in range(len(nodes)):
            if rms_type[k] != 15 and rms_type[k] != 17:
                rm_arr = np.zeros((out_size, out_size))
                inds = np.where(fp_mk==k+1)
                rm_arr[inds] = 1.0
                rms_masks[k] = rm_arr
        # convert to array
        nodes = np.array(nodes)
        triples = np.array(triples)
        rms_masks = np.array(rms_masks)
        return nodes, triples, rms_masks

    def get_input_boundary(self,boundary_cor):
        external = boundary_cor.astype(np.int32) # [:, :2]
        # door = boundary_cor[:2, :2]

        boundary = np.zeros((256, 256), dtype=float)
        inside = np.zeros((256, 256), dtype=float)
        global_mask = np.zeros((256, 256), dtype=float)

        pts = np.concatenate([external, external[:1]])
        # pts = np.concatenate([external])
        # print(pts.shape)
        # pts_door = door

        cv.fillPoly(inside, pts.reshape(1, -1, 2), 1.0)  # 将边界内部的区域用color=1填充
        cv.polylines(boundary, pts.reshape(1, -1, 2), True, 0.5, 3)  # 将边界区域给画出来
        # cv2.polylines(boundary, pts_door.reshape(1, -1, 2), False, 0.25, 3) # 将前门给画出来
        cv.fillPoly(global_mask, pts.reshape(1, -1, 2), 1.0)  # 将边界内部的区域用color=1填充
        cv.polylines(global_mask, pts.reshape(1, -1, 2), True, 0.5, 3)  # 将边界区域给画出来
        # cv2.polylines(global_mask, pts_door.reshape(1, -1, 2), False, 0.25, 3)  # 将边界区域给画出来
        # _, indices = distance_transform_edt(1 - inside, return_indices=True) # 2,H,W


        input_image = np.stack([inside, boundary, global_mask], -1)  # 将以上的三个堆在一起
        input_image = input_image.transpose(2, 0, 1).astype(np.float32)
        # input_image = np.concatenate([input_image, indices], 0)  # 将以上的三个堆在一起
        return input_image

def is_adjacent(box_a, box_b, threshold=0.03):
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b
    h1, h2 = x1-x0, x3-x2
    w1, w2 = y1-y0, y3-y2
    xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
    yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0
    delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
    delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0
    delta = max(delta_x, delta_y)
    return delta < threshold

def reader(filename):
    with open(filename) as f:
        info =json.load(f)
        name = info.get('name', os.path.splitext(os.path.basename(filename))[0])
        rms_bbs=np.asarray(info['boxes'])
        fp_eds=info['edges']
        rms_type=info['room_type']
        eds_to_rms=info['ed_rm']
        s_r=0
        for rmk in range(len(rms_type)):
            if(rms_type[rmk]!=17):
                s_r=s_r+1   
        rms_bbs = np.array(rms_bbs)/256.0
        fp_eds = np.array(fp_eds)/256.0
        fp_eds = fp_eds[:, :4]
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl+br)/2.0 - 0.5 # 正中心和[0.5,0.5]的差距
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        tl -= shift
        br -= shift


        return name, rms_type, fp_eds, rms_bbs, eds_to_rms


def test_load_demo_to_model_input():
    # Ensure relative paths inside rplanhg_datasets.py resolve correctly.
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, ".."))
    os.chdir(project_root)

    demo_path = os.path.join(this_dir, "demo.json")

    # Patch1 sanity: should only require the single real json under dataprocess/rplan_json/{name}.json
    with open(demo_path) as f:
        demo = __import__("json").load(f)
    name = str(demo["name"])
    real_json = os.path.join(project_root, "dataprocess", "rplan_json", f"{name}.json")
    if not os.path.exists(real_json):
        raise FileNotFoundError(f"Missing real sample json required by patch1: {real_json}")

    arr, cond = load_demo_to_model_input(
        demo_json_path=demo_path,
        analog_bit=False,
        target_set=8,
        set_name="eval",
    )
    print(arr,cond)

    print("arr shape:", arr.shape)
    print("cond shape:", cond.shape)
    print("syn_graph shape:", cond["syn_graph"].shape)

if __name__ == '__main__':
    # dataset = RPlanhgDataset('eval', False, 8)

    data = load_rplanhg_data(
        batch_size=2,
        analog_bit=False,
        target_set=8,
        set_name="eval",
    )
    data_sample, model_kwargs = next(data)
    print(data_sample)
    print(model_kwargs)
