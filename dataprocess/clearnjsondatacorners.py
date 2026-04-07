import json
import numpy as np
from glob import glob
import os
import cv2 as cv

filter_num = 0
door_no_two_rooms_count = 0

def is_valid_file(filename, max_corners=32):
    global filter_num
    global door_no_two_rooms_count
    try:
        with open(filename) as f:
            info = json.load(f)

        # 获取基础字段 # 类型、每一个墙的起点和重点、房间盒子、每一条边对应的房间索引、每一条边对应的第一个房间索引、边界
        rms_bbs = np.array(info.get('boxes', []))
        fp_eds = np.array(info.get('edges', [])) # 每一个墙的起点和终点
        rms_type = info.get('room_type', [])
        eds_to_rms = info.get('ed_rm', []) # 每一条边对应的房间索引
        # print(rms_type)
        door_num = len([t for t in rms_type if t in (15, 17)])
        total_num = len(rms_type)
        # print(door_num,total_num)
        if ((door_num*2)!=total_num):  # 去掉门和房间不匹配的数据
            filter_num +=1
            return False
            # print(filename," is not a valid file")

        # 3. 第二步：找到所有门的边，并按“4条边一组”分组（核心适配逻辑）
        door_edge_indices = []  # 存储所有门的边的索引（在edges中的位置）
        for edge_idx, edge in enumerate(fp_eds):
            if len(edge) >= 5 and edge[4] == 17:  # 找到类型17的门的边
                door_edge_indices.append(edge_idx)
        # 校验：门的边总数必须是4的倍数（每个门4条边），否则文件无效
        if len(door_edge_indices) != (door_num-1) * 4:
            door_no_two_rooms_count += 1
            return False

        # 4. 第三步：按“4条边一组”校验每个门是否连接两个房间
        # 遍历每一组门的边（每组4条边，对应1个门）
        for i in range(door_num-1):
            # 获取当前门的4条边的索引（从door_edge_indices中截取）
            group_start = i * 4
            group_end = group_start + 4
            door_group_indices = door_edge_indices[group_start:group_end]

            # 收集当前门4条边关联的所有房间索引（去重，避免重复统计）
            group_room_indices = set()
            for edge_idx in door_group_indices:
                if edge_idx >= len(eds_to_rms):
                    # 某条边没有对应的房间索引，整组门无效
                    door_no_two_rooms_count += 1
                    return False
                # 将当前边的房间索引加入集合（自动去重）
                group_room_indices.update(eds_to_rms[edge_idx])
            # print(len(group_room_indices),group_room_indices)

            # 校验：整组门必须关联且仅关联2个不同房间（门连接两个房间的核心条件）
            if len(group_room_indices) != 3:
                print(filename)
                door_no_two_rooms_count += 1
                return False

        # TODO 门的交集不行
        # 基本结构检查
        if len(fp_eds.shape) != 2 or fp_eds.shape[1] < 4 or len(fp_eds) == 0:
            return False
        if len(eds_to_rms) != len(fp_eds):
            return False

        # 位移归一化（因为绘图需要）
        rms_bbs = rms_bbs / 256.0
        fp_eds = fp_eds[:, :4] / 256.0
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl + br) / 2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift

        # 构建 room masks，用于角点数检查
        out_size = 64
        im_size = 256
        nodes = rms_type
        eds_to_rms_tmp = [[e[0]] for e in eds_to_rms]

        for k in range(len(nodes)):
            eds = [l for l, e_map in enumerate(eds_to_rms_tmp) if k in e_map]
            if not eds:
                continue
            rm_im = np.zeros((im_size, im_size), dtype=np.uint8)
            pts = []
            for l in eds:
                edge = fp_eds[l]
                pts.append(((edge[0]*im_size, edge[1]*im_size), (edge[2]*im_size, edge[3]*im_size)))
            img = np.zeros((im_size, im_size), dtype=np.uint8)
            for p1, p2 in pts:
                cv.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, 1)
            contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                if contour.shape[0] > max_corners or contour.shape[0] < 4:
                    return False

        return True

    except Exception as e:
        # 文件结构或处理异常都认为是非法文件
        return False

# 清洗目录
file_list = glob('./rplan_json/*')
# filter_num = [0]
# 删除不符合要求的文件
for file_path in file_list:
    if not is_valid_file(file_path):
        try:
            pass
            os.remove(file_path)
            print(f"Deleted invalid file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
print(filter_num)
print(door_no_two_rooms_count)