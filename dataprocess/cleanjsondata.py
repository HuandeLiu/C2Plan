import json
import numpy as np
from glob import glob


def reader(filename):
    with open(filename) as f:
        info = json.load(f)
    rms_bbs = np.asarray(info['boxes'])
    fp_eds = info['edges']
    rms_type = info['room_type']
    eds_to_rms = info['ed_rm']
    # boundary = info['boundary']
    s_r = 0
    for rmk in range(len(rms_type)):
        if rms_type[rmk] != 17:
            s_r = s_r + 1
    rms_bbs = np.array(rms_bbs) / 256.0
    fp_eds = np.array(fp_eds) / 256.0
    fp_eds = fp_eds[:, :4]
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl + br) / 2.0 - 0.5
    rms_bbs[:, :2] -= shift
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift
    tl -= shift
    br -= shift
    eds_to_rms_tmp = []

    for l in range(len(eds_to_rms)):
        eds_to_rms_tmp.append([eds_to_rms[l][0]])

    return rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp #,boundary # 类型、每一个墙的起点和重点、房间盒子、每一条边对应的房间索引、每一条边对应的第一个房间索引、边界


file_list = glob('./rplan_json/*')
# with open('file_list.txt','r') as f:
#     lines = f.readlines()

lines = file_list

# out_size = 64
length_edges = []
subgraphs = []
for line in lines:
    a = []
    with open(line) as f2:
        rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp,boundary = reader(line) # 类型、每一个墙的起点和重点、房间盒子、每一条边对应的房间索引、每一条边对应的第一个房间索引、边界

    eds_to_rms_tmp = []
    for l in range(len(eds_to_rms)):
        eds_to_rms_tmp.append([eds_to_rms[l][0]])

    rms_masks = []
    im_size = 256
    # fp_mk = np.zeros((out_size, out_size))
    nodes = rms_type
    for k in range(len(nodes)): # 遍历每个房间，提取关联的墙体线段
        eds = [] # 存储当前房间关联的墙体线段索引
        # 查找所有与当前房间（k）关联的线段
        for l, e_map in enumerate(eds_to_rms_tmp):
            if (k in e_map):
                eds.append(l) # 找到房间的所有边
        b = []
        for eds_poly in [eds]:
            length_edges.append((line, np.array([fp_eds[l][:4] for l in eds_poly]))) # # 提取线段坐标（fp_eds[l][:4]即x1,y1,x2,y2）
# 检查所有线段数据的形状是否有效（必须是二维数组，即(线段数量, 4)）
chk = [x[1].shape for x in length_edges] # 提取每个线段数组的形状 可能是4 5  6 7
idx = [i for i, x in enumerate(chk) if len(x) != 2] # 找出形状无效的索引（非二维数组的情况） 只有奇数条边
final = [length_edges[i][0] for i in idx] # 记录文件名
final = [x.replace('\n', '') for x in final] # 添加换行

import os
# 删除所有无效文件
for fin in final:
    try:
        os.remove(fin)
        print(f"Deleted invalid file: {fin}")
    except:
        pass