#!/usr/bin/env python3
"""
边界图导出脚本 - 与rplanhg_datasets.py完全一致
从json数据导出房间边界图并保存为png格式
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path

# 添加模型路径以便导入相关函数
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

# 只导入需要的函数，避免导入依赖
def reader(filename):
    """复制rplanhg_datasets.py中的reader函数"""
    import json
    import numpy as np
    import os

    with open(filename) as f:
        info = json.load(f)
        name = info.get('name', os.path.splitext(os.path.basename(filename))[0])
        rms_bbs = np.asarray(info['boxes'])
        fp_eds = info['edges']
        rms_type = info['room_type']
        eds_to_rms = info['ed_rm']
        s_r = 0
        for rmk in range(len(rms_type)):
            if(rms_type[rmk] != 17):
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

        return name, rms_type, fp_eds, rms_bbs, eds_to_rms

def get_one_hot(x, z):
    """复制rplanhg_datasets.py中的get_one_hot函数"""
    import numpy as np
    return np.eye(z)[x]

def get_bin(x, z):
    """复制rplanhg_datasets.py中的get_bin函数"""
    return [int(y) for y in format(x, 'b').zfill(z)]

def extract_boundary_from_json(json_path):
    """
    完全按照RPlanhgDataset类的方式提取边界
    参考RPlanhgDataset.__init__中的处理逻辑
    """
    # 使用reader函数读取json文件（与RPlanhgDataset一致）
    name, rms_type, fp_eds, rms_bbs, eds_to_rms = reader(json_path)

    # 转换为numpy数组
    rms_bbs = np.array(rms_bbs)
    fp_eds = np.array(fp_eds)

    # 中心化处理（与reader函数中一致）
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl + br) / 2.0 - 0.5
    rms_bbs[:, :2] -= shift
    rms_bbs[:, 2:] -= shift
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift

    # 创建RPlanhgDataset实例来调用其方法
    # 注意：这里我们创建一个简化实例，只为了调用方法
    class SimpleDataset:
        def __init__(self):
            self.set_name = 'eval'

        def make_sequence(self, edges):
            """复制RPlanhgDataset.make_sequence方法"""
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
            """复制RPlanhgDataset.build_graph方法"""
            # create edges
            triples = []
            nodes = rms_type
            # encode connections
            for k in range(len(nodes)):
                for l in range(len(nodes)):
                    if l > k:
                        is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                        if is_adjacent:
                            triples.append([k, 1, l])
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
                    try:
                        poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
                        poly = [(im_size*x, im_size*y) for x, y in poly]
                        if len(poly) >= 2:
                            dr.polygon(poly, fill='white')
                        else:
                            # 如果多边形点数少于2，跳过这个房间
                            continue
                    except Exception as e:
                        # 如果make_sequence失败，跳过这个房间
                        continue
                rm_im = rm_im.resize((out_size, out_size))
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

        def get_input_boundary(self, boundary_cor):
            """复制RPlanhgDataset.get_input_boundary方法"""
            external = boundary_cor.astype(np.int32)

            boundary = np.zeros((256, 256), dtype=float)
            inside = np.zeros((256, 256), dtype=float)
            global_mask = np.zeros((256, 256), dtype=float)

            pts = np.concatenate([external, external[:1]])

            cv2.fillPoly(inside, pts.reshape(1, -1, 2), 1.0)
            cv2.polylines(boundary, pts.reshape(1, -1, 2), True, 0.5, 3)
            cv2.fillPoly(global_mask, pts.reshape(1, -1, 2), 1.0)
            cv2.polylines(global_mask, pts.reshape(1, -1, 2), True, 0.5, 3)

            input_image = np.stack([inside, boundary, global_mask], -1)
            input_image = input_image.transpose(2, 0, 1).astype(np.float32)
            return input_image

    # 创建实例
    ds = SimpleDataset()

    # 构建图（与RPlanhgDataset一致）
    graph_nodes, graph_edges, rooms_mks = ds.build_graph(rms_type, fp_eds, eds_to_rms)

    # 创建边界掩码（与RPlanhgDataset.__init__中一致）
    boundary_mask = np.zeros((256, 256), dtype=np.uint8)
    for room_mask, room_type in zip(rooms_mks, graph_nodes):
        if room_type in [15, 17]:
            continue
        room_mask = room_mask.astype(np.uint8)
        room_mask = cv2.resize(room_mask, (256, 256), interpolation=cv2.INTER_AREA)
        boundary_mask = cv2.bitwise_or(boundary_mask, room_mask)

    # 查找轮廓
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 调试：保存边界掩码
    debug_dir = os.path.join(os.path.dirname(__file__), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    json_name = os.path.splitext(os.path.basename(json_path))[0]
    cv2.imwrite(os.path.join(debug_dir, f"{json_name}_boundary_mask.png"), boundary_mask * 255)

    print(f"找到 {len(contours)} 个轮廓")
    if not contours:
        print(f"警告: {json_path} 未找到轮廓")
        print(f"边界掩码非零像素数: {np.count_nonzero(boundary_mask)}")
        return None

    contour_boundary = max(contours, key=cv2.contourArea)
    print(f"最大轮廓面积: {cv2.contourArea(contour_boundary)}")

    # 生成边界图像
    input_image = ds.get_input_boundary(contour_boundary[:, 0, :])

    return input_image, contour_boundary

def save_boundary_image(boundary, output_path, filename):
    if boundary is None:
        print(f"Error: The boundary data is empty and cannot be saved {filename}")
        return False

    boundary_line = boundary[1]  # The second channel is the boundary line

    boundary_mask = (boundary_line == 0.5).astype(np.uint8)

    img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

    full_path = os.path.join(output_path, filename)
    cv2.imwrite(full_path, img)
    print(f"save boundary: {full_path}")
    return True

def process_json_file(json_path, output_dir):
    """Process a single JSON file"""
    try:
        print(f"Process file: {json_path}")

        boundary_data, contour = extract_boundary_from_json(json_path)
        if boundary_data is None:
            return False

        json_name = Path(json_path).stem
        output_filename = f"{json_name}.png"

        success = save_boundary_image(boundary_data, output_dir, output_filename)

        return success

    except Exception as e:
        print(f"Process file {json_path} erro: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_json_files(input_dir):
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '..', 'dataprocess', 'rplan_json')

    input_files = find_json_files(input_dir)

    output_dir = os.path.join(current_dir, 'photo')
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个文件
    success_count = 0
    for json_file in input_files:
        if process_json_file(json_file, output_dir):
            success_count += 1


if __name__ == "__main__":
    main()