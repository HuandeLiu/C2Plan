"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
# module_path = "/home/cyq/lhd/fooldiff/housediffusion/house_diffusion-main/"
# 将路径添加到 Python 搜索路径
# if module_path not in sys.path:
#     sys.path.append(module_path)
import numpy as np
import torch as th
import cv2 as cv
import io
import PIL.Image as Image
import drawsvg as drawsvg
import cairosvg
import imageio
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 前设置
import matplotlib.pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
from rplanhg_datasets import load_rplanhg_data
import dist_util, logger
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)
import webcolors
import networkx as nx
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos
from shapely.geometry import MultiPoint
import logging
# import random
# th.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

bin_to_int = lambda x: int("".join([str(int(i.cpu().data)) for i in x]), 2)

def convert_svg_folder_to_png(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith(".svg"):
            svg_path = os.path.join(input_folder, file)
            png_path = os.path.join(output_folder, file.replace(".svg", ".png"))
            cairosvg.svg2png(url=svg_path, write_to=png_path)
def bin_to_int_sample(sample, resolution=256):
    sample_new = th.zeros([sample.shape[0], sample.shape[1], sample.shape[2], 2])
    sample[sample<0] = 0
    sample[sample>0] = 1
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(sample.shape[2]):
                sample_new[i, j, k, 0] = bin_to_int(sample[i, j, k, :8])
                sample_new[i, j, k, 1] = bin_to_int(sample[i, j, k, 8:])
    sample = sample_new
    sample = sample/(resolution/2) - 1
    return sample

def estimate_areas(args, sample_gt, sample, model_kwargs):
    # 内部函数：处理单组数据（预测或真实）
    def process_data(data,is_syn=False):
        areas_list = []
        bc_errors = []
        all_iou = []
        all_iou_area = []
        all_hausdorff = []
        bc_errors = []
        prefix = 'syn_' if is_syn else ''
        for i in range(data.shape[1]):  # 第i张图片
            for k in range(data.shape[0]):  # 第k次采样
                polys = []
                types = []
                ids = []
                resolution = 256
                for j, point in enumerate(data[k][i]):
                    # 选择参数键（真实/预测）
                    mask_key = f'{prefix}src_key_padding_mask'
                    if model_kwargs[mask_key][i][j] == 1:
                        continue
                    point = point.cpu().data.numpy()
                    if j == 0:
                        poly = []
                    index_key = f'{prefix}room_indices'
                    if j > 0 and (model_kwargs[index_key][i, j] != model_kwargs[index_key][i, j - 1]).any():
                        polys.append(poly)
                        types.append(c)
                        ids.append(id)
                        poly = []
                    # 坐标转换
                    pred_center = False
                    if pred_center:
                        point = point / 2 + 1
                        point = point * (resolution // 2)
                    else:
                        point = point / 2 + 0.5
                        point = point * resolution
                    poly.append((point[0], point[1]))
                    # 房间类型
                    c = np.argmax(model_kwargs[f'{prefix}room_types'][i][j - 1].cpu().numpy())
                    # 房间索引
                    id = np.argmax(model_kwargs[f'{prefix}room_indices'][i][j - 1].cpu().numpy())
                polys.append(poly) # 所有房间的点
                types.append(c)
                ids.append(id)
                polys_out_door = polys
                # for cl in range(len(types)):
                #     if types[cl] not in [11, 12]:
                #         polys_out_door.append(polys[cl])
                # 计算每个房间的房间面积
                room_areas = []
                for a in range(len(polys_out_door)):
                    p1 = polys_out_door[a]
                    p1 = Polygon(p1)
                    if not p1.is_valid:
                        p1 = geom_factory(lgeos.GEOSMakeValid(p1._geom))
                    room_areas.append({ids[a]: p1.area})
                areas_list.append(room_areas) # 把整张图片放进去
                # 计算凸包和边界面积
                flattened_points = [p for poly in polys_out_door for p in poly]
                room_points = MultiPoint(flattened_points) # if len(flattened_points) >= 3 else 0.0
                # room_edge = Polygon(flattened_points)
                room_hull = room_points.convex_hull # if len(flattened_points) >= 3 else 0.0
                boundary_key = 'boundary'
                boundary_mask = (model_kwargs[boundary_key][i][0] == 1.).cpu().detach().numpy().astype(np.uint8)
                # contours, _ = cv.findContours(boundary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # boundary_contour = contours[0]
                # contours, _ = cv.findContours(boundary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours, _ = cv.findContours(boundary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                boundary_contour = max(contours, key=cv.contourArea)
                # boundary_contour = boundary_contour.squeeze(axis=1)
                boundary_contour = boundary_contour[:,0,:]
                bp_area = 0.0
                boundary_pts = [tuple(pt) for pt in boundary_contour]
                bp = Polygon(boundary_pts)
                if not bp.is_valid:
                    bp = geom_factory(lgeos.GEOSMakeValid(bp._geom))
                bp_convex_hull_area = bp.convex_hull.area
                bc_errors.append(abs(room_hull.area - bp_convex_hull_area)/bp_convex_hull_area)

                # 计算凸包IoU（交并比）
                intersection = room_hull.intersection(bp.convex_hull).area
                union = room_hull.union(bp.convex_hull).area
                iou = intersection / union if union > 0 else 0
                all_iou.append(iou)

                # 计算面积IOU
                def polygon_to_mask(poly, shape):
                    """
                    使用 cv2.fillPoly 将多边形 rasterize 成 mask
                    :param polygons: list of shapely.Polygon
                    :param shape: (H, W) 图像大小
                    :return: mask (H, W) uint8
                    """
                    mask = np.zeros(shape, dtype=np.uint8)

                    coords = np.array(poly.exterior.coords, dtype=np.int32)
                    coords = coords.reshape((-1, 1, 2))  # fillPoly 需要 (N,1,2) 格式
                    cv.fillPoly(mask, [coords], 1)
                    return mask

                def compute_pixel_iou_cv(polys_out_door, bp, image_size=(256, 256)):
                    """
                    :param polys_out_door: list of coords (N,2)，预测多边形点
                    :param bp: shapely Polygon，GT 多边形
                    :param image_size: (H, W)
                    """
                    try:
                        room_mask = np.zeros(image_size, dtype=np.uint8)
                        for coords in polys_out_door: # 每个房间的拐点
                            if len(coords) >= 3:
                                poly = Polygon(coords)
                                if not poly.is_valid:
                                    poly = poly.buffer(0)

                                rm = polygon_to_mask(poly,image_size)
                                room_mask = cv.bitwise_or(room_mask, rm)

                        gt_mask = polygon_to_mask(bp, image_size)

                        # IoU 计算
                        intersection = np.logical_and(room_mask, gt_mask).sum()
                        union = np.logical_or(room_mask, gt_mask).sum()
                        iou = intersection / union if union > 0 else 0.0
                        return iou
                    except Exception as e:
                        return 0.0
                iou_area = compute_pixel_iou_cv(polys_out_door,bp)
                all_iou_area.append(iou_area)

        return areas_list, bc_errors,all_iou,all_iou_area

    # 结果存储结构：新增批次级累加器
    results = {
        "per_image": {  # 单张图片的结果
            "pred": {"areas": [], "bc_errors": []},
            "gt": {"areas": [], "bc_errors": []},
            "diffs": {"mean_area_error": [], "bc_error_gt": [], "bc_error_pred": []}
        },
        "batch": {  # 批次级累加结果
            "mean_area_error": 0.0,  # 按类别累加的总面积差（真实-预测）
            "mean_iou_pred": 0.0,     # 新增：预测数据的平均IoU
            "mean_iou_area_pred": 0.0,     # 新增：预测数据的平均IoU
            "mean_bc_pred": 0.0,  # BC误差差值平均值

            "mean_iou_gt": 0.0,  # 新增：真实数据的平均IoU
            "mean_iou_area_gt": 0.0,  # 新增：真实数据的平均IoU
            "mean_bc_gt": 0.0,  # BC误差差值总和
            "total_bc_pred": 0.0,  # BC误差差值平均值
            "total_bc_gt": 0.0,  # BC误差差值总和
        }
    }
    # 处理预测和真实数据
    pred_areas, pred_bc,pred_iou,pred_iou_area = process_data(sample,is_syn=args.is_syn)
    gt_areas, gt_bc,gt_iou,gt_iou_area = process_data(sample_gt,is_syn=False)

    # 1. 按图片计算面积差和BC误差差
    num_images = len(gt_areas)
    for img_idx in range(num_images):
        # 计算单张图片的类别面积差
        gt_area_dict = {}
        for room in gt_areas[img_idx]:
            for ids, area in room.items(): # ids表示的是房间索引
                # gt_area_dict[ids] = gt_area_dict.get(ids, 0.0) + area
                gt_area_dict[ids] = area

        pred_area_dict = {}
        for room in pred_areas[img_idx]:
            for ids, area in room.items():
                pred_area_dict[ids] = area

        all_ids = set(gt_area_dict.keys()) #.union(pred_area_dict.keys())
        # total_area = sum(gt_area_dict.get(ids, 0.0) for ids in all_ids)
        # img_area_diff = {ids: abs(gt_area_dict.get(ids, 0.0) - pred_area_dict.get(ids, 0.0))/total_area for ids in all_ids} # 单张照片不同类别的差别
        # print(gt_area_dict,pred_area_dict)
        img_area_diff = {ids: abs(gt_area_dict.get(ids, 0.0) - pred_area_dict.get(ids, 0.0))/gt_area_dict.get(ids, 0.0) for ids in all_ids} # 单张照片不同类别的差别,这种方法相当于是对不同面积的加了权重，我会优先去满足面积大的房间
        mean_area_error = sum(img_area_diff.values())/len(img_area_diff) # 平均每个房间的面积错误率
        results["per_image"]["diffs"]["mean_area_error"].append(mean_area_error)

    # 计算平均值（除以图片数量）
    # total_mean_area_error = sum(results["per_image"]["diffs"]["mean_area_error"])/num_images
    total_mean_area_error = sum(results["per_image"]["diffs"]["mean_area_error"])/len(results["per_image"]["diffs"]["mean_area_error"])

    # 存储批次级结果
    results["batch"]["mean_area_error"] = total_mean_area_error
    results["batch"]["total_bc_gt"] = sum(gt_bc)
    results["batch"]["total_bc_pred"] = sum(pred_bc)
    results["batch"]["mean_bc_gt"] = sum(gt_bc) / len(gt_bc) if gt_bc else 0.0
    results["batch"]["mean_bc_pred"] = sum(pred_bc) / len(pred_bc) if pred_bc else 0.0
    # 计算平均IoU和Hausdorff距离
    results["batch"]["mean_iou_pred"] = sum(pred_iou) / len(pred_iou) if pred_iou else 0
    results["batch"]["mean_iou_gt"] = sum(gt_iou) / len(gt_iou) if gt_iou else 0
    results["batch"]["mean_iou_area_pred"] = sum(pred_iou_area) / len(pred_iou_area) if pred_iou else 0
    results["batch"]["mean_iou_area_gt"] = sum(gt_iou_area) / len(gt_iou_area) if gt_iou else 0
    return results['batch']
def get_graph(args,indx, g_true, ID_COLOR, draw_graph, save_svg):
    # build true graph
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    # add nodes
    for k, (label,area) in enumerate(zip(g_true[0],g_true[-1])):
        _type = label
        if _type >= 0 and _type not in [11, 12]:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            # node_size.append(1000)
            node_size.append(int(area))
            edgecolors.append('blue')
            linewidths.append(0.0)
    # add outside node
    G_true.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    # add edges
    for k, m, l in g_true[1]:
        k = int(k)
        l = int(l)
        _type_k = g_true[0][k]
        _type_l = g_true[0][l]
        if m > 0 and (_type_k not in [11, 12] and _type_l not in [11, 12]):
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')
        elif m > 0 and (_type_k==11 or _type_l==11):
            if _type_k==11:
                G_true.add_edges_from([(l, -1)])
            else:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    # print(draw_graph,"get_graph")
    if draw_graph:
        plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
        nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
                font_weight='bold', edgecolors=edgecolors, width=4.0, with_labels=False)
        if save_svg:
            plt.savefig(f'{args.output_path}/graphs_gt/{indx}.svg')
        else:
            plt.savefig(f'{args.output_path}/graphs_gt/{indx}.jpg')
        plt.close('all')
    return G_true

def estimate_graph(args,indx, polys, nodes, G_gt,ID_COLOR, draw_graph, save_svg,is_syn=False):  # polys, nodes, G_gt 点、类型、图
    nodes = np.array(nodes)
    area = []
    for poly in polys:
        p1 = Polygon(poly)
        if not p1.is_valid:
            p1 = geom_factory(lgeos.GEOSMakeValid(p1._geom))
        area.append(p1.area/2)
    area = np.array(area)
    # G_gt = G_gt[1-th.where((G_gt == th.tensor([0,0,0], device='cuda')).all(dim=1))[0]]
    G_gt = G_gt[~(G_gt == th.tensor([0, 0, 0], device='cuda')).all(dim=1)]
    G_gt = get_graph(args,indx, [nodes, G_gt,area],ID_COLOR, draw_graph= not is_syn, save_svg=save_svg) # 如果是预测的时候就不画真实的图了，避免覆盖
    G_estimated = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    edge_labels = {}
    # add nodes
    for k, (label,a) in enumerate(zip(nodes,area)):
        _type = label
        if _type >= 0 and _type not in [11, 12]:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(int(a))
            linewidths.append(0.0)
    # add outside node
    G_estimated.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    # add node-to-door connections
    doors_inds = np.where((nodes == 11) | (nodes == 12))[0]
    rooms_inds = np.where((nodes != 11) & (nodes != 12))[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:
                p1, p2 = polys[k], polys[l]
                p1, p2 = Polygon(p1), Polygon(p2)
                if not p1.is_valid:
                    p1 = geom_factory(lgeos.GEOSMakeValid(p1._geom))
                if not p2.is_valid:
                    p2 = geom_factory(lgeos.GEOSMakeValid(p2._geom))
                iou = p1.intersection(p2).area/ p1.union(p2).area
                if iou > 0 and iou < 0.2:
                    doors_rooms_map[k].append((l, iou))
    # draw connections
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)
        _conn_top2 = _conn[:2]
        if nodes[k] != 11:
            if len(_conn_top2) > 1:
                l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
                edge_labels[(l1, l2)] = k
                G_estimated.add_edges_from([(l1, l2)])
        else:
            if len(_conn) > 0:
                l1 = _conn[0][0]
                edge_labels[(-1, l1)] = k
                G_estimated.add_edges_from([(-1, l1)])
    # add missed edges
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])
    # add edges colors
    colors = []
    mistakes = 0
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')
    # if draw_graph:
    # print(is_syn,"estimate_graph")
    if is_syn:
        plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
        weights = [4 for u, v in G_estimated_complete.edges()]
        nx.draw(G_estimated_complete, pos, edge_color=colors, linewidths=linewidths, edgecolors=edgecolors, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', width=weights, with_labels=False)
        if save_svg:
            plt.savefig(f'{args.output_path}/graphs_pred/{indx}.svg')
        else:
            plt.savefig(f'{args.output_path}/graphs_pred/{indx}.jpg')
        plt.close('all')
    return mistakes

def save_samples(args,
        sample, ext, model_kwargs,
        tmp_count, num_room_types,
        save_gif=False, save_edges=False,
        door_indices = [11, 12, 13], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False):
    prefix = 'syn_' if is_syn else ''
    graph_errors = []
    if not save_gif:
        sample = sample[-1:]
    for i in tqdm(range(sample.shape[1])):
        resolution = 256
        images = []
        images2 = []
        images3 = []
        boundaries = model_kwargs.get('boundary', None)[i][1]  # 边界线掩码
        # print(model_kwargs.get('boundary', None)[i].shape)
        boundary_mask = ((boundaries == 0.5)).cpu().detach().numpy().astype(np.uint8)
        contours, _ = cv.findContours(boundary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        # contours, _ = cv.findContours(boundary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # boundary_contour = max(contours, key=cv.contourArea)
        # contours = boundary_contour.squeeze(axis=1)
        boundary_pts = [tuple(pt[0]) for pt in contours]
        for k in range(sample.shape[0]):
            draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw2 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw2.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw3 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw3.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='white'))

            draw_color_b = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color_b.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))  # 用白色填充
            polys = []
            types = []
            for j, point in (enumerate(sample[k][i])):
                if model_kwargs[f'{prefix}src_key_padding_mask'][i][j]==1:
                    continue
                point = point.cpu().data.numpy()
                if j==0:
                    poly = []
                if j>0 and (model_kwargs[f'{prefix}room_indices'][i, j]!=model_kwargs[f'{prefix}room_indices'][i, j-1]).any():
                    polys.append(poly)
                    types.append(c)
                    poly = []
                pred_center = False
                if pred_center:
                    point = point/2 + 1
                    point = point * resolution//2
                else:
                    point = point/2 + 0.5
                    # point = point + 0.5
                    point = point * resolution
                poly.append((point[0], point[1]))
                c = np.argmax(model_kwargs[f'{prefix}room_types'][i][j-1].cpu().numpy())
            polys.append(poly)
            types.append(c)
            for poly, c in zip(polys, types):
                if c in door_indices or c==0:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                draw_color_b.append(drawsvg.Lines(*np.array(poly).flatten().tolist(),close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=0.7))

                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            for poly, c in zip(polys, types):
                if c not in door_indices:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                draw_color_b.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type],fill_opacity=1.0, stroke='black', stroke_width=1))

                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            ###############################################################################
            if 'boundary_pts' in locals():  # 若提取到边界轮廓
                # 对应 cv2.polylines(..., True, 1.0, 3)：闭合、白色、线宽3
                draw.append(drawsvg.Lines(
                    *np.array(boundary_pts).flatten().tolist(),  # 展开顶点坐标
                    close=True,  # 闭合多边形
                    fill='none',  # 不填充（只画线）
                    stroke='#000000',  # 1.0对应白色
                    stroke_width=3  # 线宽3（与cv2一致）
                ))
                # 在其他画布（draw2、draw_color）同步绘制
                draw2.append(drawsvg.Lines(
                    *np.array(boundary_pts).flatten().tolist(),
                    close=True, fill='none', stroke='#000000', stroke_width=3
                ))
                draw_color_b.append(drawsvg.Lines(
                    *np.array(boundary_pts).flatten().tolist(),
                    close=True, fill='none', stroke='#000000', stroke_width=3
                ))
            ###############################################################################
            images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw.as_svg()))))
            images2.append(Image.open(io.BytesIO(cairosvg.svg2png(draw2.as_svg()))))
            images3.append(Image.open(io.BytesIO(cairosvg.svg2png(draw3.as_svg()))))
            if k==sample.shape[0]-1 or True:
                if save_edges:
                    draw.save_svg(f'{args.output_path}/{ext}/{tmp_count+i}_{k}_{ext}.svg')
                if save_svg:
                    draw_color.save_svg(f'{args.output_path}/{ext}/{tmp_count+i}c_{k}_{ext}.svg')
                    draw_color_b.save_svg(f'{args.output_path}/{ext}_b/{tmp_count + i}c_{k}_{ext}.svg')
                else:
                    Image.open(io.BytesIO(cairosvg.svg2png(draw_color.as_svg()))).save(f'{args.output_path}/{ext}/{tmp_count+i}c_{ext}.png')
                    Image.open(io.BytesIO(cairosvg.svg2png(draw_color_b.as_svg()))).save(f'{args.output_path}/{ext}_b/{tmp_count + i}c_{ext}.png')
            if k==sample.shape[0]-1:
                if 'graph' in model_kwargs:
                    graph_errors.append(estimate_graph(args,tmp_count+i, polys, types, model_kwargs[f'{prefix}graph'][i],ID_COLOR=ID_COLOR, draw_graph=draw_graph, save_svg=save_svg, is_syn=is_syn))
                else:
                    graph_errors.append(0)
        if save_gif:
            imageio.mimwrite(f'{args.output_path}/gif/{tmp_count+i}.gif', images, fps=10, loop=1)
            imageio.mimwrite(f'{args.output_path}/gif/{tmp_count+i}_v2.gif', images2, fps=10, loop=1)
            imageio.mimwrite(f'{args.output_path}/gif/{tmp_count+i}_v3.gif', images3, fps=10, loop=1)
    return graph_errors

def create_logger_ckpts(args):
    # formatter = logging.Formatter(f'{args.model_id}: %(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(f'{args.model_id}: %(message)s')
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 文件处理器 - 输出到文件
    file_handler = logging.FileHandler(args.logger_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    # 控制台处理器 - 输出到控制台
    import  sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    # 设置shapely日志级别为ERROR
    shapely_logger = logging.getLogger("shapely")
    shapely_logger.setLevel(logging.ERROR)
    # 禁止shapely日志传播到根记录器
    shapely_logger.propagate = False
    return root_logger

def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)
    dist_util.setup_dist()

    log_obj = create_logger_ckpts(args)
    log_obj.info("日志管理器配置完成")
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    errors = []
    output_path_base = args.output_path
    for _ in range(1):
        logger.log("sampling...")
        tmp_count = 0
        os.makedirs(f'{args.output_path}/pred', exist_ok=True)
        os.makedirs(f'{args.output_path}/gt', exist_ok=True)
        os.makedirs(f'{args.output_path}/pred_b', exist_ok=True)
        os.makedirs(f'{args.output_path}/gt_b', exist_ok=True)
        os.makedirs(f'{args.output_path}/gif', exist_ok=True)
        os.makedirs(f'{args.output_path}/graphs_gt', exist_ok=True)
        os.makedirs(f'{args.output_path}/graphs_pred', exist_ok=True)
        # 1:客厅 2:厨房 3：卧室 4：卫生间 5：阳台 6：通道 7：餐厅 8：书房 10：储藏室 11：前门 12：内门 13：未知区域
        if args.dataset=='rplan':
            ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                        13: '#785A67', 12: '#D3A2C7'}
            num_room_types = 14
            data = load_rplanhg_data(
                batch_size=args.batch_size,
                analog_bit=args.analog_bit,
                set_name=args.set_name,
                target_set=args.target_set,
            )
        else:
            print("dataset does not exist!")
            assert False
        graph_errors = []
        while tmp_count < args.num_samples:
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            data_sample, model_kwargs = next(data)
            # 打印data_sample的维度
            print(f"data_sample shape: {data_sample.shape}")

            # 打印model_kwargs中每个key的维度
            print("model_kwargs dimensions:")
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].cuda()
                if isinstance(model_kwargs[key], th.Tensor):
                    print(f"  {key}: {model_kwargs[key].shape}")
                else:
                    print(f"  {key}: {type(model_kwargs[key])}")

            sample = sample_fn(
                model,
                data_sample.shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                analog_bit=args.analog_bit,
            )
            sample_gt = data_sample.cuda().unsqueeze(0)
            sample = sample.permute([0, 1, 3, 2])
            sample_gt = sample_gt.permute([0, 1, 3, 2])
            if args.analog_bit:
                sample_gt = bin_to_int_sample(sample_gt)
                sample = bin_to_int_sample(sample)

            graph_error = save_samples(args,sample_gt, 'gt', model_kwargs, tmp_count, num_room_types, ID_COLOR=ID_COLOR, draw_graph=args.draw_graph, save_svg=args.save_svg)
            graph_error = save_samples(args,sample, 'pred', model_kwargs, tmp_count, num_room_types, ID_COLOR=ID_COLOR, is_syn=args.is_syn, draw_graph=args.draw_graph, save_svg=args.save_svg)
            graph_errors.extend(graph_error)
            res = estimate_areas(args, sample_gt, sample, model_kwargs)
            log_obj.info(f'res: {res}')

            tmp_count+=sample_gt.shape[1]
        # logger.log("sampling complete")
        # convert_svg_folder_to_png(args.output_path + "/gt", args.output_path + "/gt_png")
        # convert_svg_folder_to_png(args.output_path + "/pred", args.output_path + "/pred_png")
        # fid_score = calculate_fid_given_paths([args.output_path+'/gt_png', args.output_path+'/pred_png'], 64, 'cuda', 2048)
        # fid_score = calculate_fid_given_paths([args.output_path+'/gt', args.output_path+'/pred'], 64, 'cuda', 2048)
        # log_obj.info(f'fid_score: {fid_score}')
    #     log_obj.info(f'Compatibility: {np.mean(graph_errors)}')
    #
    #     print(f'FID: {fid_score}')
    #     print(f'Compatibility: {np.mean(graph_errors)}')
    #     errors.append([fid_score, np.mean(graph_errors)])
    # errors = np.array(errors)
    # log_obj.info(f'Diversity mean: {errors[:, 0].mean()} \t Diversity std: {errors[:, 0].std()}')
    # log_obj.info(f'Compatibility mean: {errors[:, 1].mean()} \t Compatibility std: {errors[:, 1].std()}')
    # print(f'Diversity mean: {errors[:, 0].mean()} \t Diversity std: {errors[:, 0].std()}')
    # print(f'Compatibility mean: {errors[:, 1].mean()} \t Compatibility std: {errors[:, 1].std()}')


def create_argparser():
    defaults = dict(
        dataset='',
        output_path="../outputs",
        logger_path="../output/log.txt",
        model_id='model002000.pt',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        draw_graph=True,
        save_svg=False,
        is_syn=True,
        param=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    main()
