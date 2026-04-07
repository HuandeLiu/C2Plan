"""
根据用户提供的 demo.json 文件，运行模型生成户型图。
输出保存到 output 目录，包含：
- 真实边界图
- 真实户型图
- 预测户型图
"""

import argparse
import os
import sys
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import webcolors
import networkx as nx
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos

from rplanhg_datasets import load_demo_to_model_input
import dist_util
from logger import configure
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)


def bin_to_int(x):
    """将二进制列表转换为整数"""
    return int("".join([str(int(i.cpu().data)) for i in x]), 2)


def bin_to_int_sample(sample, resolution=256):
    """将二进制编码的样本转换为坐标"""
    sample_new = th.zeros([sample.shape[0], sample.shape[1], sample.shape[2], 2])
    sample[sample < 0] = 0
    sample[sample > 0] = 1
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(sample.shape[2]):
                sample_new[i, j, k, 0] = bin_to_int(sample[i, j, k, :8])
                sample_new[i, j, k, 1] = bin_to_int(sample[i, j, k, 8:])
    sample = sample_new
    sample = sample / (resolution / 2) - 1
    return sample


# def save_boundary_image(boundary, output_path, filename):
#     """保存边界图像"""
#     boundary_line = boundary[1]  # 边界线通道
#     boundary_mask = ((boundary_line == 0.5).cpu().detach().numpy() * 255).astype(np.uint8)
#     boundary_color = cv.cvtColor(boundary_mask, cv.COLOR_GRAY2BGR)
#     cv.imwrite(os.path.join(output_path, filename), boundary_color)

def save_boundary_image(boundary, output_path, filename):
    boundary_line = boundary[1]

    boundary_mask = (boundary_line == 0.5).cpu().detach().numpy().astype(np.uint8)

    # 创建白底
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    # 找轮廓
    contours, _ = cv.findContours(boundary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 画黑色边界
    cv.drawContours(img, contours, -1, (0, 0, 0), 3)

    cv.imwrite(os.path.join(output_path, filename), img)


def save_samples(args, sample, ext, model_kwargs, ID_COLOR=None, is_syn=False, save_svg=False):
    """保存生成的户型图样本

    sample 形状：[num_samples, batch_size, 100, 2/16]
    """
    prefix = 'syn_' if is_syn else ''
    resolution = 256

    # sample.shape[1] = batch_size, sample.shape[0] = num_samples
    for i in range(sample.shape[1]):  # batch 维度
        boundaries = model_kwargs.get('boundary', None)[i][1]
        boundary_mask = ((boundaries == 0.5)).cpu().detach().numpy().astype(np.uint8)
        contours, _ = cv.findContours(boundary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        boundary_contour = max(contours, key=cv.contourArea)
        boundary_pts = [tuple(pt[0]) for pt in boundary_contour]

        for k in range(sample.shape[0]):  # num_samples 维度
            draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='black'))

            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))

            draw_color_b = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color_b.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))

            polys = []
            types = []

            for j, point in enumerate(sample[k, i]):
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
                    poly = []

                point = point / 2 + 0.5
                point = point * resolution
                poly.append((point[0], point[1]))
                room_type_key = f'{prefix}room_types'
                c = np.argmax(model_kwargs[room_type_key][i][j - 1].cpu().numpy())

            polys.append(poly)
            types.append(c)
            # print("types:",types)
            door_indices = [11, 12, 13]
            for poly, c in zip(polys, types):
                if c in door_indices or c == 0:
                    continue
                room_type = c
                c_rgb = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                               fill=ID_COLOR[room_type], fill_opacity=1.0,
                                               stroke='black', stroke_width=1))
                draw_color_b.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                 fill=ID_COLOR[room_type], fill_opacity=1.0,
                                                 stroke='black', stroke_width=0.7))
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                         fill='black', fill_opacity=0.0,
                                         stroke=webcolors.rgb_to_hex([int(x/2) for x in c_rgb]),
                                         stroke_width=0.5 * (resolution / 256)))

            for poly, c in zip(polys, types):
                if c not in door_indices:
                    continue
                room_type = c
                c_rgb = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                               fill=ID_COLOR[room_type], fill_opacity=1.0,
                                               stroke='black', stroke_width=1))
                draw_color_b.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                 fill=ID_COLOR[room_type], fill_opacity=1.0,
                                                 stroke='black', stroke_width=1))
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                         fill='black', fill_opacity=0.0,
                                         stroke=webcolors.rgb_to_hex([int(x/2) for x in c_rgb]),
                                         stroke_width=0.5 * (resolution / 256)))

            # 绘制边界
            draw.append(drawsvg.Lines(*np.array(boundary_pts).flatten().tolist(),
                                     close=True, fill='none', stroke='#000000', stroke_width=3))
            draw_color_b.append(drawsvg.Lines(*np.array(boundary_pts).flatten().tolist(),
                                              close=True, fill='none', stroke='#000000', stroke_width=3))

            if k == sample.shape[0] - 1 or True:
                if save_svg:
                    draw_color.save_svg(os.path.join(args.output_path, f'{ext}/{i}c_{k}_{ext}.svg'))
                    draw_color_b.save_svg(os.path.join(args.output_path, f'{ext}_b/{i}c_{k}_{ext}.svg'))
                else:
                    Image.open(io.BytesIO(cairosvg.svg2png(draw_color.as_svg()))).save(
                        os.path.join(args.output_path, f'{ext}/{i}c_{ext}.png'))
                    Image.open(io.BytesIO(cairosvg.svg2png(draw_color_b.as_svg()))).save(
                        os.path.join(args.output_path, f'{ext}_b/{i}c_{ext}.png'))


def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)
    dist_util.setup_dist()

    print(f"加载模型：{args.model_path}")
    print(f"输入文件：{args.demo_json}")
    print(f"输出目录：{args.output_path}")

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'gt_b'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'pred_b'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'boundary'), exist_ok=True)

    # 加载模型
    print("创建模型和扩散...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # 从 demo.json 加载数据
    print(f"从 {args.demo_json} 加载数据...")
    arr, model_kwargs = load_demo_to_model_input(
        demo_json_path=args.demo_json,
        analog_bit=args.analog_bit,
        target_set=args.target_set,
        set_name='eval',
    )
    input_shape = arr.shape  # 保存输入形状用于 sample_fn

    # 处理 model_kwargs 中的每个字段
    for key in model_kwargs:
        model_kwargs[key] = model_kwargs[key].cuda()

    # print(f"输入形状：{input_shape}")
    # # 打印 model_kwargs 中各字段的形状用于调试
    # print("model_kwargs 字段形状:")
    # for key in model_kwargs:
    #     if isinstance(model_kwargs[key], th.Tensor):
    #         print(f"  {key}: {model_kwargs[key].shape}")

    # 运行模型推理
    print("运行模型推理...")
    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop

    sample = sample_fn(
        model,
        input_shape,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        analog_bit=args.analog_bit,
    )
    sample_gt = arr.cuda().unsqueeze(0)
    sample_gt = sample_gt.permute([0, 1, 3, 2])
    sample = sample.permute([0, 1, 3, 2])  # [num_samples, batch_size, 100, 2/16]
    if args.analog_bit:
        sample_gt = bin_to_int_sample(sample_gt)
        sample = bin_to_int_sample(sample)



    # 保存结果
    print("保存结果...")
    ID_COLOR = {
        1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
        13: '#785A67', 12: '#D3A2C7'
    }

    # 保存真实边界图
    boundary = model_kwargs.get('boundary', None)[0]
    save_boundary_image(boundary, os.path.join(args.output_path, 'boundary'), 'boundary.png')

    # === 保存 GT 户型图（新增）===
    save_samples(args, sample_gt, 'gt', model_kwargs,
                 ID_COLOR=ID_COLOR,
                 is_syn=False,
                 save_svg=args.save_svg)

    # 保存预测结果
    save_samples(args, sample, 'pred', model_kwargs, ID_COLOR=ID_COLOR,
                 is_syn=args.is_syn, save_svg=args.save_svg)

    print(f"完成！结果保存到：{args.output_path}")


def create_argparser():
    defaults = dict(
        demo_json="",
        output_path="./output",
        model_id='model.pt',
        clip_denoised=True,
        use_ddim=False,
        model_path="",
        save_svg=False,
        is_syn=True,
        param=0.1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()