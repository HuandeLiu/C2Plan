"""
Train a diffusion model on images.
"""

import argparse
import os
import sys
# 替换为 house_diffusion 所在的父目录路径
# module_path = "/home/cyq/lhd/fooldiff/housediffusion/house_diffusion-main/"
# 将路径添加到 Python 搜索路径
# if module_path not in sys.path:
#     sys.path.append(module_path)
import shutil
import blobfile as bf

import dist_util, logger
from rplanhg_datasets import load_rplanhg_data
from resample import create_named_schedule_sampler
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    update_arg_parser,
)
from train_util import TrainLoop

def backup(source_dir,backup_path):
    backup_path = os.path.join(backup_path, "code")
    print(source_dir, backup_path)
    try:
        # 创建备份目录
        os.makedirs(backup_path, exist_ok=True)
        print(f"备份目录创建成功: {backup_path}")
        # 统计找到的Python文件数量
        py_file_count = 0
        # 遍历源目录及其子目录
        for root, dirs, files in os.walk(source_dir):
            # 检查当前目录是否需要排除（可根据需要添加）
            exclude_dirs = {'outputs', 'logs', 'processed_rplan'}
            if any(exclude in root for exclude in exclude_dirs):
                continue
            for file in files:
                # 只处理.py文件
                if file.endswith(".py") or file.endswith(".sh"):
                    py_file_count += 1
                    source_file = os.path.join(root, file)
                    # 保持原有的目录结构
                    relative_path = os.path.relpath(root, source_dir)
                    dest_dir = os.path.join(backup_path, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)

                    # 复制文件
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(source_file, dest_file)  # copy2会保留文件元数据

                    # 每备份10个文件输出一次进度
                    if py_file_count % 10 == 0:
                        print(f"已备份 {py_file_count} 个Python文件...")

        print(f"备份完成！共备份了 {py_file_count} 个Python文件")
        print(f"备份文件存储在: {backup_path}")

    except Exception as e:
        print(f"备份过程中发生错误: {str(e)}")

def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()

    if args.backup:
        backup_path = logger.get_dir()
        # backup(os.path.dirname(os.getcwd()),bf.join("outputs",backup_path))
        backup(os.getcwd(),bf.join("outputs",backup_path))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.dataset=='rplan':
        data = load_rplanhg_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            target_set=args.target_set,
            set_name=args.set_name,
        )
    else:
        print('dataset not exist!')
        assert False

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        analog_bit=args.analog_bit,
        backup=args.backup,
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset_dir='./data',
        dataset = '',
        backup=1,
        schedule_sampler= "uniform", #"loss-second-moment", "uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    parser = argparse.ArgumentParser()
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
