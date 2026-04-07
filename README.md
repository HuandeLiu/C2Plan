# C2Plan: Conditional Floor Plan Generation System

C2Plan是一个基于深度学习的平面图生成系统，能够根据用户定义的房间类型、面积比例和连接关系自动生成建筑平面图。

## 项目概述

本项目实现了一个条件平面图生成模型，支持：
- 根据用户定义的房间类型、数量和面积比例生成平面图
- 支持房间之间的连接关系定义
- 生成符合建筑规范的平面布局
- 可视化边界图和房间布局

## 项目结构

```
C2Plan/
├── README.md                 # 项目说明文档
├── dataprocess/              # 数据处理模块
│   ├── rplan_json/           # JSON格式的平面图数据
│   ├── list.txt             # 数据集文件列表
│   ├── raster_to_json.py    # PNG转JSON工具
│   └── cleanjsondata.py     # 数据清洗工具
├── model/                    # 模型实现
│   ├── rplanhg_datasets.py  # 数据集加载器
│   ├── image_sample.py      # 测试机运行
│   └── ...                  # 其他模型文件
├── boundary/                 # 边界图导出工具
│   ├── export_boundary.py   # 边界图导出脚本
│   └── photo/               # 导出的边界图
├── user_data/               # 用户输入数据
│   └── demo.json            # 示例用户配置
├── output_user/             # 模型输出结果
├── ckpts/                   # 模型检查点
└── web/                     # Web界面（可选）
```

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd C2Plan

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

1. **下载数据集**：
   - [RPLAN](https://drive.google.com/file/d/1qBAHTwl2irL80V2vNH-doUZcRi8EEeT1/view?usp=drive_link)
   - 解压到项目根目录

2. **数据处理**：
   ```bash
   cd dataprocess
   # 将PNG图片转换为JSON格式
   python raster_to_json.py
   # 清洗数据并生成文件列表
   python cleanjsondata.py
   ```

### 3. 模型训练

```bash
cd model
# 训练模型（根据实际情况调整参数）
sh script_train.sh
# 模型将保存到 ../ckpts/ 目录
```
预训练模型：[model](https://drive.google.com/file/d/1eJ_fwvR0nf-fL3FstTHydzEdJztbsjTU/view?usp=drive_link)
### 4. 边界图导出

```bash
cd boundary
# 导出JSON数据的边界图
python export_boundary.py
# 边界图将保存到 boundary/photo/ 目录
```

### 5. 使用示例

1. **编辑用户配置**：修改 `user_data/demo.json`
2. **运行生成**：使用模型生成平面图
3. **查看结果**：在 `output_user/` 目录查看生成的平面图

## 用户配置格式

`user_data/demo.json` 示例：

```json
{
    "name": "13",
    "room_type": [3, 4, 1, 3, 2, 5, 17, 17, 17, 17, 17, 15],
    "room_corner_nums": [4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    "room_area_rate": [0.1, 0.1, 0.3, 0.18, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    "room_connections": [[0,6],[2,6],[1,7],[2,7],[2,8],[3,8],[2,9],[4,9],[2,10],[5,10]]
}
```

**字段说明**：
- `room_type`: 房间类型编码（1:客厅, 2:厨房, 3:卧室, 4:卫生间, 5:阳台, ...）
- `room_corner_nums`: 每个房间的拐点数量
- `room_area_rate`: 每个房间的面积比例（相对总面积）
- `room_connections`: 房间连接关系列表

## 边界图导出功能

项目提供了边界图导出工具，可以将JSON格式的平面图数据导出为可视化的边界图：

### 命令行使用
```bash
cd boundary
python export_boundary.py
```

### 功能特点
- 从JSON数据提取房间边界
- 生成白底黑边的边界图
- 支持批量处理
- 输出PNG格式图像

## 开发说明

### 数据处理流程
1. **原始数据**：RPLAN数据集的PNG图像
2. **转换**：`raster_to_json.py` 将PNG转换为JSON格式
3. **清洗**：`cleanjsondata.py` 过滤异常数据
4. **训练准备**：生成 `list.txt` 文件列表

### 模型架构
- 基于扩散模型的条件生成
- 支持房间类型、面积、连接关系的条件控制
- 使用图神经网络处理房间连接关系

### 可视化工具
- `image_sample.py`: 生成结果的可视化
- `export_boundary.py`: 边界图导出工具


