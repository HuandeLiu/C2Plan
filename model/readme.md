# 数据集格式

```
rplan_{set_name}_{target_set}.npz set_name 用来判断是训练数据还是测试数据、target_set表示房间数
设样本数为 N，并且代码里 max_num_points=100，图边固定 pad 到 200，则典型维度是：
houses: (N, 100, 95)
    每个点 95 维特征，由拼接得到：2 + 25 + 32 + 32 + 1 + 2 + 1 = 95 
   具体含义：
   [:2] 坐标 (x,y)（已映射到 [-1,1]）
   [2:27] room type one-hot (25)
   [27:59] corner index one-hot (32)
   [59:91] room index one-hot (32)
   [91] padding mask 标志
   [92:94] polygon connection（当前角点连接到下一个角点的索引对）
   [94] room area ratio
graphs: (N, 200, 3) 每条三元组 [room_i, relation, room_j]，relation 为 1/-1（相邻/不相邻）
door_masks: (N, 100, 100)
self_masks: (N, 100, 100)
gen_masks: (N, 100, 100)
boundarys: (N, 3, 256, 256) 三通道分别来自 inside / boundary / global_mask
```

