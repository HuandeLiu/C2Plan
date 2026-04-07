# 代码含义

```
raster_to_json.py 把单张图片转为json文件 
run.py 是用于批量将图片转json
cleanjsondata.py、clearnjsondatacorners.py这两个是清理一些不合理的json文件
list.py 把所有合理的json文件转为一个列表
```

# json 字段含义

```text
字段含义（按 raster_to_json.py 写入）
name：样本 id（通常是原始图片文件名去掉扩展名）。

room_type：每个“节点”的类别 id 列表。这里的“节点”不止房间，也包含门/入口等特殊结构。类别 id 的含义在 dataprocess/utils.py 的 room_label 里定义（例如 15=FrontDoor，17=InteriorDoor）。

boxes：对 room_type 中每个节点给一个 bbox，格式是 [y0, x0, y1, x1]（注意 y 在前、x 在后）。这些 bbox 是从多边形/线段集合算出来的外接框。

edges：整张图的线段集合（墙、门框等），每条是 6 元组：

[y1, x1, y2, x2, semantic_type, neighbor_room_type]

前 4 个是线段两个端点坐标（仍是 y,x 顺序；这里输出就是像素坐标尺度，代码里 lenx=leny=1 没做额外缩放）。
第 5 个 semantic_type：这条线段属于什么语义（普通墙线段会用对应房间类型 id；门/入口线段会是 17/15）。
第 6 个 neighbor_room_type：如果这条线段是门/入口，会记录它“连到的相邻房间类型”；否则一般为 0。（门和房间的对应关系是通过几何匹配补出来的）
ed_rm：每条线段对应的“房间索引”映射（和 edges 一一对应）。

通常是 [room_i]（这条边属于某个房间外轮廓）
或者 [room_i, room_j]（例如门/共享边，能关联到两个房间）
```

