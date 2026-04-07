import json
import argparse
import numpy as np
# import matplotlib.pyplot as plt
from shapely.geometry import Polygon
# from descartes.patch import PolygonPatch
from read_dd import read_data
import cv2 as cv
import warnings


def raster_to_json(line, print_door_warning):
    """ convert extracted data from rasters to housegan ++ data format :  extract rooms type, bbox, doors, edges and neigbour rooms
                
    """
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    bbox_x1=[]
    bbox_y1=[]
    bbox_x2=[]
    bbox_y2=[]
    walls=[]
    # 房间类型、每个房间有几个拐点、每个门的墙壁线段（4个）、每个房间的墙壁线段
    # room_type,poly,doors_, walls,out,boundary=read_data(line)
    room_type, poly, doors_, walls, out = read_data(line)

    d=[]
    all_doors=[] # 保存每个门的四条线段
    for i in range(1,len(doors_)+1): # 每4条线为一个门
        if((i)%4==0 ) & (i+1!=1):
            d.append(doors_[i-1])
            all_doors.append(d)
            d=[]
        elif(i==1):
            d=[]
        if(i%4!=0):
            d.append(doors_[i-1])
    kh=0
    al_dr=0
    # 门是 “连接两个房间的通道”，此步骤通过几何坐标匹配，找到每个门关联的两个房间，并更新walls线段的附加信息（填补原有的占位符-1和0）。
    for hd in range(len(all_doors)):
        dr_t=[] # 存储门关联的两个房间索引
        dr_in=[] # 存储关联墙体在`walls`中的索引
        doors=all_doors[hd] # 单个门
        d_t=2
        t_x=abs(doors[0][1]-doors[1][1])
        t_y=abs(doors[0][0]-doors[3][0])
        ss=t_x # 门的“最小边长”（用于后续坐标匹配的阈值）
        if(t_x>t_y): # 水平门
           d_t=1
           ss=t_y
        elif(t_x<t_y): # 垂直门
           d_t=3
        # 迭代5次扩大匹配容错范围
        for pmc in range(5): # 匹配门与相邻墙体 通过 “坐标差值阈值（ss-1）” 和 “线段重叠范围”，找到门两侧的墙体，再通过墙体的walls[nw][6]（区域索引）关联到具体房间：
            for dw in range(len(doors)):      # 遍历当前门的每条线段
                for nw in range(len(walls)) :  # 遍历所有墙体线段
                    # 跳过门（17）和入口（15）的线段，只匹配房间墙体
                    if(walls[nw][5]==17):
                        continue
                    if(walls[nw][5]==15):
                        continue
                    # 水平门匹配逻辑（d_t<=2）：行坐标接近，列坐标有重叠
                    if (d_t<=2) &(doors[dw][0]-doors[dw][2]<=1)& (walls[nw][0]-walls[nw][2]<=1)& (abs(doors[dw][0]-walls[nw][0])<=(ss-1)) & (abs(doors[dw][2]-walls[nw][2])<=(ss-1)): 
                        l=doors[dw][1]
                        r=doors[dw][3]
                        if(l>r):
                            t=l
                            l=r
                            r=t
                        l_=walls[nw][1]
                        r_=walls[nw][3]
                        if(l_>r_):
                            t=l_
                            l_=r_
                            r_=t
                        if (((r-r_)<=pmc )& (pmc>=(l_-l))) :
                            if(len(dr_in)<2):
                                if(walls[nw][6] not in dr_t):
                                    dr_t.append(walls[nw][6])
                                    dr_in.append(nw)
                    # 垂直门匹配逻辑（d_t>=2）：列坐标接近，行坐标有重叠
                    elif (d_t>=2)& (doors[dw][1]-doors[dw][3]<=1)& (walls[nw][1]-walls[nw][3]<=1) &(abs(doors[dw][1]-walls[nw][1])<=(ss-1)) & (abs(doors[dw][3]-walls[nw][3])<=(ss-1)): 
                        l=doors[dw][0]
                        r=doors[dw][2]
                        if(l>r):
                            t=l
                            l=r
                            r=t
                        l_=walls[nw][0]
                        r_=walls[nw][2]
                        if(l_>r_):
                            t=l_
                            l_=r_
                            r_=t
                        if(((r-r_)<=pmc )& (pmc>=(l_-l))):
                            if(len(dr_in)<2):
                                if(walls[nw][6] not in dr_t):
                                    dr_t.append(walls[nw][6])
                                    dr_in.append(nw)
        # 若门成功关联 2 个房间（len(dr_t)==2），则更新这两个房间墙体线段的附加信息（walls的第 7、8 位，原占位符），标识 “相邻房间的索引和类型”：
        if(len(dr_t)==2):
            walls[dr_in[0]][8]=walls[dr_in[1]][5]
            walls[dr_in[0]][7]=walls[dr_in[1]][6]    
            walls[dr_in[1]][8]=walls[dr_in[0]][5]
            walls[dr_in[1]][7]=walls[dr_in[0]][6]  
            al_dr=al_dr+1
                
        else:
            if print_door_warning:
                print("sometime not 2 dooor",hd,doors)		
    
        assert(len(dr_t)<=2)
  

    assert(al_dr==(len(all_doors)-1))
    
    # 针对入口和未完全关联的门，继续匹配相邻墙体，补全其walls线段的附加信息（逻辑与步骤 3 类似，重点处理入口）：
    omn=[]
    tr=0
    en_pp=0  # 入口关联成功标识（需确保=1）
    for nw in range(len(walls)-(len(all_doors)*4),len(walls)): # 遍历门/入口的线段
        if(tr%4==0):
            omn=[]
        tr=tr+1
        for kw in range(len(walls)-(len(all_doors)*4)+1): # 遍历房间墙体线段
            if(walls[kw][5]==17)&(walls[nw][5]==17):
                continue		
            if(walls[kw][5]==15)&(walls[nw][5]==15):
                continue		
            if(walls[kw][5]==15)&(walls[nw][5]==17):
                continue		
            for pmc in range (5):
                if(abs(walls[kw][0]-walls[nw][0])<=(ss-1)) & (abs(walls[kw][2]-walls[nw][2])<=(ss-1)):
                    l=walls[kw][1]
                    r=walls[kw][3]
                    if(l>r):
                        t=l
                        l=r
                        r=t
                    l_=walls[nw][1]
                    r_=walls[nw][3]
                    if(l_>r_):
                        t=l_
                        l_=r_
                        r_=t
                    if(pmc>=r_-r )& (l-l_<=pmc) &( nw!=kw):
                        if(walls[nw][5]==17) &(walls[nw][8]==0)& (walls[kw][6] not in omn):
                            walls[nw][8]=walls[kw][5]
                            walls[nw][7]=walls[kw][6]    
                            omn.append(walls[kw][6])
                        if(walls[nw][5]==15) &(walls[nw][8]==0):
                            walls[nw][8]=walls[kw][5]
                            walls[nw][7]=walls[kw][6] 
                            en_pp=1 
                 
                if(abs(walls[kw][1]-walls[nw][1])<=(ss-1)) & (abs(walls[kw][3]-walls[nw][3])<=(ss-1)):
                    l=walls[kw][0]
                    r=walls[kw][2]
                    if(l>r):
                        t=l
                        l=r
                        r=t
                    l_=walls[nw][0]
                    r_=walls[nw][2]
                    if(l_>r_):
                        t=l_
                        l_=r_
                        r_=t
                    if(pmc>=r_-r )& (l-l_<=pmc) &( nw!=kw):
                        if(walls[nw][5]==17) & (walls[nw][8]==0)& (walls[kw][6] not in omn):
                            walls[nw][8]=walls[kw][5]
                            walls[nw][7]=walls[kw][6]    
                            omn.append(walls[kw][6])
                
                        if(walls[nw][5]==15) & (walls[nw][8]==0):
                            walls[nw][8]=walls[kw][5]
                            walls[nw][7]=walls[kw][6]    
                            en_pp=1
         
                    
    for i in range(1):
        for iw in range(len(walls)):
            tp_out=-1
            dif_x=10
            dif_y=10

            type_out=0
        for jw in range(len(walls)):
            if(walls[iw][0]==walls[iw][2]):
                if (walls[jw][0]!=walls[jw][2]):
                    continue
                if ((walls[iw][0]-walls[jw][0])==(walls[iw][2]- walls[jw][2])):
                    rnp=walls[jw][1]
                    fnp=walls[jw][3]
                    rmp=walls[iw][1]
                    fmp=walls[iw][3]
                    if( rnp<fnp):
                        t=fnp
                        fnp=rnp
                        rnp=t
                    if(rmp<fmp):
                        t=fmp
                        fmp=rmp
                        rmp=t
                    if(abs(rmp)<=abs(rnp))| (abs(fmp)<=abs(fnp)):
                        dif_x_temp=walls[iw][0]-walls[jw][0]
                        if(abs(dif_x)>abs(dif_x_temp)) & (iw!=jw):
                            dif_x=dif_x_temp
                            tp_out=walls[jw][6]
                            type_out=walls[jw][5]
                          
            elif(walls[iw][1]==walls[iw][3]):
                if ((walls[iw][1]-walls[jw][1])==(walls[iw][3]- walls[jw][3])) :           
                    rnp=walls[jw][0]
                    fnp=walls[jw][2]
                    rmp=walls[iw][0]
                    fmp=walls[iw][2]
                    if( rnp<fnp):
                        t=fnp
                        fnp=rnp
                        rnp=t
                    if(rmp<fmp):
                        t=fmp
                        fmp=rmp
                        rmp=t
                    if(abs(rmp)<=abs(rnp))| (abs(fmp)<=abs(fnp)):
                        dif_y_temp=walls[iw][1]-walls[jw][1]
                        if(abs(dif_y)>abs(dif_y_temp))&( iw!=jw ):
                            dif_y=dif_y_temp
                            tp_out=walls[jw][6]
                            type_out=walls[jw][5]
      
    km=0
    assert(en_pp==1) #throwing out really strange layouts
    
    
    lenx=1.0
    leny=1.0
    min_x=0.0
    min_y=0.0
    bboxes=[]  
    edges=[]
    ed_rm=[]
    info=dict()

    #  The edges for the graph
    for w_i in range(len(walls)):
        edges.append([((walls[w_i][0]-min_x)/lenx),((walls[w_i][1]-min_y)/leny),((walls[w_i][2]-min_x)/lenx),((walls[w_i][3]-min_y)/leny),walls[w_i][5],walls[w_i][8]])
        if(walls[w_i][6]==-1):
            ed_rm.append([walls[w_i][7]])
        elif(walls[w_i][7]==-1): 
            ed_rm.append([walls[w_i][6]])
        else:
            ed_rm.append([walls[w_i][6],walls[w_i][7]])
    
    #  The bbox for room masks
    for i in range(len(poly)):
        p=poly[i]
        pm=[]
        for p_i in range((p)):
            pm.append(([edges[km+p_i][0],edges[km+p_i][1]]))
        km=km+p
        polygon = Polygon(pm)
        # plot_coords(ax, polygon.exterior, alpha=0)
        bbox=np.asarray(polygon.bounds)
        bboxes.append(bbox.tolist())
     
        
        # patch = PolygonPatch(polygon, facecolor=semantics_cmap["bedroom"], alpha=0.7)
        # ax.add_patch(patch)
    info['name']=line.split('/')[-1].split('.png')[0]
    info['room_type'] = room_type # 房间类型
    info['boxes'] = bboxes # 每个房间的边界框，包括前门和内门
    # 第5个元素：当前线段的类型（例如，15代表入口，17代表门，其他数字代表普通墙体） ，第6个元素：相邻房间的类型（如果当前线段是门或入口，则记录相邻房间的类型；否则为0）。
    info['edges'] = edges # 每一个墙的起点和终点（前4个），列表的列表，每个内层列表有6个元素
    info['ed_rm'] = ed_rm # 每一个墙连接的房间索引
    # info['boundary'] = boundary.tolist() # 每一个墙连接的房间索引

    # # 创建256x256的空白图像
    # canvas = np.zeros((256, 256), dtype=np.uint8)
    # # 1. 绘制所有线段到图像上
    # for edge in edges:
    #     if edge[4] in [15,17]: continue
    #     # 将归一化坐标转换为像素坐标 (0-255)
    #     x1 = int(edge[0])
    #     y1 = int(edge[1])
    #     x2 = int(edge[2])
    #     y2 = int(edge[3])
    #
    #     # 在画布上绘制白色线段 (厚度2确保线段连接)
    #     cv.line(canvas, (x1, y1), (x2, y2), 255, 2)
    #
    # # 2. 形态学操作填充间隙
    # kernel = np.ones((3, 3), np.uint8)
    # dilated = cv.dilate(canvas, kernel, iterations=1)
    #
    # # 3. 填充闭合区域 (房屋内部)
    # filled = dilated.copy()
    # contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(filled, contours, -1, 255, thickness=cv.FILLED)
    #
    # # 4. 提取外部轮廓
    # contours, _ = cv.findContours(filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    # if contours:
    #     # 找到最大的轮廓 (房屋主体)
    #     main_contour = max(contours, key=cv.contourArea)
    #
    #     # 5. 简化轮廓 (提取关键拐点)
    #     epsilon = 0.005 * cv.arcLength(main_contour, True)
    #     approx = cv.approxPolyDP(main_contour, epsilon, True)
    #
    #     # 6. 转换为归一化坐标
    #     boundary_points = []
    #     for point in approx:
    #         x, y = point[0]
    #         norm_x = x / 1.0
    #         norm_y = y / 1.0
    #         boundary_points.append([norm_x, norm_y])
    #
    #     # 确保多边形闭合 (首尾点相同)
    #     if len(boundary_points) > 0 and boundary_points[0] != boundary_points[-1]:
    #         boundary_points.append(boundary_points[0])
    #
    #     info['boundary'] = boundary_points
    # else:
    #     info['boundary'] = []
    #     print(f"警告: 无法为文件 {line} 找到有效轮廓")
    # # info['boundary'] = boundary.tolist() # 边界拐点


    fp_id = line.split("/")[-1].split(".")[0]
    # print(info)
    ## saving json files
    with open(f"rplan_json/{fp_id}.json","w") as f:

         json.dump(info, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 3D Visualization")
    parser.add_argument("--path", required=False,default="../RPLAN_dataset/dataset/floorplan_dataset/1.png",
                        help="dataset path", metavar="DIR")
    return parser.parse_args()


def main():
    args = parse_args()
    line=args.path
    # raster_to_json(line, print_door_warning=False)

    try:
        raster_to_json(line, print_door_warning=False)
    except (AssertionError, ValueError, IndexError) as e:
        fp_id = line.split("/")[-1].split(".")[0]

        with open(f"failed_rplan_json/{fp_id}", "w") as f:
            f.write(str(e))

if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
