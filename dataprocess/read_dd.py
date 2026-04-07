import numpy as np 
from PIL import Image
import os
import sys
module_path = os.path.abspath("../")  # 可根据实际路径修改
if module_path not in sys.path:
    sys.path.append(module_path)
import mask_to_poly as mr
from floorplan import Floorplan
# 该函数的核心是对原始门掩码（door_img）进行像素级筛选和分类，排除无效门区域，并按像素特征给门分配唯一标识，便于后续区分不同门。
def read_door(door_img,img,tmp_diff):
	# 1. 复制原始门掩码，避免修改原数据（tmp3存结果，tmp4存原始门区域）
	tmp3=door_img.copy()
	tmp4=door_img.copy()
	# 2. 遍历图像所有像素（256x256），仅处理原始门区域（tmp4[k][h]==1）
	for k in range(256):
		for h in range(256):
			has=[]
			# 初始化一个长度为10的标记列表，用于记录门区域的像素值类型
			for knm in range(10):
				has.append(0)
			if(tmp4[k][h]==1):
				# 提取当前门像素周围的局部区域（以(k,h)为中心，边长为2*(tmp_diff+4)+1的正方形）
				# 取图像的第2通道（RGB的B通道，索引为2），推测该通道存储门的细节特征
				p=img[k-tmp_diff-4:k+tmp_diff+4,h-tmp_diff-4:h+tmp_diff+4,2]
				s=p[np.nonzero(p)]
				if(len(s)==0):
					continue
				# 提取局部区域中所有不重复的像素值（记录门的特征类型）
				r=[]
				kmmmm=s[0]
				for t in range(len(s)):
					if(t==0):
						r.append(s[t])
					elif(s[t] not in r):
						r.append(s[t])
				# 3. 根据特征值数量筛选有效门：
				# - 若特征值≥3：判定为无效门区域，将tmp3对应位置设为0（排除）
				if(len(r)>=3):
					 tmp3[k][h]=0
				# - 若特征值=2：判定为有效门，用has列表标记特征值，再转换为唯一整数标识
				elif(len(r)==2):
					has[r[0]]=1
					has[r[1]]=1
					# 将has列表按二进制权重转换为整数（如has[0]=1→1*1，has[1]=1→1*2，避免重复标识）
					tmp3[k][h]= has[0]*1+has[1]*2+has[3]*4+has[4]*8+has[5]*16+has[6]*32+has[7]*64+has[8]*128+has[9]*256
	# 4. 统一门标识：将tmp3中所有非零值（有效门）映射为连续整数（0,1,2...）
	s=np.unique(tmp3)
	tmp4=tmp3.copy()
	# 遍历所有门标识，将每个标识替换为其在“唯一值列表”中的索引（实现连续编号）
	for ks in range(len(s)):
		for k in range(256):
			for h in range(256):
				if(tmp3[k][h]==s[ks]):
					tmp4[k][h]=int(ks)
	# 返回优化后的门掩码（每个门有唯一连续标识）
	return tmp4

# 该函数根据k_d参数（排序模式），将输入的角点列表（corners，如房间 / 门的顶点）按特定规则排序，确保轮廓坐标的连续性（便于后续生成多边形）。
def sort_corners(corners ,k_d):
	coords=[]
	ind=[]
	if(k_d==0):
		for j in range(len(corners)):
			ind.append(0)
		for i in range(len(corners)):
			if (i==0):
				coords.append(corners[0])
				ind[i]=1

			elif(i%2==1):
				k=coords[i-1][0]
				for j in range(len(corners)):
					if(corners[j][0]==k)& ( ind[j]!=1):
						coords.append(corners[j])
						ind[j]=1
						break
			elif(i%2==0):
				k=coords[i-1][1]
				for j in range(len(corners)):
					if(corners[j][1]==k)&  (ind[j]!=1):
						coords.append(corners[j])
						ind[j]=1
						break
					
	if(k_d==1):
		for s in range(len(corners)):
			ind.append(0)
		p=0		
		for i in range(len(corners)):
			if (i%4==0):
				coords.append(corners[i])
				ind[i]=1
				p=p+1
			elif(i%2==1):
				k=coords[i-1][0]
				tk=coords[i-1][1]  
				pp=0
				for j in range(len(corners)):
					if(corners[j][0]==k)& (ind[j]!=1):
						if(pp==0) :
							pp=pp+1
							coords.append(corners[j])
							fn=j
						else:
							if (abs(corners[j][1]-tk)<=abs(coords[i][1]-tk)) &(abs(corners[j][1]-tk)!=0):
								coords[i]=corners[j]
								fn=j
				ind[fn]=ind[fn]+1
				fn=-1		
				p=p+1
			elif(i%2==0)& (i%4!=0):
				p=p+1
				k=coords[i-1][1]
				tk=coords[i-1][0] 
				pp=0
				for j in range(len(corners)):
					if(corners[j][1]==k) & (ind[j]!=1):
						if(pp==0):
							pp=pp+1
							coords.append(corners[j])
							fn=j  						
						else:
							if (abs(corners[j][0]-tk)<=abs(coords[i][0]-tk)) &(abs(corners[j][0]-tk)!=0):
								coords[i]=corners[j]
								fn=j  
				ind[fn]=ind[fn]+1 
				fn=-1
	return coords

# 该函数是整个代码的入口，负责读取图像文件、提取房间 / 墙体 / 门 / 入口、转换数据格式、验证有效性，最终返回标准化结果。line为输入参数，代表图像文件的路径。
def read_data(line):
	poly=[] # 存储每个区域（房间/门/入口）的角点数量（用于验证是否为有效多边形）
	# 1. 读取图像并转换为NumPy数组
	img = np.asarray(Image.open(line)) # (256, 256, 4)
	dec=0 # 计数变量：记录无效区域数量（用于后续标识修正）
	img_room_type=img[:,:,1] # 类别
	img_room_number=img[:,:,2] # order 房间的序号
	# 3. 提取墙体掩码（第1通道中值为16的像素标记为“墙体”，值为17的标记为“门”）
	wall_img=np.zeros((256, 256))
	for k in range(256):
		for h in range(256):
			if(img_room_type[k][h]==16):
				wall_img[k][h]=16	# 标记为墙体
			if(img_room_type[k][h]==17):
				wall_img[k][h]=10 # 标记为门（临时值，后续单独处理）
	# 4. 提取每个房间的掩码与类型（按房间编号分组）
	room_no=img_room_number.max() # 房间总数（最大编号即为总数）
	room_imgs=[] # 存储每个房间的掩码（每个房间对应一个256x256矩阵，1=房间区域，0=其他）
	rm_types=[] # 存储每个房间的类型（转换为HouseGAN++模型兼容的类型编码）
	for i in range(room_no): # 遍历每个房间（编号1~room_no）
		# 初始化当前房间的掩码
		room_img=np.zeros((256, 256))
		for k in range(256):
			for h in range(256):
				if(img_room_number[k][h]==i+1):
					room_img[k][h]=1
					k_=k
					h_=h
		rm_t=img_room_type[k_][h_] # 4.1 获取当前房间的原始类型（通过已记录的像素位置取第1通道值）

		#changing rplan rooms_type to housegan++ rooms_type
		if(rm_t==0):
			rm_types.append(1)		
		elif(rm_t==1):
			rm_types.append(3)
		elif(rm_t==2):
			rm_types.append(2)		
		elif(rm_t==3):
			rm_types.append(4)
		elif(rm_t==4):
			rm_types.append(7)
		elif(rm_t==5):
			rm_types.append(3)
		elif(rm_t==6):
			rm_types.append(8)
		elif(rm_t==7):
			rm_types.append(3)
		elif(rm_t==8):
			rm_types.append(3)
		elif(rm_t==9):
			rm_types.append(5)
		elif(rm_t==10):
			rm_types.append(6)
		elif(rm_t==11):
			rm_types.append(10)
		else:
			rm_types.append(16)
		room_imgs.append(room_img) # 将当前房间的掩码加入列表
	# 5. 房间掩码优化：去除孤立像素、填补微小空缺（确保房间区域是完整连通的）
	walls=[] # 存储墙体的线段坐标（格式：[起点行,起点列,终点行,终点列, ... 附加信息]） 房间的墙
	doors=[] # 存储门的线段坐标（格式同上）
	rm_type=rm_types
	for t in range(len(room_imgs)): # 遍历每个房间的掩码
		tmp=room_imgs[t] # 当前房间的掩码
		for k in range(254):
			for h in range(254):
				if(tmp[k][h]==1) & (tmp[k+1][h]==0) & (tmp[k+2][h]==1):
					tmp[k+1][h] =1				
		for k in range(254):
			for h in range(254):
				if(tmp[h][k]==1) & (tmp[h][k+1]==0) & (tmp[h][k+2]==1):
					tmp[h][k+1] =1				
		for k in range(254):
			for h in range(254):
				if(tmp[k][h]==0) & (tmp[k+1][h]==1) & (tmp[k+2][h]==0):
					tmp[k+1][h] =0				
		for k in range(254):
			for h in range(254):
				if(tmp[h][k]==0) & (tmp[h][k+1]==1) & (tmp[h][k+2]==0):
					tmp[h][k+1] =0

		room_imgs[t]=tmp
		poly2=mr.get_polygon(room_imgs[t]) # 获取房间区域的多边形对象
		coords_1=list(poly2.exterior.coords)  # 提取多边形的外轮廓坐标（格式：(列，行)）
		coords=[]
		for kn in range(len(coords_1)):
			coords.append([list(coords_1[kn])[1],list(coords_1[kn])[0],0,0,t,rm_type[t]])  # 转换坐标格式，将 (列，行) 转换为 (行，列)，并添加一些附加信息（房间索引、房间类型等），存储在coords中。
		p=0
		for c in range(len(coords)-1): # 遍历coords，将相邻角点组成的线段添加到walls列表中，线段信息包含起点和终点坐标以及一些附加信息。
			walls.append([coords[c][0],coords[c][1],coords[c+1][0],coords[c+1][1],-1,coords[c][5],coords[c][4],-1,0])
		p=len(coords)-1
		poly.append(p) # 将当前房间轮廓的角点数量添加到poly列表中。
	tmp=img[:,:,1]

	# 创建 256x256 的零矩阵door_img用于存储门的掩码，初始化doors_img列表用于存储每个门的单独掩码。
	door_img=np.zeros((256, 256))
	doors_img=[]		
	for k in range(256):
		for h in range(256):
			if(tmp[k][h]==17):
				door_img[k][h]=1
	# 遍历图像每个像素，在door_img中，将房间类型通道中值为 17 的像素标记为 1（表示门）。
	tmp=door_img
	rms_type=rm_type
	coords=[] # tmp设为door_img，rms_type为rm_type的引用，初始化coords列表用于存储门的角点。
	for k in range(2,254): # 遍历门掩码中值为 1 的像素，通过特定的邻域像素模式识别门的角点，并添加到coords列表中。
		for h in range(2,254):
			if(tmp[k][h]==1):
				if((tmp[k-1][h]==0) & (tmp[k-1][h-1]==0)&(tmp[k][h-1]==0)):
					coords.append([h,k,0,0])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h-1]==0)&(tmp[k][h-1]==0):
					coords.append([h,k,0,0])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0])
				elif (tmp[k-1][h]==0)&(tmp[k-1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0])					
				elif(tmp[k+1][h]==1)&(tmp[k][h+1]==1)& (tmp[k+1][h+1]==0):
					coords.append([h,k,0,0])					
				elif(tmp[k-1][h]==1)&(tmp[k][h+1]==1)& (tmp[k-1][h+1]==0):
					coords.append([h,k,0,0])					
				elif(tmp[k+1][h]==1)&(tmp[k][h-1]==1)&(tmp[k+1][h-1]==0) : 
					coords.append([h,k,0,0])				
				elif(tmp[k-1][h]==1) & (tmp[k][h-1]==1) & (tmp[k-1][h-1]==0):
					coords.append([h,k,0,0])
					
	tmp_diff=1000000  # 保存水平或者垂直距离的最小值，可能判断是水平门还是垂直门
	p_x_1=coords[0][0] # 计算门角点之间的最小水平距离（排除距离≤1 的点），存储在tmp_diff中。
	for k in range(1, len(coords)):
		p_x_2=coords[k][0]
		tmp_dif=abs(p_x_1-p_x_2)
		if(tmp_dif<tmp_diff)&(tmp_dif>1):
			tmp_diff=tmp_dif
	p_y_1=coords[0][1]
	for k in range(1, len(coords)): # 计算门角点之间的最小垂直距离（排除距离≤1 的点），更新tmp_diff为最小的距离值。
		p_y_2=coords[k][1]
		tmp_dif=abs(p_y_1-p_y_2)
		if(tmp_dif<tmp_diff)&(tmp_dif>1):
			tmp_diff=tmp_dif
	# 调用read_door函数，对门掩码进行优化和分类，得到每个门的单独掩码，存储在door_imgs中。
	door_imgs=read_door(door_img,img,tmp_diff) # 函数返回一个 256x256 的 NumPy 数组，每个独立的门区域被分配一个唯一的整数标识（如 1、2、3...），用于区分不同的门，无效的门区域被标记为 0（被过滤掉）
	door_no=int(door_imgs.max()) # 通过door_imgs的最大值得到门的数量door_no，初始化door_tp列表（未实际使用）。
	door_tp=[]
	for i in range(door_no): # 为每个门创建单独的掩码，将door_imgs中标识为i+1的像素在新掩码中标记为 1，并添加到doors_img列表中。
		door_img=np.zeros((256, 256))
		for k in range(256):
			for h in range(256):
				if(door_imgs[k][h]==i+1):door_img[k][h]=1
		doors_img.append(door_img)
	rmpn=len(doors_img)	 # 门的掩码，每个门被一个id值标记
	for t in range(len(doors_img)): # rmpn为门的实际数量，遍历每个门的掩码，如果掩码最大值≤0（表示无效门），则dec加 1 并跳过后续处理。
		tmp=doors_img[t]
		kpp=np.max(tmp)
		if(kpp<=0):
			dec=dec+1
			continue
		poly2=mr.get_polygon(doors_img[t])
		coords_1=list(poly2.exterior.coords) #
		coords=[]
		for kn in range(len(coords_1)):
			coords.append([list(coords_1[kn])[1],list(coords_1[kn])[0],0,0,t,17]) 
		p=0
		for c in range(len(coords)-1):
			walls.append([coords[c][0],coords[c][1],coords[c+1][0],coords[c+1][1],-1,17,len(rms_type)+coords[c][4]-dec,-1,0])
			doors.append([coords[c][0],coords[c][1],coords[c+1][0],coords[c+1][1]])
		p=len(coords)-1
		poly.append(p)
	tmp=img[:,:,1]
	en_img=np.zeros((256, 256))
	for k in range(256):
		for h in range(256):
			if(tmp[k][h]==15):
				en_img[k][h]=1
	tmp=en_img
	coords=[]
	for k in range(2,254):
		for h in range(2,254):
			if(tmp[k][h]==1):
				if((tmp[k-1][h]==0) & (tmp[k-1][h-1]==0)&(tmp[k][h-1]==0)):
					coords.append([h,k,0,0])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h-1]==0)&(tmp[k][h-1]==0):
					coords.append([h,k,0,0])
				elif (tmp[k+1][h]==0)&(tmp[k+1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0])
				elif (tmp[k-1][h]==0)&(tmp[k-1][h+1]==0)&(tmp[k][h+1]==0): 
					coords.append([h,k,0,0])
				elif(tmp[k+1][h]==1)&(tmp[k][h+1]==1)& (tmp[k+1][h+1]==0):
					coords.append([h,k,0,0])					
				elif(tmp[k-1][h]==1)&(tmp[k][h+1]==1)& (tmp[k-1][h+1]==0):
					coords.append([h,k,0,0])					
				elif(tmp[k+1][h]==1)&(tmp[k][h-1]==1)&(tmp[k+1][h-1]==0) : 
					coords.append([h,k,0,0])					
				elif(tmp[k-1][h]==1) & (tmp[k][h-1]==1) & (tmp[k-1][h-1]==0):
					coords.append([h,k,0,0])
					
	en_imgs=[]
	for i in range(1):

		door_img=np.zeros((256, 256))
		for k in range(256):
			for h in range(256):
				if(en_img[k][h]==i+1):
					en_img[k][h]=1
		en_imgs.append(en_img)
	for t in range(len(en_imgs)):
		tmp=en_imgs[t]
		kpp=np.max(tmp)
		if(kpp<=0):
			dec=dec+1
			continue
		poly2=mr.get_polygon(en_imgs[t])
		coords_1=list(poly2.exterior.coords) # 这个会输出n+1个点，第一个点和最后一个点相同
		coords=[]
		for kn in range(len(coords_1)):
			coords.append([list(coords_1[kn])[1],list(coords_1[kn])[0],0,0,t,15]) 
		p=0
		for c in range(len(coords)-1):
			walls.append([coords[c][0],coords[c][1],coords[c+1][0],coords[c+1][1],-1,15,rmpn+len(rms_type)+coords[c][4]-dec,-1,0])
			doors.append([coords[c][0],coords[c][1],coords[c+1][0],coords[c+1][1]])
		p=len(coords)-1
		poly.append(p)
	
	no_doors=int(len(doors)/4)
	rms_type=rm_type
	for i in range(no_doors-1):
		rms_type.append(17)
	rms_type.append(15)
	out=1
	### check if it was noy polygon 
	for i in range(len(poly)):
		if(poly[i]<4):
			out=-1
	if (len(doors)%4!=0):
			out=-3	
	##saving the name out standard (usable) layout		
	# if(out!=1):
	# 	h.write(line)
	"""f=open("output.txt", "a+")
	f.write(str(rms_type).strip('[]'))
	f.write("   ")
	f.write(str(len(rms_type)))
	if((len(rms_type)-no_doors)>(no_doors)):
		h1=open("door.txt","a+")
		out=-4	
		h1.write(line)"""
	assert(out==1), f"error in reading the file {line}, {out} but expected out==1"
	# TODO 边界拐点
	# fp = Floorplan(line)
	# fp = fp.to_dict(np.uint8)
	# boundary = fp['boundary'] # xy
	# rms_type，对应的房间类型（房间、门、前门）  poly：存储每个区域（房间、门、入口）的轮廓角点数量  doors：存储所有门和入口的轮廓线段坐标  walls：存储所有墙体、门、入口的轮廓线段坐标及附加信息
	return rms_type,poly,doors,walls,out #,boundary
	
