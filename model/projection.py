# # import torch as th
# # import torch.nn.functional as F
# # import cv2 as cv
# # from shapely.geometry import Polygon
# # from shapely.geometry.base import geom_factory
# # from shapely.geos import lgeos
# # from scipy import ndimage
# #
# # from shapely.geometry import MultiPoint
# # import math
# # class ProjectionV0:
# #     def __init__(self, eps=1e-6):
# #         """
# #         :param boundary_mask: Tensor of shape [H, W] or [B, H, W], where 1=inside, 0=outside
# #         :param eps: small tolerance for projection convergence
# #         """
# #         self.eps = eps
# #
# #     def setup_boundary_coords(self,boundary_mask): # boundary_mask维度是[B,3,H,W]
# #         """Precompute all interior points coordinates"""
# #         if boundary_mask.dim() == 4:
# #             # Use first batch if multiple
# #             mask = boundary_mask[:,0,:,:]
# #         else:
# #             mask = boundary_mask
# #
# #         B, H, W = mask.shape
# #         # Create coordinate grid
# #         y_coords, x_coords = torch.meshgrid(
# #             torch.arange(H, device=boundary_mask.device),
# #             torch.arange(W, device=boundary_mask.device),
# #             indexing='ij'
# #         )
# #         x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
# #         y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
# #
# #         interior_mask = mask > 0.5
# #
# #         # 收集每个批次的坐标（返回列表，每个元素是该批次的坐标 [N, 2]）
# #         batch_coords = []
# #         for b in range(B):
# #             coords = torch.stack([
# #                 x_coords[b][interior_mask[b]],
# #                 y_coords[b][interior_mask[b]]
# #             ], dim=1).float()
# #             batch_coords.append(coords)
# #         return batch_coords  # 返回列表，长度为B，每个元素是 [N_b, 2]
# #
# #     def differentiable_convex_hull_area(self,points, num_directions=128, alpha=50.0):
# #         """计算点集的凸包面积（可微近似）"""
# #         device = points.device
# #         thetas = torch.linspace(0, 2 * math.pi, num_directions,device = points.device)
# #         dirs = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)  # (num_directions, 2)
# #
# #         # 计算支撑函数 h(theta)
# #         proj = points @ dirs.T  # (N, D)
# #         h = (1.0 / alpha) * torch.logsumexp(alpha * proj, dim=0)  # (D,)
# #
# #         # 边界点 (r(θ) * cosθ, r(θ) * sinθ)
# #         boundary_x = h * torch.cos(thetas)
# #         boundary_y = h * torch.sin(thetas)
# #
# #         # 用 shoelace formula 计算面积
# #         x1 = boundary_x
# #         y1 = boundary_y
# #         x2 = torch.roll(boundary_x, -1)
# #         y2 = torch.roll(boundary_y, -1)
# #         area = 0.5 * torch.abs(torch.sum(x1 * y2 - x2 * y1))
# #
# #         return area, thetas, boundary_x, boundary_y
# #
# #
# #     def apply(self, points,model_kwargs):
# #         """
# #         Project points onto the boundary interior using GPU-based nearest neighbor search.
# #         :param points: Tensor of shape [B, 2, N] - (x, y) coordinates of N turning points
# #         :return: projected points of same shape
# #         """
# #         B, _, N = points.shape
# #         boundary_mask = model_kwargs['boundary']
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状应为[B, N]
# #         valid_mask_bool = (valid_mask == 0)  # [B, N]
# #
# #         interior_coords_list = self.setup_boundary_coords(boundary_mask)
# #
# #         # Reshape points to [B*N, 2]
# #         # points_flat = points.permute(0, 2, 1).reshape(-1, 2)  # [B*N, 2]
# #
# #         # 初始化投影结果（先复制原始点，后续仅修改有效点）
# #         points = ((points.clone()/2.)+0.5)*256.
# #         projected_points = points.clone()
# #         # print(points.shape)
# #
# #         # 逐批次处理（因为每个批次的边界内坐标数量可能不同）
# #         for b in range(B):
# #             # 当前批次的有效拐点掩码
# #             batch_valid = valid_mask_bool[b]  # [N]
# #             # 提取当前批次的有效拐点
# #             valid_points = points[b, :, batch_valid].permute(1, 0)  # [N_valid, 2]
# #             # 当前批次的边界内坐标
# #             interior_coords = interior_coords_list[b]  # [N_interior, 2]
# #             # 计算有效点到边界内所有点的距离
# #             diff = valid_points.unsqueeze(1) - interior_coords.unsqueeze(0)  # [N_valid, N_interior, 2]
# #             distances = torch.norm(diff, dim=2)  # [N_valid, N_interior]
# #             # print("====",b,diff.shape,distances.shape) # torch.Size([70, 19280])
# #             # 找到最近邻点
# #             min_indices = torch.argmin(distances, dim=1)  # [N_valid]
# #             nearest_points = interior_coords[min_indices]  # [N_valid, 2]
# #             # print("****",b,nearest_points.shape)
# #             # 将投影结果放回到对应位置
# #             projected_points[b, :, batch_valid] = nearest_points.permute(1, 0)  # 恢复[2, N_valid]形状
# #
# #         projected_points = ((projected_points/256.)-0.5)*2.
# #         return projected_points
# #
# #     def apply_optimized(self, points, chunk_size=1000):
# #         """
# #         Optimized version for large numbers of interior points using chunking.
# #         :param points: Tensor of shape [B, 2, N]
# #         :param chunk_size: chunk size for memory efficiency
# #         :return: projected points
# #         """
# #         B, _, N = points.shape
# #         points_flat = points.permute(0, 2, 1).reshape(-1, 2)  # [B*N, 2]
# #         num_points = points_flat.shape[0]
# #         num_interior = self.interior_coords.shape[0]
# #
# #         projected_flat = torch.zeros_like(points_flat)
# #
# #         # Process in chunks to avoid OOM
# #         for i in range(0, num_points, chunk_size):
# #             chunk_points = points_flat[i:i + chunk_size]
# #
# #             # Compute distances in chunks
# #             chunk_projected = []
# #             for j in range(0, num_interior, chunk_size):
# #                 chunk_interior = self.interior_coords[j:j + chunk_size]
# #
# #                 # Compute distances between chunk points and chunk interior points
# #                 diff = chunk_points.unsqueeze(1) - chunk_interior.unsqueeze(0)
# #                 distances_chunk = torch.norm(diff, dim=2)
# #
# #                 # Find min distances in this chunk
# #                 min_distances, min_indices = torch.min(distances_chunk, dim=1)
# #                 chunk_projected.append(chunk_interior[min_indices])
# #
# #             # For each point, find the best among all chunks
# #             chunk_projected = torch.stack(chunk_projected, dim=1)  # [chunk_size, num_chunks, 2]
# #             final_distances = torch.norm(chunk_points.unsqueeze(1) - chunk_projected, dim=2)
# #             best_indices = torch.argmin(final_distances, dim=1)
# #
# #             # Gather best projections
# #             best_projections = chunk_projected[torch.arange(chunk_points.shape[0]), best_indices]
# #             projected_flat[i:i + chunk_size] = best_projections
# #
# #         # Reshape back
# #         projected_points = projected_flat.reshape(B, N, 2).permute(0, 2, 1)
# #         return projected_points
# #
# #     def __call__(self, x, model_kwargs):
# #         """
# #         Interface for use in diffusion sampling loop.
# #         :param x: current state tensor
# #         :param model_kwargs: dict containing 'boundary' and 'points'
# #         :return: x unchanged (we modify model_kwargs in-place)
# #         """
# #         if 'points' not in model_kwargs:
# #             raise KeyError("model_kwargs must contain 'points' key for turning points.")
# #
# #         points = model_kwargs['points']
# #
# #         # Choose appropriate method based on interior points count
# #         if self.interior_coords.shape[0] > 10000:
# #             projected_points = self.apply_optimized(points)
# #         else:
# #             projected_points = self.apply(points)
# #
# #         model_kwargs['points'] = projected_points
# #         return x
# #
# #
# #
# #
# # class FastBoundaryAreaLoss():
# #     """优化版本：支持批量计算和GPU加速"""
# #
# #     def __init__(self, image_size=256, area_weight=0.01, boundary_weight=1,
# #                  precompute_distance=True):
# #         super().__init__()
# #         self.image_size = image_size
# #         self.area_weight = area_weight
# #         self.boundary_weight = boundary_weight
# #         self.precompute_distance = precompute_distance
# #
# #     def forward(self,point_pred, model_kwargs):
# #         B, N, _ = point_pred.shape
# #
# #         # 批量计算边界损失
# #         boundary_loss = self.boundary_weight * self.batch_boundary_loss(point_pred, model_kwargs)
# #
# #         # 批量计算面积损失
# #         # area_loss = self.area_weight * self.batch_area_loss(point_pred, model_kwargs)
# #         area_loss = self.area_weight * self.enhanced_distance_field_loss(point_pred, model_kwargs)
# #
# #         total_loss = (
# #                  boundary_loss +
# #                  area_loss
# #         )
# #
# #         return {
# #             'total_loss': total_loss,
# #             'boundary_loss': boundary_loss,
# #             'area_loss': area_loss
# #         }
# #
# #     def setup_boundary_coords(self,boundary_mask): # boundary_mask维度是[B,3,H,W]
# #         """Precompute all interior points coordinates"""
# #         mask = boundary_mask[:, 0, :, :]
# #         B, H, W = mask.shape
# #         # Create coordinate grid
# #         y_coords, x_coords = torch.meshgrid(
# #             torch.arange(H, device=boundary_mask.device),
# #             torch.arange(W, device=boundary_mask.device),
# #             indexing='ij'
# #         )
# #         x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
# #         y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
# #
# #         interior_mask = mask > 0.5
# #
# #         # 收集每个批次的坐标（返回列表，每个元素是该批次的坐标 [N, 2]）
# #         batch_coords = []
# #         for b in range(B):
# #             coords = torch.stack([
# #                 x_coords[b][interior_mask[b]],
# #                 y_coords[b][interior_mask[b]]
# #             ], dim=1).float()
# #             batch_coords.append(coords)
# #         return batch_coords  # 返回列表，长度为B，每个元素是 [N_b, 2]
# #
# #     def batch_boundary_loss(self, points, model_kwargs):
# #         B, N,_ = points.shape
# #         boundary_mask = model_kwargs['boundary']
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状应为[B, N]
# #         valid_mask_bool = (valid_mask == 0)  # [B, N]
# #         interior_coords_list = self.setup_boundary_coords(boundary_mask) # 边界内的所有点
# #
# #         points = ((points.clone() / 2.) + 0.5) * 256.
# #
# #         total_loss =0.
# #         # 逐批次处理（因为每个批次的边界内坐标数量可能不同）
# #         for b in range(B):
# #             # 当前批次的有效拐点掩码
# #             batch_valid = valid_mask_bool[b]  # [N]
# #             # 提取当前批次的有效拐点
# #             valid_points = points[b, batch_valid,:]#.permute(1, 0)  # [N_valid, 2]
# #             # 当前批次的边界内坐标
# #             interior_coords = interior_coords_list[b]  # [N_interior, 2]
# #             # 计算有效点到边界内所有点的距离
# #             diff = valid_points.unsqueeze(1) - interior_coords.unsqueeze(0)  # [N_valid, N_interior, 2]
# #             distances = torch.norm(diff, dim=2)  # [N_valid, N_interior]
# #             min_distances, min_indices = torch.min(distances, dim=1)
# #             # print("====",b,diff.shape,distances.shape) # torch.Size([70, 19280])
# #             # 找到最近邻点
# #             min_indices = torch.argmin(distances, dim=1)  # [N_valid]
# #             nearest_points = interior_coords[min_indices]  # [N_valid, 2]
# #             total_loss += (min_distances.mean())/256
# #         return total_loss/B
# #
# #     # def create_distance_field(self, boundary_mask):
# #     #     """创建可导的距离场（包含边界距离和中心信息）"""
# #     #     B, _, H, W = boundary_mask.shape
# #     #     device = boundary_mask.device
# #     #
# #     #     # 创建坐标网格
# #     #     y_coords, x_coords = torch.meshgrid(
# #     #         torch.arange(H, device=device),
# #     #         torch.arange(W, device=device),
# #     #         indexing='ij'
# #     #     )
# #     #
# #     #     distance_fields = []
# #     #     center_fields = []  # 新增：中心距离场
# #     #
# #     #     for b in range(B):
# #     #         mask = (boundary_mask[b, 1] == 0.5)
# #     #
# #     #         if torch.any(mask):
# #     #             # 找到边界点
# #     #             boundary_y, boundary_x = torch.where(mask)
# #     #             boundary_points = torch.stack([boundary_x, boundary_y], dim=1).float()
# #     #
# #     #             # 计算图像中心点
# #     #             center_point = torch.tensor([W / 2, H / 2], device=device, dtype=torch.float32)
# #     #
# #     #             # 计算每个网格点到边界点的距离
# #     #             grid_points = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
# #     #             diff_boundary = grid_points.unsqueeze(1) - boundary_points.unsqueeze(0)
# #     #             distances_boundary = torch.norm(diff_boundary, dim=2)  # [H*W, N_boundary]
# #     #
# #     #             # 计算每个网格点到中心点的距离
# #     #             diff_center = grid_points - center_point
# #     #             distances_center = torch.norm(diff_center, dim=1)  # [H*W]
# #     #
# #     #             # 可导的最小边界距离计算
# #     #             min_boundary_distances = torch.min(distances_boundary, dim=1)[0].reshape(H, W)
# #     #             center_distances = distances_center.reshape(H, W)
# #     #
# #     #             distance_fields.append(min_boundary_distances)
# #     #             center_fields.append(center_distances)
# #     #         else:
# #     #             # 如果没有边界，使用默认值
# #     #             distance_fields.append(torch.zeros(H, W, device=device))
# #     #             center_fields.append(torch.zeros(H, W, device=device))
# #     #
# #     #     return distance_fields, center_fields  # 返回两个距离场
# #     #
# #     # def enhanced_distance_field_loss(self, points, model_kwargs,
# #     #                                  boundary_weight=1.0, center_weight=0.5,
# #     #                                  center_type='image_center'):
# #     #     """
# #     #     增强的距离场边界损失：边界距离 + 中心点距离
# #     #
# #     #     Args:
# #     #         points: 预测的点坐标 [B, N, 2]
# #     #         model_kwargs: 包含边界掩码等信息的字典
# #     #         boundary_weight: 边界距离权重
# #     #         center_weight: 中心距离权重
# #     #         center_type: 中心点类型 ('image_center', 'boundary_centroid', 'dynamic')
# #     #     """
# #     #     B, N, _ = points.shape
# #     #     boundary_mask = model_kwargs['boundary']
# #     #     valid_mask = model_kwargs['syn_src_key_padding_mask']
# #     #
# #     #     # 创建双距离场
# #     #     boundary_fields, center_fields = self.create_distance_field(boundary_mask)  # 距离场是在256*256上
# #     #
# #     #     points_normalized = ((points / 2.) + 0.5) * 256.  # [0, 256]
# #     #     total_loss = 0.0
# #     #
# #     #     for b in range(B):
# #     #         valid_points = points_normalized[b, valid_mask[b] == 0]
# #     #         if len(valid_points) == 0:
# #     #             continue
# #     #
# #     #         # 边界距离查询
# #     #         boundary_grid = boundary_fields[b].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
# #     #         points_grid = (valid_points / 256.0) * 2 - 1  # 映射到[-1, 1]
# #     #         points_grid = points_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N_valid, 2]
# #     #
# #     #         boundary_distances = F.grid_sample(boundary_grid, points_grid, align_corners=True)
# #     #         boundary_distances = boundary_distances.squeeze()  # [N_valid]
# #     #
# #     #         # 中心距离查询
# #     #         center_grid = center_fields[b].unsqueeze(0).unsqueeze(0)
# #     #         center_distances = F.grid_sample(center_grid, points_grid, align_corners=True)
# #     #         center_distances = center_distances.squeeze()  # [N_valid]
# #     #
# #     #         # 组合损失
# #     #         boundary_loss = boundary_distances.mean() / 256.0
# #     #         center_loss = center_distances.mean() / 256.0
# #     #
# #     #         total_loss += (boundary_weight * boundary_loss +
# #     #                        center_weight * center_loss)
# #     #
# #     #     return total_loss / B
# #
# #     def create_distance_field(self, boundary_mask):
# #         """优化的距离场计算，避免大矩阵操作"""
# #         B, _, H, W = boundary_mask.shape
# #         device = boundary_mask.device
# #
# #         # 预计算坐标网格 (优化点1: 避免重复计算)
# #         y_coords, x_coords = torch.meshgrid(
# #             torch.arange(H, device=device),
# #             torch.arange(W, device=device),
# #             indexing='ij'
# #         )
# #         grid_points = torch.stack([x_coords, y_coords], dim=-1).float()  # [H, W, 2]
# #
# #         # 图像中心点 (优化点2: 向量化计算)
# #         center_point = torch.tensor([W / 2, H / 2], device=device).view(1, 1, 2)
# #
# #         boundary_fields = []
# #         center_fields = []
# #
# #         for b in range(B):
# #             mask = boundary_mask[b, 1] == 0.5
# #             if not mask.any():
# #                 boundary_fields.append(torch.zeros(H, W, device=device))
# #                 center_fields.append(torch.zeros(H, W, device=device))
# #                 continue
# #
# #             # 获取边界点坐标 (优化点3: 直接使用坐标)
# #             boundary_y, boundary_x = torch.where(mask)
# #             boundary_points = torch.stack([boundary_x, boundary_y], dim=1).float()  # [N_b, 2]
# #
# #             # 计算中心距离场 (优化点4: 避免展开大矩阵)
# #             center_distances = torch.norm(grid_points - center_point, dim=-1)
# #             center_fields.append(center_distances)
# #
# #             # 优化边界距离计算 (关键优化)
# #             # 方法: 分块计算避免O(H*W*N)内存
# #             chunk_size = 256  # 根据显存调整
# #             min_dists = torch.full((H, W), float('inf'), device=device)
# #
# #             # 将边界点分块处理
# #             for i in range(0, len(boundary_points), chunk_size):
# #                 chunk = boundary_points[i:i + chunk_size]
# #                 # 向量化计算分块距离 [H, W, chunk_size]
# #                 diff = grid_points.view(H, W, 1, 2) - chunk.view(1, 1, -1, 2)
# #                 dists_chunk = torch.norm(diff, dim=-1)  # [H, W, chunk_size]
# #                 min_dists = torch.min(min_dists, dists_chunk.min(dim=-1)[0])
# #
# #             boundary_fields.append(min_dists)
# #
# #         return boundary_fields, center_fields
# #
# #     def enhanced_distance_field_loss(self, points, model_kwargs,
# #                                      boundary_weight=1.0, center_weight=0.5):
# #         """优化的距离场损失计算"""
# #         B, N, _ = points.shape
# #         boundary_mask = model_kwargs['boundary']
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']
# #
# #         # 创建距离场 (使用优化后的版本)
# #         boundary_fields, center_fields = self.create_distance_field(boundary_mask)
# #
# #         # 坐标转换 (保持原有逻辑)
# #         points_normalized = ((points / 2.) + 0.5) * 256.0  # [0, 256]
# #         total_loss = 0.0
# #         valid_count = 0
# #
# #         for b in range(B):
# #             if valid_mask[b] is None:
# #                 continue
# #             valid_idx = valid_mask[b] == 0
# #             if not valid_idx.any():
# #                 continue
# #
# #             valid_points = points_normalized[b, valid_idx]
# #             # 转换为整数坐标用于索引 (优化点: 避免grid_sample)
# #             grid_y = (valid_points[:, 1].clamp(0, 255)).long()
# #             grid_x = (valid_points[:, 0].clamp(0, 255)).long()
# #
# #             # 直接从距离场取值 (优化点: O(1)访问)
# #             boundary_dist = boundary_fields[b][grid_y, grid_x]
# #             center_dist = center_fields[b][grid_y, grid_x]
# #
# #             # 归一化距离
# #             boundary_loss = boundary_dist.mean() / 256.0
# #             center_loss = center_dist.mean() / 256.0
# #
# #             total_loss += boundary_weight * boundary_loss + center_weight * center_loss
# #             valid_count += 1
# #
# #         return total_loss / valid_count if valid_count > 0 else 0.0
# #
# #
# #     def differentiable_convex_hull_area(self,points, num_directions=128, alpha=50.0):
# #         """计算点集的凸包面积（可微近似）"""
# #         device = points.device
# #         thetas = torch.linspace(0, 2 * math.pi, num_directions,device = points.device)
# #         dirs = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)  # (num_directions, 2)
# #
# #         # 计算支撑函数 h(theta)
# #         proj = points @ dirs.T  # (N, D)
# #         h = (1.0 / alpha) * torch.logsumexp(alpha * proj, dim=0)  # (D,)
# #
# #         # 边界点 (r(θ) * cosθ, r(θ) * sinθ)
# #         boundary_x = h * torch.cos(thetas)
# #         boundary_y = h * torch.sin(thetas)
# #
# #         # 用 shoelace formula 计算面积
# #         x1 = boundary_x
# #         y1 = boundary_y
# #         x2 = torch.roll(boundary_x, -1)
# #         y2 = torch.roll(boundary_y, -1)
# #         area = 0.5 * torch.abs(torch.sum(x1 * y2 - x2 * y1))
# #
# #         return area, thetas, boundary_x, boundary_y
# #
# #     def batch_area_loss(self, points_pred,model_kwargs):
# #         B, N, _ = points_pred.shape
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状: [B, N]
# #         room_areas = model_kwargs['syn_room_areas']  # 形状: [B, N, 1]，真实房间面积
# #         room_indices = th.argmax(model_kwargs['syn_room_indices'], dim=-1)  # 形状: [B, N]，每个点所属房间索引
# #         room_type = th.argmax(model_kwargs['syn_room_types'], dim=-1)  # 形状: [B, N]，每个点所属房间索引
# #         valid_mask_bool = (valid_mask == 0)  # 形状: [B, N]，True表示有效点
# #         room_type_mask = (room_type != 0) & (room_type != 11) & (room_type != 12)
# #         # print(room_type_mask[0])
# #
# #         total_loss = 0.0
# #
# #         for b in range(B):
# #             area_gt = 0.
# #             batch_valid = valid_mask_bool[b]  # 形状: [N]
# #             room_type_mask_bool = room_type_mask[b]
# #             valid_room_areas = room_areas[b, batch_valid]  # 形状: [N_valid, 1]，真实面积
# #             valid_room_indices = room_indices[b, batch_valid]  # 形状: [N_valid]
# #
# #             unique_rooms = th.unique(valid_room_indices)
# #             for i,room_idx in enumerate(unique_rooms):
# #                 if i >= len(unique_rooms)//2:break
# #                 # 找到当前房间的有效点掩码
# #                 room_mask = (valid_room_indices == room_idx)
# #                 # 提取当前房间的真实面积（取平均值，因为同一房间面积应相同）
# #                 true_area_room_per = valid_room_areas[room_mask][0]
# #                 area_gt += true_area_room_per
# #                 # print(true_area_room_per)
# #
# #
# #             room_points_pred = points_pred[b, room_type_mask_bool, :]
# #             # print(room_points_pred)
# #             area_pred,_,_,_ = self.differentiable_convex_hull_area(room_points_pred)
# #             total_loss += th.sqrt(F.l1_loss(th.tensor(area_gt), area_pred))  # 需要排掉噪声的干扰
# #             # print(area_pred,area_gt)
# #         return total_loss / B
# #
# #     def batch_area_loss1(self, points, model_kwargs):
# #         """批量计算面积损失
# #
# #         损失计算逻辑：计算每个房间的预测面积与真实面积之间的差异，
# #         仅考虑有效点，并按房间维度平均
# #         """
# #         B, N, _ = points.shape
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状: [B, N]
# #         room_areas = model_kwargs['syn_room_areas']  # 形状: [B, N, 1]，真实房间面积
# #         room_indices = th.argmax(model_kwargs['syn_room_indices'], dim=-1)  # 形状: [B, N]，每个点所属房间索引
# #         room_type = th.argmax(model_kwargs['syn_room_types'], dim=-1)  # 形状: [B, N]，每个点所属房间索引
# #         valid_mask_bool = (valid_mask == 0)  # 形状: [B, N]，True表示有效点
# #
# #         total_loss = 0.0
# #
# #         for b in range(B):
# #             # 获取当前批次的有效掩码
# #             batch_valid = valid_mask_bool[b]  # 形状: [N]
# #
# #             # 提取当前批次的有效点及其房间索引
# #             valid_room_indices = room_indices[b, batch_valid]  # 形状: [N_valid]
# #             valid_room_areas = room_areas[b, batch_valid]  # 形状: [N_valid, 1]，真实面积
# #             valid_room_type = room_type[b, batch_valid]  # 形状: [N_valid, 1]，真实面积
# #
# #             # 获取当前批次的所有唯一房间索引
# #             unique_rooms = th.unique(valid_room_indices)
# #             batch_loss = 0.0
# #             room_count = 0
# #
# #             type_weights = {
# #                 1: 1.0,  # living_room
# #                 2: 0.5,  # kitchen
# #                 3: 1.0,  # bedroom
# #                 4: 0.5,  # bathroom
# #                 5: 0.5,  # balcony
# #                 6: 0.5,  # entrance
# #                 7: 0.5,  # dining room
# #                 8: 0.5,  # study room
# #                 10: 0.5,  # storage
# #                 11: 0.5,  # front door
# #                 12: 0.5,  # interior_door
# #                 13: 0.5,  # unknown
# #                 # 可以继续添加其他房间类型的权重
# #             }
# #
# #             # 对每个房间计算面积损失
# #             for i,room_idx in enumerate(unique_rooms):
# #                 if i >= len(unique_rooms)//2:break
# #                 # if room_idx == 0: continue
# #                 # 找到当前房间的有效点掩码
# #                 room_mask = (valid_room_indices == room_idx)
# #                 # 提取当前房间的真实面积（取平均值，因为同一房间面积应相同）
# #                 true_area = valid_room_areas[room_mask][0]
# #                 true_type = valid_room_type[room_mask][0]
# #                 # 获取当前房间类型的权重
# #                 weight = type_weights.get(true_type.item(), 0.001)
# #
# #                 # 计算当前房间的预测面积
# #                 # 这里假设points包含房间的坐标点，通过点集计算面积
# #                 # 实际实现可能需要根据你的数据格式调整
# #                 room_points = points[b, batch_valid, :][room_mask]  # 形状: [N_room, 2]
# #                 pred_area = self.shoelace_formula(room_points)
# #
# #                 # 计算当前房间的面积损失（使用L1损失）
# #                 room_loss = F.l1_loss(pred_area, true_area) #* weight
# #                 # print(room_idx,"true area:",true_area,"pred_area:",pred_area)
# #                 batch_loss += room_loss
# #                 room_count += 1
# #
# #             # 计算当前批次的平均损失
# #             if room_count > 0:
# #                 total_loss += batch_loss / room_count
# #
# #         # 计算所有批次的平均损失
# #         return total_loss / B
# #
# #     def shoelace_formula(self, vertices):
# #         # """优化的鞋带公式实现"""
# #         # if len(vertices) < 3:
# #         #     return torch.tensor(0.0, device=vertices.device)
# #
# #         # 确保多边形闭合
# #         if not torch.allclose(vertices[0], vertices[-1]):
# #             vertices = torch.cat([vertices, vertices[0:1]], dim=0)
# #
# #         x = vertices[:, 0]
# #         y = vertices[:, 1]
# #
# #         # 向量化计算
# #         area = 0.5 * torch.abs(
# #             torch.sum(x[:-1] * y[1:]) - torch.sum(x[1:] * y[:-1])
# #         )
# #
# #         return area
# #
# # class Projection:
# #     def __init__(self, eps=1e-6, alpha=0.5, beta=0.0):
# #         """
# #         :param eps: small tolerance
# #         :param alpha: weight for boundary attraction (0-1)
# #         :param beta: weight for distribution uniformity (0-1)
# #         """
# #         self.eps = eps
# #         self.alpha = alpha  # 边界吸引力权重
# #         self.beta = beta  # 分布均匀性权重 排斥力
# #         self.loss_computer = FastBoundaryAreaLoss()
# #
# #     def setup_boundary_coords1(self, boundary_mask):
# #         """Precompute boundary information including distance transform"""
# #         if boundary_mask.dim() == 4:
# #             mask = boundary_mask[:, 0, :, :]  # [B, H, W]
# #         else:
# #             mask = boundary_mask
# #
# #         B, H, W = mask.shape
# #         device = mask.device
# #
# #         # Create coordinate grid
# #         y_coords, x_coords = torch.meshgrid(
# #             torch.arange(H, device=device),
# #             torch.arange(W, device=device),
# #             indexing='ij'
# #         )
# #         x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
# #         y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
# #
# #         interior_mask = mask > 0.5
# #
# #         # Precompute distance transform for boundary attraction
# #         distance_maps = []
# #         gradient_maps_x = []
# #         gradient_maps_y = []
# #
# #         for b in range(B):
# #             # Convert to numpy for distance transform
# #             mask_np = interior_mask[b].cpu().numpy().astype(np.uint8)
# #
# #             # Distance transform: distance to boundary
# #             distance_map = ndimage.distance_transform_edt(mask_np)
# #
# #             # Compute gradient (direction towards interior)
# #             grad_y, grad_x = np.gradient(distance_map)
# #
# #             # Normalize gradients
# #             grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8
# #             grad_x /= grad_magnitude
# #             grad_y /= grad_magnitude
# #
# #             # Convert back to tensor
# #             distance_map = torch.from_numpy(distance_map).float().to(device)
# #             grad_x = torch.from_numpy(grad_x).float().to(device)
# #             grad_y = torch.from_numpy(grad_y).float().to(device)
# #
# #             distance_maps.append(distance_map)
# #             gradient_maps_x.append(grad_x)
# #             gradient_maps_y.append(grad_y)
# #
# #         # Collect interior coordinates
# #         batch_coords = []
# #         for b in range(B):
# #             coords = torch.stack([
# #                 x_coords[b][interior_mask[b]],
# #                 y_coords[b][interior_mask[b]]
# #             ], dim=1).float()
# #             batch_coords.append(coords)
# #
# #         return batch_coords, distance_maps, gradient_maps_x, gradient_maps_y
# #
# #     def setup_boundary_coords(self,boundary_mask): # boundary_mask维度是[B,3,H,W]
# #         """Precompute all interior points coordinates"""
# #         if boundary_mask.dim() == 4:
# #             # Use first batch if multiple
# #             mask = boundary_mask[:,0,:,:]
# #         else:
# #             mask = boundary_mask
# #
# #         B, H, W = mask.shape
# #         # Create coordinate grid
# #         y_coords, x_coords = torch.meshgrid(
# #             torch.arange(H, device=boundary_mask.device),
# #             torch.arange(W, device=boundary_mask.device),
# #             indexing='ij'
# #         )
# #         x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
# #         y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
# #
# #         interior_mask = mask > 0.5
# #
# #         # 收集每个批次的坐标（返回列表，每个元素是该批次的坐标 [N, 2]）
# #         batch_coords = []
# #         for b in range(B):
# #             coords = torch.stack([
# #                 x_coords[b][interior_mask[b]],
# #                 y_coords[b][interior_mask[b]]
# #             ], dim=1).float()
# #             batch_coords.append(coords)
# #         return batch_coords  # 返回列表，长度为B，每个元素是 [N_b, 2]
# #
# #     def compute_boundary_attraction(self, points, distance_maps, grad_x_maps, grad_y_maps,valid_mask):
# #         """Compute attraction force towards boundary interior"""
# #         B, _, N = points.shape
# #         attraction_forces = torch.zeros_like(points) # B,2,N
# #
# #         for b in range(B):
# #             valid_indices = torch.where(valid_mask[b] == 0)[0]
# #
# #             scaled_points = points[b, :, valid_indices]  # [2, N_valid]
# #
# #             # Scale points to image coordinates
# #             # scaled_points = ((points[b].clone() / 2.0) + 0.5) * 256.0
# #             # scaled_points = points[b].clone()
# #             scaled_points = scaled_points.clamp(0, 255).long()
# #
# #             x_coords = scaled_points[0].clamp(0, 255) #[2,N_vaild]
# #             y_coords = scaled_points[1].clamp(0, 255)
# #
# #             # Get distance values at point locations
# #             distances = distance_maps[b][y_coords, x_coords]
# #
# #             # Get gradient directions
# #             grad_x = grad_x_maps[b][y_coords, x_coords]
# #             grad_y = grad_y_maps[b][y_coords, x_coords]
# #
# #             # Attraction force is proportional to distance from boundary
# #             # Points near boundary get stronger pull inward
# #             force_magnitude = torch.exp(-distances / 10.0)  # Stronger near boundary
# #
# #             attraction_forces[b, 0,valid_indices] = force_magnitude * grad_x
# #             attraction_forces[b, 1,valid_indices] = force_magnitude * grad_y
# #
# #         return attraction_forces
# #
# #     def enforce_area_constraint(self,points, room_indices, room_areas, boundary_poly=None):
# #         """
# #         :param points: [N, 2] 拐点坐标
# #         :param room_indices: [N, R] one-hot 房间索引
# #         :param room_areas: [N] 每个拐点的目标面积 (同房间相同)
# #         :param boundary_poly: 可选 shapely Polygon，用于再次投影到边界内部
# #         """
# #         new_points = points.clone()
# #         room_ids = torch.argmax(room_indices, dim=-1)  # [N]
# #         for r in torch.unique(room_ids):
# #             if r == 0: continue
# #             mask = (room_ids == r)
# #
# #             poly_pts = new_points[mask]
# #             # ---- 1. 合法化 (凸包 or 角度排序) ----
# #             cx, cy = poly_pts.mean(dim=0)
# #             angles = torch.atan2(poly_pts[:, 1] - cy, poly_pts[:, 0] - cx)
# #             order = torch.argsort(angles)
# #             poly_pts = poly_pts[order]
# #
# #             # 计算当前面积 (Shoelace)
# #             x, y = poly_pts[:, 0], poly_pts[:, 1]
# #             area_pred = 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1) - torch.roll(x, -1) * y))
# #
# #             # 目标面积
# #             area_target = room_areas[mask][0]  # 所有拐点一样，取一个即可
# #
# #             if area_pred < 1e-6 or abs(area_target - area_pred)/area_target < 0.1:
# #                 continue
# #
# #             # 缩放因子
# #             scale = torch.sqrt(2*area_target / (area_target + area_pred + 1e-6))
# #
# #             # 中心
# #             cx, cy = poly_pts.mean(dim=0)
# #
# #             # 更新点
# #             poly_pts = torch.stack([
# #                 cx + scale * (x - cx),
# #                 cy + scale * (y - cy)
# #             ], dim=-1)
# #
# #             new_points[mask] = poly_pts.float()
# #
# #         return new_points
# #
# #     def compute_uniformity_force(self, points, valid_mask):
# #         """Compute repulsion force for uniform distribution"""
# #         B, _, N = points.shape
# #         repulsion_forces = torch.zeros_like(points)
# #
# #         for b in range(B):
# #             valid_indices = torch.where(valid_mask[b] == 0)[0]
# #
# #             valid_points = points[b, :, valid_indices]  # [2, N_valid]
# #
# #             # Compute pairwise distances
# #             diff = valid_points.unsqueeze(2) - valid_points.unsqueeze(1)  # [2, N, N]
# #             distances = torch.norm(diff, dim=0)  # [N, N] # 4. 计算两两之间的欧氏距离（基于坐标差）
# #
# #             # Avoid division by zero
# #             distances = distances + torch.eye(len(valid_indices), device=points.device) * 1e-6 # 5. 避免“自身距离为0”导致的除以0错误（给对角线加微小值）
# #
# #             # Repulsion force: 1/r^2
# #             repulsion = diff / (distances.unsqueeze(0) ** 3 + 1e-8) # 6. 计算排斥力：基于“1/r²”的力模型（距离越近，排斥力越强）
# #             repulsion = repulsion.sum(dim=2)  # Sum over other points # 7. 对每个点，求和所有其他点对它的排斥力（沿“其他点”维度求和）
# #
# #             # Normalize # 8. 归一化排斥力（让力的大小在0~1之间，避免个别点的力过大）
# #             repulsion_magnitude = torch.norm(repulsion, dim=0)   # 计算每个点排斥力的模长（形状[N_valid]）
# #             if repulsion_magnitude.max() > 0:
# #                 repulsion = repulsion / repulsion_magnitude.unsqueeze(0)
# #             # 9. 将有效点的排斥力填充到最终结果中（无效点保持初始0）
# #             repulsion_forces[b, :, valid_indices] = repulsion
# #
# #         return repulsion_forces # [B,2, N]
# #
# #     # 根据拐点绘制出mask
# #     def polygon_to_mask(self,poly, shape):
# #         """
# #         使用 cv2.fillPoly 将多边形 rasterize 成 mask
# #         :param polygons: list of shapely.Polygon
# #         :param shape: (H, W) 图像大小
# #         :return: mask (H, W) uint8
# #         """
# #         mask = np.zeros(shape, dtype=np.uint8)
# #
# #         coords = np.array(poly.exterior.coords, dtype=np.int32)
# #         coords = coords.reshape((-1, 1, 2))  # fillPoly 需要 (N,1,2) 格式
# #         cv.fillPoly(mask, [coords], 1)
# #         return mask
# #
# #     def compute_pixel_iou_cv(self,polys_out_door, bp, image_size=(256, 256)):
# #         """
# #         :param polys_out_door: list of coords (N,2)，预测多边形点
# #         :param bp: shapely Polygon，GT 多边形
# #         :param image_size: (H, W)
# #         """
# #         gt_mask = self.polygon_to_mask(bp, image_size)
# #         room_mask = np.zeros(image_size, dtype=np.uint8)
# #
# #         try:
# #             for coords in polys_out_door:  # 每个房间的拐点
# #                 if len(coords) >= 3:
# #                     poly = Polygon(coords)
# #                     if not poly.is_valid:
# #                         poly = poly.buffer(0)
# #
# #                     rm = self.polygon_to_mask(poly, image_size)
# #                     room_mask = cv.bitwise_or(room_mask, rm)
# #
# #
# #             # IoU 计算
# #             intersection = np.logical_and(room_mask, gt_mask).sum()
# #             union = np.logical_or(room_mask, gt_mask).sum()
# #             iou = intersection / union if union > 0 else 0.0
# #             return gt_mask,room_mask,iou
# #         except Exception as e:
# #             # print("异常：",gt_mask,room_mask)
# #             return gt_mask,room_mask,0.0
# #
# #     def get_iou(self,data,b,model_kwargs): # 第b张照片的数据
# #         prefix = 'syn_'
# #         polys = []
# #         types = []
# #         ids = []
# #         resolution = 256
# #         for j, point in enumerate(data): # 2,N个拐点
# #             # 选择参数键（真实/预测）
# #             mask_key = f'{prefix}src_key_padding_mask'
# #             if model_kwargs[mask_key][b][j] == 1:
# #                 continue
# #             point = point.cpu().data.numpy()
# #             if j == 0:
# #                 poly = []
# #             index_key = f'{prefix}room_indices'
# #             if j > 0 and (model_kwargs[index_key][b, j] != model_kwargs[index_key][b, j - 1]).any():
# #                 polys.append(poly)
# #                 types.append(c)
# #                 ids.append(id)
# #                 poly = []
# #             # 坐标转换
# #             pred_center = False
# #             if pred_center:
# #                 point = point / 2 + 1
# #                 point = point * (resolution // 2)
# #             else:
# #                 point = point / 2 + 0.5
# #                 point = point * resolution
# #             poly.append((point[0], point[1]))
# #             # 房间类型
# #             type_key = f'{prefix}room_types'
# #             c = np.argmax(model_kwargs[type_key][b][j - 1].cpu().numpy())
# #             # 房间索引
# #             id = np.argmax(model_kwargs[f'{prefix}room_indices'][b][j - 1].cpu().numpy())
# #         polys.append(poly)  # 所有房间的点
# #         types.append(c)
# #         ids.append(id)
# #         polys_out_door = []
# #         for cl in range(len(types)):
# #             if types[cl] not in [11, 12]:
# #                 polys_out_door.append(polys[cl])
# #
# #         boundary_key = 'boundary'
# #         boundary_mask = (model_kwargs[boundary_key][b][0] == 1.).cpu().detach().numpy().astype(np.uint8)
# #         contours, _ = cv.findContours(boundary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# #         boundary_contour = max(contours, key=cv.contourArea)
# #         boundary_contour = boundary_contour[:, 0, :]
# #         boundary_pts = [tuple(pt) for pt in boundary_contour]
# #         bp = Polygon(boundary_pts)
# #         if not bp.is_valid:
# #             bp = geom_factory(lgeos.GEOSMakeValid(bp._geom))
# #
# #         boundary_mask,room_mask,iou = self.compute_pixel_iou_cv(polys_out_door, bp, image_size=(resolution, resolution))
# #         return boundary_mask,room_mask,iou
# #
# #     def point_in_polygon(self, points, polygon):
# #         """
# #         points: [M, 2] 待判断点（M=H*W）
# #         polygon: [N, 2] 多边形顶点
# #         return: [M] 布尔张量，True=内部，False=外部
# #         """
# #         M = points.shape[0]
# #         N = polygon.shape[0]
# #         inside = torch.zeros(M, dtype=torch.bool, device=points.device)
# #
# #         for i in range(N):
# #             # 取当前边的两个顶点（闭合多边形：最后一条边连回起点）
# #             p1 = polygon[i]
# #             p2 = polygon[(i + 1) % N]
# #
# #             # 过滤y坐标不在当前边范围内的点（射线与边无交点）
# #             y_min = torch.min(p1[1], p2[1])
# #             y_max = torch.max(p1[1], p2[1])
# #             y_mask = (points[:, 1] > y_min) & (points[:, 1] <= y_max)
# #             if not y_mask.any():
# #                 continue
# #
# #             # 计算射线（水平向右）与当前边的交点x坐标
# #             # 避免除零：加1e-8
# #             denominator = (p2[1] - p1[1]) + 1e-8
# #             x_intersect = (points[y_mask, 1] - p1[1]) * (p2[0] - p1[0]) / denominator + p1[0]
# #
# #             # 异或更新：射线穿过边奇数次→内部，偶数次→外部
# #             inside[y_mask] ^= (points[y_mask, 0] < x_intersect)
# #
# #         return inside
# #
# #     def differentiable_point_in_polygon(self, points, polygon, temperature=0.1):
# #         """
# #         可微的点在多边形内判断
# #         """
# #         M = points.shape[0]
# #         N = polygon.shape[0]
# #         inside = torch.zeros(M, device=points.device)
# #
# #         for i in range(N):
# #             p1 = polygon[i]
# #             p2 = polygon[(i + 1) % N]
# #
# #             # 使用sigmoid进行软比较
# #             y_min = torch.min(p1[1], p2[1])
# #             y_max = torch.max(p1[1], p2[1])
# #
# #             # 软掩码：点在y范围内的概率
# #             in_y_range = torch.sigmoid((points[:, 1] - y_min) / temperature) * \
# #                          torch.sigmoid((y_max - points[:, 1]) / temperature)
# #
# #             # 计算交点x坐标
# #             denominator = (p2[1] - p1[1]) + 1e-8
# #             x_intersect = (points[:, 1] - p1[1]) * (p2[0] - p1[0]) / denominator + p1[0]
# #
# #             # 软比较：点在交点左侧的概率
# #             left_of_intersect = torch.sigmoid((x_intersect - points[:, 0]) / temperature)
# #
# #             # 累积贡献
# #             inside += in_y_range * left_of_intersect
# #
# #         # 使用模2运算的软版本
# #         return torch.sigmoid((inside % 2 - 0.5) / temperature)
# #
# #     def soft_rasterize(self,points, H, W, gamma=50.0):
# #         """
# #         points: [N,2] 房间多边形顶点 (x,y)，归一化到 [0,W],[0,H]
# #         return: [H,W] soft mask
# #         """
# #         device = points.device
# #         y_coords, x_coords = torch.meshgrid(
# #             torch.arange(H, device=device),
# #             torch.arange(W, device=device),
# #             indexing='ij'
# #         )
# #         grid = torch.stack([x_coords, y_coords], dim=-1).float()  # [H,W,2]
# #
# #         # 点到多边形边的最小距离
# #         dists = []
# #         for i in range(points.shape[0]):
# #             p1 = points[i]
# #             p2 = points[(i + 1) % points.shape[0]]
# #             v = p2 - p1
# #             w = grid - p1
# #             t = torch.clamp((w * v).sum(-1) / (v * v).sum(), 0, 1).unsqueeze(-1)  # [H,W,1]
# #             proj = p1 + t * v
# #             dist = torch.norm(grid - proj, dim=-1)  # [H,W]
# #             dists.append(dist)
# #         dist_map = torch.stack(dists, dim=0).min(0).values  # [H,W]
# #
# #         # 多边形内部判定 (winding rule / ray casting)
# #         inside = self.differentiable_point_in_polygon(grid.reshape(-1, 2), points).reshape(H, W)  # bool
# #         # signed_dist = dist_map * (~inside) - dist_map * inside  # 内部负，外部正
# #         weight = 1 - 2 * inside  # 内部→-1，外部→1
# #         signed_dist = inside * weight
# #
# #
# #         # soft mask
# #         mask = torch.sigmoid(-gamma * signed_dist)
# #         return mask
# #
# #
# #     def apply1(self, points, t,model_kwargs, num_iterations=1):
# #         """
# #         Improved projection with boundary awareness and distribution uniformity
# #         """
# #         B, _, N = points.shape
# #         boundary_mask = model_kwargs['boundary']
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']
# #         room_areas = model_kwargs['room_areas']
# #         valid_mask_bool = (valid_mask == 0)
# #
# #         # Precompute boundary information
# #         # interior_coords_list, distance_maps, grad_x_maps, grad_y_maps = self.setup_boundary_coords(boundary_mask)
# #
# #         # Initialize projected points
# #         projected_points = points.clone()
# #         # projected_points = ((points.clone() / 2.0) + 0.5) * 256.0
# #         current_points = points.clone()
# #
# #         # Multi-iteration optimization
# #         interior_coords_list = self.setup_boundary_coords(boundary_mask)
# #         for iteration in range(num_iterations):
# #             # Scale to image coordinates for processing
# #             current_points = ((current_points / 2.0) + 0.5) * 256.0
# #             # current_points = projected_points
# #
# #             # Compute boundary attraction forces 吸引力
# #             # attraction_forces = self.compute_boundary_attraction(
# #             #     current_points, distance_maps, grad_x_maps, grad_y_maps,valid_mask
# #             # )
# #
# #             # Compute uniformity repulsion forces 排斥力
# #             # repulsion_forces = self.compute_uniformity_force(current_points, valid_mask)
# #
# #             # Combine forces
# #             # total_forces = self.alpha * attraction_forces + self.beta * repulsion_forces
# #             # print(total_forces)
# #
# #             # Apply forces with adaptive step size
# #             # step_size = 2.0 * (1.0 - iteration / num_iterations)  # Decreasing step size
# #             new_points = current_points #+ step_size * total_forces
# #
# #             # Project to nearest interior point
# #             for b in range(B):
# #                 # TODO 边界
# #                 batch_valid = valid_mask_bool[b]
# #                 valid_points = new_points[b, :, batch_valid].permute(1, 0)
# #                 interior_coords = interior_coords_list[b]
# #                 # Find nearest interior points
# #                 diff = valid_points.unsqueeze(1) - interior_coords.unsqueeze(0)
# #                 distances = torch.norm(diff, dim=2)
# #                 min_indices = torch.argmin(distances, dim=1)
# #                 nearest_points = interior_coords[min_indices]
# #
# #                 projected_points[b, :, batch_valid] = (((nearest_points/256.)-0.5)*2.0).permute(1, 0)
# #
# #                 # if t[b] < 1000:
# #                 #     projected_points[b] = self.enforce_area_constraint(
# #                 #         projected_points[b].permute(1, 0),
# #                 #         model_kwargs['room_indices'][b],
# #                 #         model_kwargs['room_areas'][b]
# #                 #     ).permute(1, 0) #.unsqueeze(0)
# #             # Scale back to normalized coordinates
# #             current_points = ((current_points / 256.0) - 0.5) * 2.0
# #
# #         return projected_points
# #
# #
# #     def apply2(self, points, t,model_kwargs, num_iterations=3):
# #         """
# #         Improved projection with boundary awareness and distribution uniformity
# #         """
# #         points = points.permute([0, 2, 1])
# #         B, N,_ = points.shape
# #         boundary_mask = model_kwargs['boundary']
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']
# #         room_areas = model_kwargs['syn_room_areas']
# #         room_indices = th.argmax(model_kwargs['syn_room_indices'],dim=-1) # B,N
# #         valid_mask_bool = (valid_mask == 0)
# #
# #         x_dist = points.detach().clone().requires_grad_(True)
# #         target_points = points.detach().clone().requires_grad_(False)
# #         optimizer = torch.optim.Adam([x_dist], lr=0.1)
# #         with th.enable_grad():
# #             for iteration in range(num_iterations):
# #                 optimizer.zero_grad()
# #                 total_iou_loss = 0.
# #                 mse_loss = F.mse_loss((x_dist) * valid_mask_bool.unsqueeze(-1).float(), points * valid_mask_bool.unsqueeze(-1).float())
# #                 # mse_loss = F.mse_loss(x_dist, target_points)
# #                 # print(mse_loss.shape)
# #                 for b in range(B):
# #                     # boundary_mask,room_mask,iou = self.get_iou(x_dist[b], b, model_kwargs)
# #                     # iou = self.soft_get_iou(x_dist[b], b, model_kwargs)
# #                     # iou_loss += (1. - iou)
# #                     # iou_loss += F.binary_cross_entropy(room_mask, boundary_mask)
# #                     # 处理第b张图
# #                     batch_points = x_dist[b]  # [N_total, 2]：当前图的所有顶点
# #                     batch_indices = room_indices[b]  # [N_total]：当前图每个顶点的多边形ID
# #                     unique_rooms = torch.unique(batch_indices)  # 该图包含的多边形ID（如[0,1,2]）
# #                     room_iou_loss = 0.0
# #                     all_outer_product = torch.ones(256, 256, device=points.device)
# #                     for room_id in unique_rooms:
# #                         if room_id == 0:continue
# #                         # 1. 提取当前多边形的所有顶点
# #                         # 筛选出属于当前room_id的顶点索引
# #                         room_mask = (batch_indices == room_id)
# #                         room_points = batch_points[room_mask]  # [Ni, 2]：第room_id个多边形的顶点
# #
# #                         # 3. 生成当前多边形的软掩码
# #                         room_soft_mask = self.soft_rasterize(room_points, 256, 256, gamma=50)  # [H,W]
# #                         room_outer_mask = 1 - room_soft_mask  # [H, W]
# #                         all_outer_product *= room_outer_mask
# #                     batch_union_mask = 1 - all_outer_product  # [H, W]
# #
# #                     # 4. 计算当前多边形与目标边界的IoU
# #                     # 假设 boundary_mask[b] 中每个多边形区域已标注（可能需要根据room_id取对应区域）
# #                     # 若boundary_mask是整体掩码，则直接计算；若分区域，需修改此处
# #                     target_mask = boundary_mask[b][0]  # [H,W]：目标掩码（若分区域，需用room_id索引）
# #                     inter = (batch_union_mask * target_mask).sum()
# #                     union = (batch_union_mask + target_mask - batch_union_mask * target_mask).sum()
# #                     iou = inter / (union + 1e-6)
# #                     room_iou_loss += (1 - iou)
# #
# #                     total_iou_loss += room_iou_loss / (len(unique_rooms)-1)
# #
# #                 print("mse_loss:",mse_loss.item(),"iou_loss:",total_iou_loss/B)
# #                 total_loss = mse_loss + 0.1 * (total_iou_loss)/B
# #                 total_loss.backward()
# #                 optimizer.step()
# #
# #         return x_dist.detach().permute([0, 2, 1])
# #
# #     def apply(self, points, t,model_kwargs, num_iterations=5):
# #         """
# #         Improved projection with boundary awareness and distribution uniformity
# #         """
# #         points = points.permute([0, 2, 1])
# #         B, N,_ = points.shape
# #         boundary_mask = model_kwargs['boundary']
# #         valid_mask = model_kwargs['syn_src_key_padding_mask']
# #         room_areas = model_kwargs['syn_room_areas']
# #         room_indices = th.argmax(model_kwargs['syn_room_indices'],dim=-1) # B,N
# #         valid_mask_bool = (valid_mask == 0)
# #
# #         x_dist = points.detach().clone().requires_grad_(True)
# #         best_loss = float('inf')
# #         best_x_dist = x_dist.detach().clone()  # 初始化为初始值
# #
# #         # 动态损失权重
# #         boundary_weight = 1.0 # (t[0].item() / 1000.0) 权重越大、边界的约束能力越强，等于2的时候几乎能够完全约束住，但也会导致其他指标在变差
# #         area_weight = 1 * (1.0 - t[0].item() / 1000.0)
# #         initial_lr = 0.01 * (t[0].item() / 1000.0) # /100就不行
# #         print("t:",t[0],"initial_lr:",initial_lr)
# #
# #         # optimizer = torch.optim.Adam([x_dist], lr=0.0001)#, weight_decay=0.5) # 3 0.0001  4 0.001
# #         optimizer = torch.optim.Adam([x_dist], lr=initial_lr)#, weight_decay=0.5) # 3 0.0001  4 0.001
# #         with th.enable_grad():
# #             for iteration in range(num_iterations):
# #                 optimizer.zero_grad()
# #                 total_loss = 0.
# #                 coord_diff = x_dist - points  # [B,N,2]，每个元素是 (x_pred-x_gt, y_pred-y_gt)
# #                 offset_distance = torch.norm(coord_diff, dim=-1)  # [B,N]，每个元素是对应点的偏移距离
# #                 mse_loss = offset_distance[valid_mask_bool].mean()  # [K]，K是有效点总数（B*N中的有效个数）
# #                 loss = self.loss_computer.forward(x_dist,model_kwargs)
# #                 # total_loss = mse_loss + loss['area_loss'] # +  loss['boundary_loss']
# #                 total_loss = mse_loss + boundary_weight * loss['boundary_loss'] + area_weight * loss['area_loss']
# #
# #                 # total_loss = 100 * mse_loss + 100 * loss['boundary_loss'] + 0.1 * loss['area_loss']
# #                 # total_loss += 0.5 * loss['area_loss']
# #
# #                 # total_loss = 100 * loss['boundary_loss'] # + (1000 - t[0])/1000 * loss['area_loss'] # + 100 * mse_loss
# #                 # if 0:
# #                 #     total_loss += loss['area_loss']/100
# #
# #                 total_loss.backward()
# #                 optimizer.step()
# #
# #                 print("total_loss",total_loss.item(),
# #                       ",mse loss:", mse_loss.item(),
# #                       ",boundary_loss:",loss['boundary_loss'].item(),
# #                       ",area_loss:", loss['area_loss'].item())
# #                 # 更新最佳状态（只在损失变小时更新）
# #                 # if total_loss.item() < best_loss:
# #                 #     best_loss = total_loss.item()
# #                 #     best_x_dist = x_dist.detach().clone()  # 保存当前最佳结果
# #
# #         return x_dist.permute([0, 2, 1])
# #
# #     def __call__(self, x, model_kwargs):
# #         """Interface for diffusion sampling loop"""
# #         if 'points' not in model_kwargs:
# #             raise KeyError("model_kwargs must contain 'points' key")
# #
# #         points = model_kwargs['points']
# #         projected_points = self.apply(points, model_kwargs)
# #         model_kwargs['points'] = projected_points
# #         return x
# #
# #
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.distributions.categorical import Categorical
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # import numpy as np
# # from typing import List, Callable, Dict, Tuple
# #
# #
# # class ConstrainedDiscreteDiffusion:
# #
# #
# #     def constrained_projection(self,
# #                                x_dist: torch.Tensor,
# #                                constraints: List[Callable],
# #                                max_iter: int = 100,
# #                                lr: float = 0.1) -> torch.Tensor:
# #         """
# #         增强拉格朗日约束投影
# #
# #         Args:
# #             x_dist: 当前概率分布 [batch_size, seq_len, vocab_size]
# #             constraints: 约束函数列表
# #             max_iter: 最大迭代次数
# #             lr: 学习率
# #
# #         Returns:
# #             projected_dist: 投影后的分布
# #         """
# #         # 启用梯度跟踪
# #         x_dist = x_dist.clone().detach().requires_grad_(True)
# #
# #         # 初始化拉格朗日乘子和惩罚项
# #         lambda_params = [torch.tensor(0.0, requires_grad=False, device=self.device)
# #                          for _ in constraints]
# #         mu_params = [torch.tensor(1.0, requires_grad=False, device=self.device)
# #                      for _ in constraints]
# #
# #         optimizer = torch.optim.Adam([x_dist], lr=lr)
# #
# #         for iteration in range(max_iter):
# #             optimizer.zero_grad()
# #
# #             # 计算KL散度损失
# #             kl_loss = F.kl_div(F.log_softmax(x_dist, dim=-1),
# #                                F.softmax(x_dist.detach(), dim=-1),
# #                                reduction='batchmean')
# #
# #             # 计算约束违反损失
# #             constraint_loss = 0
# #             for i, constraint_fn in enumerate(constraints):
# #                 # 使用Gumbel-Softmax获得可微的近似采样
# #                 gumbel_softmax = self.gumbel_softmax(x_dist, temperature=0.1)
# #
# #                 # 计算约束违反程度
# #                 violation = constraint_fn(gumbel_softmax)
# #                 constraint_loss += lambda_params[i] * violation + mu_params[i] * violation ** 2
# #
# #             # 总损失
# #             total_loss = kl_loss + constraint_loss
# #             total_loss.backward()
# #             optimizer.step()
# #
# #             # 更新拉格朗日参数
# #             with torch.no_grad():
# #                 for i, constraint_fn in enumerate(constraints):
# #                     current_violation = constraint_fn(self.gumbel_softmax(x_dist, temperature=0.1))
# #                     lambda_params[i] += mu_params[i] * current_violation
# #                     mu_params[i] = min(mu_params[i] * 1.1, 1000)
# #
# #         return F.softmax(x_dist, dim=-1)
# #
# #     def gumbel_softmax(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
# #         """
# #         Gumbel-Softmax松弛
# #
# #         Args:
# #             logits: 输入logits
# #             temperature: 温度参数
# #
# #         Returns:
# #             松弛后的连续分布
# #         """
# #         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
# #         y = logits + gumbel_noise
# #         return F.softmax(y / temperature, dim=-1)
# #
# #     def generate(self,
# #                  prompt: str = None,
# #                  constraints: List[Callable] = None,
# #                  num_samples: int = 1,
# #                  guidance_scale: float = 3.0) -> List[str]:
# #         """
# #         生成受约束的文本
# #
# #         Args:
# #             prompt: 提示文本
# #             constraints: 约束函数列表
# #             num_samples: 生成样本数量
# #             guidance_scale: 引导强度
# #
# #         Returns:
# #             生成的文本列表
# #         """
# #         print("开始约束文本生成...")
# #         print(f"约束条件: {len(constraints) if constraints else 0} 个")
# #
# #         # 初始化噪声序列
# #         batch_size = num_samples
# #         xt = torch.randint(0, self.vocab_size, (batch_size, self.seq_length), device=self.device)
# #
# #         # 逐步去噪
# #         for t in range(self.timesteps, 0, -1):
# #             current_t = torch.full((batch_size,), t, device=self.device)
# #             xt = self.reverse_process_step(xt, current_t, constraints)
# #
# #             if t % 100 == 0:
# #                 print(f"时间步: {t}/{self.timesteps}")
# #
# #         # 解码为文本
# #         generated_texts = []
# #         for i in range(batch_size):
# #             tokens = xt[i].cpu().numpy()
# #             text = self.tokenizer.decode(tokens, skip_special_tokens=True)
# #             generated_texts.append(text)
# #
# #         return generated_texts
#
#
#
#
# import torch as th
# import torch.nn.functional as F
# import cv2 as cv
# from shapely.geometry import Polygon
# from shapely.geometry.base import geom_factory
# from shapely.geos import lgeos
# from shapely.geometry import MultiPoint
# import math
# import torch
# import numpy as np
# from scipy import ndimage
#
# import cv2 as cv
#
# class Projection:
#     def __init__(self,param=0.1, area_ratio=1.0,eps=1e-3):
#         """
#         :param boundary_mask: Tensor of shape [H, W] or [B, H, W], where 1=inside, 0=outside
#         :param eps: small tolerance for projection convergence
#         """
#         self.eps = eps
#         self.area_ratio = area_ratio
#         self.param = param
#         # print("==param:",self.param)
#
#
#     def setup_boundary_coords(self,boundary_mask): # boundary_mask维度是[B,3,H,W]
#         """Precompute all interior points coordinates"""
#         if boundary_mask.dim() == 4:
#             # Use first batch if multiple
#             mask = boundary_mask[:,0,:,:]
#         else:
#             mask = boundary_mask
#
#         B, H, W = mask.shape
#         # Create coordinate grid
#         y_coords, x_coords = torch.meshgrid(
#             torch.arange(H, device=boundary_mask.device),
#             torch.arange(W, device=boundary_mask.device),
#             indexing='ij'
#         )
#         x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
#         y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
#
#         interior_mask = mask > 0.5
#
#         # 收集每个批次的坐标（返回列表，每个元素是该批次的坐标 [N, 2]）
#         batch_coords = []
#         for b in range(B):
#             coords = torch.stack([
#                 x_coords[b][interior_mask[b]],
#                 y_coords[b][interior_mask[b]]
#             ], dim=1).float()
#             batch_coords.append(coords)
#         return batch_coords  # 返回列表，长度为B，每个元素是 [N_b, 2]
#
#     def differentiable_convex_hull_area(self,points, num_directions=360, alpha=10.0):
#         """计算点集的凸包面积（可微近似）"""
#         device = points.device
#         points = points - points.mean(dim=0, keepdim=True)
#         thetas = torch.linspace(0, 2 * math.pi, num_directions,device = points.device)
#         dirs = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)  # (num_directions, 2)
#
#         # 计算支撑函数 h(theta)
#         proj = points @ dirs.T  # (N, D)
#         h = (1.0 / alpha) * torch.logsumexp(alpha * proj, dim=0)  # (D,)
#
#         # 边界点 (r(θ) * cosθ, r(θ) * sinθ)
#         boundary_x = h * torch.cos(thetas)
#         boundary_y = h * torch.sin(thetas)
#
#         # 用 shoelace formula 计算面积
#         x1 = boundary_x
#         y1 = boundary_y
#         x2 = torch.roll(boundary_x, -1)
#         y2 = torch.roll(boundary_y, -1)
#         # area = 0.5 * torch.abs(torch.sum(x1 * y2 - x2 * y1))
#         area = 0.5 * torch.sum(x1 * y2 - x2 * y1)
#
#         return area, thetas, boundary_x, boundary_y
#
#     from shapely.geometry import MultiPoint
#     import torch
#
#     def convex_hull_area(self,points):
#         # points: [N, 2] tensor
#         pts = points.detach().cpu().numpy()
#         hull = MultiPoint(pts).convex_hull
#         return hull.area
#
#     # def room_area(self, point, room_idx):
#     #     """
#     #     计算每个房间的面积（有序多边形版本，忽略编号为0的无效房间）
#     #     :param point: Tensor [N, 2]，所有拐点坐标（按房间顺序拼接）
#     #     :param room_idx: Tensor [N]，每个拐点的房间编号（0为无效）
#     #     :return: total_area (Tensor)，所有有效房间面积之和
#     #     """
#     #
#     #     unique_rooms = torch.unique(room_idx)
#     #     total_area = torch.tensor(0.0, device=point.device)
#     #
#     #     for rid in unique_rooms:
#     #         if rid.item() == 0:continue
#     #         room_points = point[room_idx == rid]
#     #         p = Polygon(room_points)
#     #         if not p.is_valid:
#     #             p = geom_factory(lgeos.GEOSMakeValid(p._geom))
#     #
#     #         # x1, y1 = room_points[:, 0], room_points[:, 1]
#     #         # x2, y2 = torch.roll(x1, -1), torch.roll(y1, -1)
#     #         # area = 0.5 * torch.abs(torch.sum(x1 * y2 - x2 * y1))
#     #         total_area += p.area
#     #     return total_area
#
#     # def room_area(self,point, room_idx):
#     #     # 仅支持CPU
#     #     point = point.detach().cpu().numpy()
#     #     room_idx = room_idx.detach().cpu().numpy()
#     #
#     #     total_area = 0.0
#     #     for rid in set(room_idx):
#     #         if rid == 0. or rid == 11. or rid == 12.:
#     #             continue
#     #         room_points = point[room_idx == rid]
#     #         if len(room_points) < 3:
#     #             continue  # 至少3点
#     #         p = Polygon(room_points)
#     #         if not p.is_valid:
#     #             p = p.buffer(0)  # shapely推荐修复方式
#     #         total_area += p.area
#     #     return torch.tensor(total_area, dtype=torch.float32)
#
#     def boundary_area(self, boundary_mask):
#         mask = (boundary_mask[0] == 1.0)
#         contours, _ = cv.findContours(mask.cpu().detach().numpy().astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         boundary_contour = max(contours, key=cv.contourArea)
#         boundary_contour = boundary_contour[:, 0, :]
#         boundary_pts = [tuple(pt) for pt in boundary_contour]
#         bp = Polygon(boundary_pts)
#         bp_convex_hull_area = bp.convex_hull.area
#
#         # area = mask.float().sum()
#         return torch.tensor(bp_convex_hull_area, dtype=torch.float32)
#
#     def computer_scale(self,point,model_kwargs,b,batch_valid):
#         room_areas = model_kwargs['syn_room_areas'][b] * 256 * 256
#         room_type = th.argmax(model_kwargs['syn_room_types'],dim=-1)[b] # B,N
#         scale = np.ones(len(point), dtype=np.float32)
#
#         room_areas = room_areas.detach().cpu().numpy()
#         room_type = room_type.detach().cpu().numpy()
#         point = point.detach().cpu().numpy()
#
#         for rid in set(room_type):
#             if rid == 0.or rid == 11. or rid == 12. or rid == 1.:
#                 continue
#             # try:
#             room_points = point[room_type == rid]
#             area_gt = room_areas[room_type == rid][0]
#             pts = room_points #.detach().cpu().numpy()
#             # print("rid:",rid,",pts:",pts.shape,",area_gt:",area_gt)
#             # area_pred = MultiPoint(pts).convex_hull.area
#             area_pred = Polygon(pts).area
#             scale[room_type == rid] = area_gt / area_pred
#             # except:
#             #     pass
#         scale = np.clip(scale, 0.75, 1.5)
#         return scale
#
#
#     def apply(self, points, t, model_kwargs, num_iterations=5):
#
#         """
#         Project points onto the boundary interior using GPU-based nearest neighbor search.
#         :param points: Tensor of shape [B, 2, N] - (x, y) coordinates of N turning points
#         :return: projected points of same shape
#         """
#         B, _, N = points.shape
#         boundary_mask = model_kwargs['boundary']
#         valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状应为[B, N]
#         room_type = th.argmax(model_kwargs['syn_room_types'], dim=-1) #[B,N]
#         valid_mask_bool = (valid_mask == 0)  # [B, N]
#
#         interior_coords_list = self.setup_boundary_coords(boundary_mask)
#
#         # Reshape points to [B*N, 2]
#         # points_flat = points.permute(0, 2, 1).reshape(-1, 2)  # [B*N, 2]
#
#         # 初始化投影结果（先复制原始点，后续仅修改有效点）
#         points = ((points.clone()/2.)+0.5)*256.
#         projected_points = points.clone()
#         # print(points.shape)
#
#         # 逐批次处理（因为每个批次的边界内坐标数量可能不同）
#         for b in range(B):
#             # TODO 边界内投影
#             batch_valid = valid_mask_bool[b]  # [N]
#
#             # if t[0]>1000:
#             # 当前批次的有效拐点掩码
#             # 提取当前批次的有效拐点
#             valid_points = projected_points[b, :, batch_valid].permute(1, 0)  # [N_valid, 2]
#             # 当前批次的边界内坐标
#             interior_coords = interior_coords_list[b]  # [N_interior, 2]
#             # 计算有效点到边界内所有点的距离
#             diff = valid_points.unsqueeze(1) - interior_coords.unsqueeze(0)  # [N_valid, N_interior, 2]
#             distances = torch.norm(diff, dim=2)  # [N_valid, N_interior]
#             # # print("====",b,diff.shape,distances.shape) # torch.Size([70, 19280])
#             # # 找到最近邻点
#             min_indices = torch.argmin(distances, dim=1)  # [N_valid]
#             nearest_points = interior_coords[min_indices]  # [N_valid, 2]
#             # # print("****",b,nearest_points.shape)
#             # 将投影结果放回到对应位置
#             # alpha = 0.8   # 控制软化程度 约小约束力越大，
#             # dist_sq = torch.sum(diff ** 2, dim=2)  # [N_valid, N_interior]
#             # weights = torch.softmax(-alpha * dist_sq, dim=1)  # soft nearest neighbor
#             # nearest_points = weights @ interior_coords  # [N_valid, 2]
#
#             projected_points[b, :, batch_valid] = nearest_points.permute(1, 0)  # 恢复[2, N_valid]形状
#
#             # TODO 面积投影
#             # if room_area > self.eps:
#             device = points.device
#             if t[0]<100:
#                 valid_points = projected_points[b, :, batch_valid].permute(1, 0)  # [N_valid, 2]
#                 # # room_area, _, _, _ = self.differentiable_convex_hull_area(valid_points)
#                 # room_area = self.convex_hull_area(valid_points)
#                 # # room_area = self.room_area(projected_points[b].permute(1, 0), room_type[b])
#                 # boundary_area = self.boundary_area(boundary_mask[b])
#                 #
#                 # # 计算缩放系数，使得房间面积落在边界范围内
#                 # scale = boundary_area * self.area_ratio / room_area
#                 scale = self.computer_scale(projected_points[b].permute(1, 0),model_kwargs,b,batch_valid)
#                 scale = torch.tensor(scale, dtype=torch.float32, device=device)[batch_valid]
#                 # print("调整前比例",scale)
#                 scale = scale.unsqueeze(1)
#                 # λ 控制调整幅度：越小越接近1，越大越激进
#                 lambda_smooth = 0.003 #* (t[0]/200) # 0.005太大了 0.003可以
#                 # lambda_smooth = self.param
#                 # lambda_smooth = t[0]/1000
#                 scale = torch.exp(0.5 * lambda_smooth * torch.log(scale))
#                 # print("调整前比例",scale)
#
#                 # print("面积放缩系数：", scale, boundary_area, room_area)  # 0.2
#                 # scale = 1.0
#                 # ===== 中心放缩 =====
#                 center = valid_points.mean(dim=0, keepdim=True)
#                 scaled_points = (valid_points - center) * scale + center
#
#                 projected_points[b, :, batch_valid] = scaled_points.permute(1, 0)
#
#         projected_points = ((projected_points/256.)-0.5)*2.
#         return projected_points
#
#
#
#
#

import torch as th
import torch.nn.functional as F
import cv2 as cv
from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos
from shapely.geometry import MultiPoint
import math
import torch
import numpy as np
from scipy import ndimage

import cv2 as cv

class Projection:
    def __init__(self,param=0.1, area_ratio=1.0,eps=1e-3):
        """
        :param boundary_mask: Tensor of shape [H, W] or [B, H, W], where 1=inside, 0=outside
        :param eps: small tolerance for projection convergence
        """
        self.eps = eps
        self.area_ratio = area_ratio
        self.param = param
        # print("==param:",self.param)


    def setup_boundary_coords(self,boundary_mask): # boundary_mask维度是[B,3,H,W]
        """Precompute all interior points coordinates"""
        if boundary_mask.dim() == 4:
            # Use first batch if multiple
            mask = boundary_mask[:,0,:,:]
        else:
            mask = boundary_mask

        B, H, W = mask.shape
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=boundary_mask.device),
            torch.arange(W, device=boundary_mask.device),
            indexing='ij'
        )
        x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
        y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)

        interior_mask = mask > 0.5

        # 收集每个批次的坐标（返回列表，每个元素是该批次的坐标 [N, 2]）
        batch_coords = []
        for b in range(B):
            coords = torch.stack([
                x_coords[b][interior_mask[b]],
                y_coords[b][interior_mask[b]]
            ], dim=1).float()
            batch_coords.append(coords)
        return batch_coords  # 返回列表，长度为B，每个元素是 [N_b, 2]

    def differentiable_convex_hull_area(self,points, num_directions=360, alpha=10.0):
        """计算点集的凸包面积（可微近似）"""
        device = points.device
        points = points - points.mean(dim=0, keepdim=True)
        thetas = torch.linspace(0, 2 * math.pi, num_directions,device = points.device)
        dirs = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)  # (num_directions, 2)

        # 计算支撑函数 h(theta)
        proj = points @ dirs.T  # (N, D)
        h = (1.0 / alpha) * torch.logsumexp(alpha * proj, dim=0)  # (D,)

        # 边界点 (r(θ) * cosθ, r(θ) * sinθ)
        boundary_x = h * torch.cos(thetas)
        boundary_y = h * torch.sin(thetas)

        # 用 shoelace formula 计算面积
        x1 = boundary_x
        y1 = boundary_y
        x2 = torch.roll(boundary_x, -1)
        y2 = torch.roll(boundary_y, -1)
        # area = 0.5 * torch.abs(torch.sum(x1 * y2 - x2 * y1))
        area = 0.5 * torch.sum(x1 * y2 - x2 * y1)

        return area, thetas, boundary_x, boundary_y

    from shapely.geometry import MultiPoint
    import torch

    def convex_hull_area(self,points):
        # points: [N, 2] tensor
        pts = points.detach().cpu().numpy()
        hull = MultiPoint(pts).convex_hull
        return hull.area

    # def room_area(self, point, room_idx):
    #     """
    #     计算每个房间的面积（有序多边形版本，忽略编号为0的无效房间）
    #     :param point: Tensor [N, 2]，所有拐点坐标（按房间顺序拼接）
    #     :param room_idx: Tensor [N]，每个拐点的房间编号（0为无效）
    #     :return: total_area (Tensor)，所有有效房间面积之和
    #     """
    #
    #     unique_rooms = torch.unique(room_idx)
    #     total_area = torch.tensor(0.0, device=point.device)
    #
    #     for rid in unique_rooms:
    #         if rid.item() == 0:continue
    #         room_points = point[room_idx == rid]
    #         p = Polygon(room_points)
    #         if not p.is_valid:
    #             p = geom_factory(lgeos.GEOSMakeValid(p._geom))
    #
    #         # x1, y1 = room_points[:, 0], room_points[:, 1]
    #         # x2, y2 = torch.roll(x1, -1), torch.roll(y1, -1)
    #         # area = 0.5 * torch.abs(torch.sum(x1 * y2 - x2 * y1))
    #         total_area += p.area
    #     return total_area

    # def room_area(self,point, room_idx):
    #     # 仅支持CPU
    #     point = point.detach().cpu().numpy()
    #     room_idx = room_idx.detach().cpu().numpy()
    #
    #     total_area = 0.0
    #     for rid in set(room_idx):
    #         if rid == 0. or rid == 11. or rid == 12.:
    #             continue
    #         room_points = point[room_idx == rid]
    #         if len(room_points) < 3:
    #             continue  # 至少3点
    #         p = Polygon(room_points)
    #         if not p.is_valid:
    #             p = p.buffer(0)  # shapely推荐修复方式
    #         total_area += p.area
    #     return torch.tensor(total_area, dtype=torch.float32)

    def boundary_area(self, boundary_mask):
        mask = (boundary_mask[0] == 1.0)
        contours, _ = cv.findContours(mask.cpu().detach().numpy().astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        boundary_contour = max(contours, key=cv.contourArea)
        boundary_contour = boundary_contour[:, 0, :]
        boundary_pts = [tuple(pt) for pt in boundary_contour]
        bp = Polygon(boundary_pts)
        bp_convex_hull_area = bp.convex_hull.area

        # area = mask.float().sum()
        return torch.tensor(bp_convex_hull_area, dtype=torch.float32)

    def normalize_scale(self,scale_raw, low=0.75, high=1.5, eps=1e-6):
        s_min = np.min(scale_raw)
        s_max = np.max(scale_raw)
        scale_norm = low + (scale_raw - s_min) / (s_max - s_min + eps) * (high - low)
        return scale_norm
    def computer_scale(self,point,model_kwargs,b,batch_valid):
        room_areas = model_kwargs['syn_room_areas'][b] * 256 * 256
        room_type = th.argmax(model_kwargs['syn_room_types'],dim=-1)[b] # B,N
        scale = np.ones(len(point), dtype=np.float32)

        room_areas = room_areas.detach().cpu().numpy()
        room_type = room_type.detach().cpu().numpy()
        point = point.detach().cpu().numpy()

        for rid in set(room_type):
            if rid == 0. :#or rid == 11. or rid == 12.: #or rid == 1.:
                continue
            # try:
            room_points = point[room_type == rid]
            area_gt = room_areas[room_type == rid][0]
            pts = room_points #.detach().cpu().numpy()
            # print("rid:",rid,",pts:",pts.shape,",area_gt:",area_gt)
            # area_pred = MultiPoint(pts).convex_hull.area
            area_pred = Polygon(pts).area
            # s = area_gt / area_pred
            # if 1.5 > s > 0.75:
            # if 2.0 > s > 0.5:
            scale[room_type == rid] = np.sqrt(area_gt / area_pred)
            # print("area_gt:",area_gt,"area_pred:",area_pred,"scale:",scale[room_type == rid][0])


            # except:
            #     pass

        # scale = self.normalize_scale(scale, low=0.998, high=1.002)
        scale = self.normalize_scale(scale, low=0.9995, high=1.0005)
        # scale = self.normalize_scale(scale, low=0.95, high=1.05)
        return scale

    def computer_center(self, point, model_kwargs, b):
        # 获取房间类型（保持为 PyTorch 张量，不转 numpy）
        room_type = th.argmax(model_kwargs['syn_room_types'], dim=-1)[b]  # shape: (N,)

        # 初始化中心坐标（复制原始点坐标的形状和类型）
        center = point.clone()

        # 获取所有唯一的房间类型（在张量上操作）
        unique_room_ids = th.unique(room_type)

        for rid in unique_room_ids:
            # 用张量掩码筛选同一房间的点（保持张量类型）
            mask = (room_type == rid)
            room_points = point[mask]

            # 计算均值（PyTorch 张量支持 dim 参数）
            room_center = room_points.mean(dim=0, keepdim=True)

            # 赋值回中心坐标
            center[mask] = room_center

        return center


    def apply(self, points, t, model_kwargs, num_iterations=5):

        """
        Project points onto the boundary interior using GPU-based nearest neighbor search.
        :param points: Tensor of shape [B, 2, N] - (x, y) coordinates of N turning points
        :return: projected points of same shape
        """
        B, _, N = points.shape
        boundary_mask = model_kwargs['boundary']
        valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状应为[B, N]
        room_type = th.argmax(model_kwargs['syn_room_types'], dim=-1) #[B,N]
        valid_mask_bool = (valid_mask == 0)  # [B, N]

        interior_coords_list = self.setup_boundary_coords(boundary_mask)

        # Reshape points to [B*N, 2]
        # points_flat = points.permute(0, 2, 1).reshape(-1, 2)  # [B*N, 2]

        # 初始化投影结果（先复制原始点，后续仅修改有效点）
        points = ((points.clone()/2.)+0.5)*256.
        projected_points = points.clone()
        # print(points.shape)

        # 逐批次处理（因为每个批次的边界内坐标数量可能不同）
        for b in range(B):
            # TODO 边界内投影
            batch_valid = valid_mask_bool[b]  # [N]
            if t[0]<1000:

                valid_points = projected_points[b, :, batch_valid].permute(1, 0)  # [N_valid, 2]
                # 当前批次的边界内坐标
                interior_coords = interior_coords_list[b]  # [N_interior, 2]
                # 计算有效点到边界内所有点的距离
                diff = valid_points.unsqueeze(1) - interior_coords.unsqueeze(0)  # [N_valid, N_interior, 2]
                distances = torch.norm(diff, dim=2)  # [N_valid, N_interior]
                # print("====",b,diff.shape,distances.shape) # torch.Size([70, 19280])
                # 找到最近邻点
                min_indices = torch.argmin(distances, dim=1)  # [N_valid]
                nearest_points = interior_coords[min_indices]  # [N_valid, 2]
                # alpha = 1/2
                alpha = self.param
                # nearest_points = (nearest_points+valid_points)/2
                nearest_points = alpha * nearest_points + (1 - alpha) * valid_points
                # print("****",b,nearest_points.shape)
                # 将投影结果放回到对应位置
                # alpha = 0.8   # 控制软化程度 约小约束力越大，
                # dist_sq = torch.sum(diff ** 2, dim=2)  # [N_valid, N_interior]
                # weights = torch.softmax(-alpha * dist_sq, dim=1)  # soft nearest neighbor
                # nearest_points = weights @ interior_coords  # [N_valid, 2]

                projected_points[b, :, batch_valid] = nearest_points.permute(1, 0)  # 恢复[2, N_valid]形状

            # TODO 面积投影
            # if room_area > self.eps:
            device = points.device
            # if t[0]<int(self.param):
            if 32<t[0]<200: # 【200，500】正在尝试
                valid_points = projected_points[b, :, batch_valid].permute(1, 0)  # [N_valid, 2]
                # # room_area, _, _, _ = self.differentiable_convex_hull_area(valid_points)
                # room_area = self.convex_hull_area(valid_points)
                # # room_area = self.room_area(projected_points[b].permute(1, 0), room_type[b])
                # boundary_area = self.boundary_area(boundary_mask[b])
                #
                # # 计算缩放系数，使得房间面积落在边界范围内
                # scale = boundary_area * self.area_ratio / room_area
                scale = self.computer_scale(projected_points[b].permute(1, 0),model_kwargs,b,batch_valid)
                scale = torch.tensor(scale, dtype=torch.float32, device=device)[batch_valid]
                # print("调整前比例",scale)
                scale = scale.unsqueeze(1)
                # λ 控制调整幅度：越小越接近1，越大越激进
                lambda_smooth = 0.003 #* (t[0]/200) # 0.005太大了 0.003可以
                # lambda_smooth = self.param
                # scale = torch.exp(lambda_smooth * torch.log(scale))
                # print("调整前比例",scale)

                # print("面积放缩系数：", scale, boundary_area, room_area)  # 0.2
                # scale = 1.0
                # ===== 中心放缩 =====
                # center = valid_points.mean(dim=0, keepdim=True)
                center = self.computer_center(projected_points[b].permute(1, 0),model_kwargs,b)
                center = center[batch_valid]
                scaled_points = (valid_points - center) * scale + center

                projected_points[b, :, batch_valid] = scaled_points.permute(1, 0)

        projected_points = ((projected_points/256.)-0.5)*2.
        return projected_points

