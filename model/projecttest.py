import torch as th
import torch.nn.functional as F
import torch
import math


class FastBoundaryAreaLoss():
    """优化版本：支持批量计算和GPU加速"""

    def __init__(self, image_size=256, area_weight=0.01, boundary_weight=0.1,
                 precompute_distance=True):
        super().__init__()
        self.image_size = image_size
        self.area_weight = area_weight
        self.boundary_weight = boundary_weight
        self.precompute_distance = precompute_distance

    def forward(self,point_pred, model_kwargs):
        B, N, _ = point_pred.shape

        # 批量计算边界损失
        boundary_loss = self.boundary_weight * self.batch_boundary_loss(point_pred, model_kwargs)

        # 批量计算面积损失
        area_loss = self.area_weight * self.batch_area_loss(point_pred, model_kwargs)

        total_loss = (
                 boundary_loss +
                 area_loss
        )

        return {
            'total_loss': total_loss,
            'boundary_loss': boundary_loss,
            'area_loss': area_loss
        }

    def setup_boundary_coords(self,boundary_mask): # boundary_mask维度是[B,3,H,W]
        """Precompute all interior points coordinates"""
        mask = boundary_mask[:, 0, :, :]
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

    def batch_boundary_loss(self, points, model_kwargs):
        B, N,_ = points.shape
        boundary_mask = model_kwargs['boundary']
        valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状应为[B, N]
        valid_mask_bool = (valid_mask == 0)  # [B, N]
        interior_coords_list = self.setup_boundary_coords(boundary_mask) # 边界内的所有点

        points = ((points.clone() / 2.) + 0.5) * 256.

        total_loss =0.
        # 逐批次处理（因为每个批次的边界内坐标数量可能不同）
        for b in range(B):
            # 当前批次的有效拐点掩码
            batch_valid = valid_mask_bool[b]  # [N]
            # 提取当前批次的有效拐点
            valid_points = points[b, batch_valid,:]#.permute(1, 0)  # [N_valid, 2]
            # 当前批次的边界内坐标
            interior_coords = interior_coords_list[b]  # [N_interior, 2]
            # 计算有效点到边界内所有点的距离
            diff = valid_points.unsqueeze(1) - interior_coords.unsqueeze(0)  # [N_valid, N_interior, 2]
            distances = torch.norm(diff, dim=2)  # [N_valid, N_interior]
            min_distances, min_indices = torch.min(distances, dim=1)
            # print("====",b,diff.shape,distances.shape) # torch.Size([70, 19280])
            # 找到最近邻点
            min_indices = torch.argmin(distances, dim=1)  # [N_valid]
            nearest_points = interior_coords[min_indices]  # [N_valid, 2]
            total_loss += (min_distances.mean())/256
        return total_loss/B

    def differentiable_convex_hull_area(self,points, num_directions=128, alpha=50.0):
        """计算点集的凸包面积（可微近似）"""
        device = points.device
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
        area = 0.5 * torch.abs(torch.sum(x1 * y2 - x2 * y1))

        return area, thetas, boundary_x, boundary_y

    def batch_area_loss(self, points_pred,model_kwargs):
        B, N, _ = points_pred.shape
        valid_mask = model_kwargs['syn_src_key_padding_mask']  # 形状: [B, N]
        room_areas = model_kwargs['syn_room_areas']  # 形状: [B, N, 1]，真实房间面积
        room_indices = th.argmax(model_kwargs['syn_room_indices'], dim=-1)  # 形状: [B, N]，每个点所属房间索引
        room_type = th.argmax(model_kwargs['syn_room_types'], dim=-1)  # 形状: [B, N]，每个点所属房间索引
        valid_mask_bool = (valid_mask == 0)  # 形状: [B, N]，True表示有效点
        room_type_mask = (room_type != 0) & (room_type != 11) & (room_type != 12)
        # print(room_type_mask[0])

        total_loss = 0.0

        for b in range(B):
            area_gt = 0.
            batch_valid = valid_mask_bool[b]  # 形状: [N]
            room_type_mask_bool = room_type_mask[b]
            valid_room_areas = room_areas[b, batch_valid]  # 形状: [N_valid, 1]，真实面积
            valid_room_indices = room_indices[b, batch_valid]  # 形状: [N_valid]

            unique_rooms = th.unique(valid_room_indices)
            for i,room_idx in enumerate(unique_rooms):
                if i >= len(unique_rooms)//2:break
                # 找到当前房间的有效点掩码
                room_mask = (valid_room_indices == room_idx)
                # 提取当前房间的真实面积（取平均值，因为同一房间面积应相同）
                true_area_room_per = valid_room_areas[room_mask][0]
                area_gt += true_area_room_per
                # print(true_area_room_per)


            room_points_pred = points_pred[b, room_type_mask_bool, :]
            # print(room_points_pred)
            area_pred,_,_,_ = self.differentiable_convex_hull_area(room_points_pred)
            total_loss += th.sqrt(F.l1_loss(th.tensor(area_gt), area_pred))  # 需要排掉噪声的干扰
            # print(area_pred,area_gt)
        return total_loss / B


class Projection:
    def __init__(self, eps=1e-6, alpha=0.5, beta=0.0):
        """
        :param eps: small tolerance
        :param alpha: weight for boundary attraction (0-1)
        :param beta: weight for distribution uniformity (0-1)
        """
        self.eps = eps
        self.alpha = alpha  # 边界吸引力权重
        self.beta = beta  # 分布均匀性权重 排斥力
        self.loss_computer = FastBoundaryAreaLoss()

    def apply(self, points, t,model_kwargs, num_iterations=5):
        """
        Improved projection with boundary awareness and distribution uniformity
        """
        points = points.permute([0, 2, 1])
        B, N,_ = points.shape
        boundary_mask = model_kwargs['boundary']
        valid_mask = model_kwargs['syn_src_key_padding_mask']
        room_areas = model_kwargs['syn_room_areas']
        room_indices = th.argmax(model_kwargs['syn_room_indices'],dim=-1) # B,N
        valid_mask_bool = (valid_mask == 0)

        x_dist = points.detach().clone().requires_grad_(True)
        best_loss = float('inf')
        best_x_dist = x_dist.detach().clone()  # 初始化为初始值

        # 动态损失权重
        boundary_weight = (t[0].item() / 1000.0)
        area_weight = (1.0 - t[0].item() / 1000.0)
        initial_lr = 0.01 * (t[0].item() / 1000.0)
        print("t:",t[0],"initial_lr:",initial_lr)

        optimizer = torch.optim.Adam([x_dist], lr=initial_lr)#, weight_decay=0.5) # 3 0.0001  4 0.001
        with th.enable_grad():
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                total_loss = 0.
                coord_diff = x_dist - points  # [B,N,2]，每个元素是 (x_pred-x_gt, y_pred-y_gt)
                offset_distance = torch.norm(coord_diff, dim=-1)  # [B,N]，每个元素是对应点的偏移距离
                mse_loss = offset_distance[valid_mask_bool].mean()  # [K]，K是有效点总数（B*N中的有效个数）
                loss = self.loss_computer.forward(x_dist,model_kwargs)
                total_loss = mse_loss + boundary_weight * loss['boundary_loss'] + area_weight * loss['area_loss']

                total_loss.backward()
                optimizer.step()


        return x_dist.permute([0, 2, 1])

    def __call__(self, x, model_kwargs):
        """Interface for diffusion sampling loop"""
        if 'points' not in model_kwargs:
            raise KeyError("model_kwargs must contain 'points' key")

        points = model_kwargs['points']
        projected_points = self.apply(points, model_kwargs)
        model_kwargs['points'] = projected_points
        return x


