import random

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from nn import timestep_embedding
from torchvision.models import efficientnet_b1
import torch.nn.functional as F

def dec2bin(xinp, bits):
    mask = 2 ** th.arange(bits - 1, -1, -1).to(xinp.device, xinp.dtype)
    return xinp.unsqueeze(-1).bitwise_and(mask).ne(0).float()


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_channels):
        super().__init__()
        self.model_channels = model_channels
        self.effnet = efficientnet_b1(pretrained=True)
        # 移除最后两层
        self.features = nn.Sequential(*list(self.effnet.children())[:-2])
        self.conv1d = nn.Conv1d(in_channels=1280, out_channels=self.model_channels, kernel_size=1)

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1280, 1280 // 8, 1),
            nn.ReLU(),
            nn.Conv2d(1280 // 8, 1, 1),
            nn.Sigmoid()
        ) # [B,1,8,8]

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, 1280 // 8, 1),
            nn.ReLU(),
            nn.Conv2d(1280 // 8, 1280, 1),
            nn.Sigmoid()
        ) # [B, 1280, 1, 1]

    def forward(self, x):  # x: 边界 tensor [batch, 3, H, W]
        x = self.features(x)#.flatten(2)  # 输出: [batch, 1280, 8, 8]
        x_o = x

        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        # 应用注意力
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        x = x + x_o

        x = x.flatten(2)
        x = self.conv1d(x)
        return x.permute([0, 2, 1]).float()  # 形状: [batch, 64, 512]


class Block(nn.Module):
    def __init__(self, model_channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(model_channels)  # nn.LayerNorm(model_channels)  #
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, model_channels, obj_emb=128):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.input_emb = nn.Linear(self.in_channels, self.model_channels)  # 输入嵌入
        self.time_emb = nn.Linear(self.model_channels, self.model_channels)
        self.obj_emb = nn.Linear(obj_emb, self.model_channels)
        self.block1 = Block(self.model_channels)
        self.block2 = Block(self.model_channels)
        self.block3 = Block(self.model_channels)

        # 添加门控融合
        self.fusion_gate = nn.Linear(2 * model_channels, model_channels)
        # 添加更多归一化层
        self.norm = nn.InstanceNorm1d(model_channels)  # nn.LayerNorm(model_channels)  # 改为LayerNorm

    def forward(self, x, t, obj_features=None, room_index=None):  # x: $X^t$ 坐标 [batch, num_points, 2], t: 时间步 [batch]
        x = self.input_emb(x)
        out = self.block1(x)
        time_emb = self.time_emb(timestep_embedding(t, self.model_channels))
        time_emb = time_emb.unsqueeze(1)

        # 门控融合
        # fusion_input = torch.cat([out, time_emb.expand_as(out)], dim=-1)
        # gate = torch.sigmoid(self.fusion_gate(fusion_input))
        # fused = gate * out + (1 - gate) * time_emb
        if obj_features is not None:
            # obj_emb = self.obj_emb(obj_features)
            # out = out + obj_emb
            # out = self.block3(out)
            pass

        out = out + time_emb
        out = self.block2(out)

        out = out + x
        return out

class Area_emb1(nn.Module):
    def __init__(self, model_channels):
        super().__init__()
        self.model_channels = model_channels
        self.cross_attention_area = MultiHeadAttention(4, self.model_channels)
        self.cross_attention_area1 = MultiHeadAttention(4, self.model_channels)
        self.self_attention_area = MultiHeadAttention(4, self.model_channels)
        self.encode1 = nn.Linear(26, model_channels//2)
        self.encode2 = nn.Linear(model_channels//2, model_channels)
        self.decode1 = nn.Linear(model_channels, model_channels)
        self.decode2 = nn.Linear(model_channels, model_channels)
        self.dropout = nn.Dropout(0.0)
        self.activation = nn.SiLU()
        self.norm_1 = nn.InstanceNorm1d(self.model_channels)
        self.norm_2 = nn.InstanceNorm1d(self.model_channels)
        self.ff = FeedForward(self.model_channels, self.model_channels * 2, 0.2, self.activation)


    def forward(self, x, area, room_indices, room_types,timesteps):
        # TODO 提取房间级面积特征
        area = area.float()  # [B, N, A]  （A=1）
        room_index = room_indices.float()  # [B, N, M]  （R=32） 房间索引里面没有1
        room_types = room_types.float() # [B,N,H] H 25
        _, _, A = area.shape
        _, _, H = room_types.shape

        # 转置索引维度以便按房间聚合
        room_idx_t = room_index.transpose(1, 2)  # [B, M, N]
        # 创建掩码标记有效顶点
        mask = room_idx_t > 0  # [B,M,N] 输出每个房间对应的拐点编码
        mask_float = mask.float()  # [B, M, N]
        # 为全0行（无顶点的房间）设置一个特殊值，避免argmax返回0
        # 这里给全0行的第一个位置设置-1，其他位置保持0
        row_sums = mask_float.sum(dim=-1, keepdim=True)  # [B, M, 1]
        is_valid = row_sums > 0  # 当前房间是否有拐点

        # 现在argmax会为无顶点的房间返回0，但我们可以通过row_sums判断
        first_vert_idx = mask_float.argmax(dim=-1).long()  # [B, M]
        first_vert_idx = th.where(is_valid.squeeze(-1), first_vert_idx,
                                  th.tensor(-1, device=first_vert_idx.device,
                                            dtype=first_vert_idx.dtype))  # 如果第一个拐点为-1说明这个房间的面积为0

        # 提取房间特征
        gather_idx = first_vert_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, A)  # 用0占位无效索引
        gather_idx_types = first_vert_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, H)  # 用0占位无效索引
        room_area = area.gather(dim=1, index=gather_idx)  # [B, M, A]
        room_type = room_types.gather(dim=1, index=gather_idx_types) # [B, M, H]
        room_features = th.cat([room_type,room_area], dim=-1)
        # print(room_features.shape)
        room_features = room_features * is_valid.float()  # 无效房间特征置0 # [B, M, A+H]
        # print(room_features[ide].argmax(dim=-1))
        # TODO 对房间级面积特征进行信息传递
        room_area_mask = is_valid.squeeze(-1)
        room_area_mask = ~(room_area_mask.unsqueeze(1) & room_area_mask.unsqueeze(2))
        room_features_emb = self.activation(self.norm_1(self.encode1(room_features)))
        room_features_emb = self.encode2(room_features_emb)

        room_features_att = self.dropout(self.self_attention_area(room_features_emb,room_features_emb,room_features_emb))
        room_features_emb = room_features_emb + room_features_att
        room_features_emb = self.activation(self.norm_2(self.decode1(room_features_emb)))
        # room_features_emb = self.norm_2(room_features_emb)
        room_features_emb = self.decode2(self.dropout(room_features_emb))  # [B, M, D]


        # TODO 将面积特征分配给每个拐点时的掩码
        B, M, _ = room_features.shape  # [B, M, A]
        _, N, _ = room_index.shape
        room_ids = th.argmax(room_index, dim=-1)  # B,N,每个拐点属于哪个房间 -1是为了让其从0开始
        room_range = (th.arange(0, M, device=room_features.device)).view(1, 1, M).expand(B, N, M)  # [ 1 2 3 ... M]
        # 扩展room_ids以匹配形状 [B, N, M]
        room_ids_expanded = room_ids.unsqueeze(-1).expand(B, N, M)
        # 创建掩码：当房间索引不等于拐点所属房间时，掩码为True（需要被掩码）
        vert_is_valid = (room_index.sum(dim=-1, keepdim=True) > 0).float()  # [B, N, 1]
        key_pad_mask = (room_range != room_ids_expanded) | (vert_is_valid.expand(B, N, M) == 0)  # [B, N, M]

        key_pad_mask_all = ~(is_valid.squeeze(-1).unsqueeze(1).expand(B, N, M))

        area_emb = room_features_emb
        area_cross_att = self.cross_attention_area(x, area_emb, area_emb)
                          # + self.cross_attention_area1(x, area_emb, area_emb, mask=key_pad_mask_all

        # area_cross_att = self.norm_2(area_cross_att)
        # area_cross_att = self.dropout(self.ff(area_cross_att))
        # area_cross_att = self.norm_2(area_cross_att)

        return area_cross_att


class Area_emb(nn.Module):
    def __init__(self, model_channels):
        super().__init__()
        self.model_channels = model_channels
        self.cross_attention_area = MultiHeadAttention(4, self.model_channels)

        self.value_projection = nn.Linear(1+25, model_channels)  # 标量值投影
        self.room_embedding = nn.Embedding(32, model_channels)  # 房间位置编码
        self.layer_norm = nn.LayerNorm(model_channels)
        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_channels, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x, area, room_indices, room_types,timesteps):
        # TODO 提取房间级面积特征
        area = area.float()  # [B, N, A]  （A=1）
        room_index = room_indices.long()  # [B, N, M]  （R=32） 房间索引里面没有1
        room_types = room_types.float() # [B,N,H] H 25
        _, _, A = area.shape
        _, _, H = room_types.shape
        _, _, K = room_types.shape

        # 转置索引维度以便按房间聚合
        room_idx_t = room_index.transpose(1, 2)  # [B, M, N]
        # 创建掩码标记有效顶点
        mask = room_idx_t > 0  # [B,M,N] 输出每个房间对应的拐点编码
        mask_float = mask.float()  # [B, M, N]
        # 为全0行（无顶点的房间）设置一个特殊值，避免argmax返回0
        # 这里给全0行的第一个位置设置-1，其他位置保持0
        row_sums = mask_float.sum(dim=-1, keepdim=True)  # [B, M, 1]
        is_valid = row_sums > 0  # 当前房间是否有拐点

        # 现在argmax会为无顶点的房间返回0，但我们可以通过row_sums判断
        first_vert_idx = mask_float.argmax(dim=-1).long()  # [B, M]
        first_vert_idx = th.where(is_valid.squeeze(-1), first_vert_idx,
                                  th.tensor(-1, device=first_vert_idx.device,
                                            dtype=first_vert_idx.dtype))  # 如果第一个拐点为-1说明这个房间的面积为0

        # 提取房间特征
        gather_area = first_vert_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, A)  # 用0占位无效索引
        gather_types = first_vert_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, H)  # 用0占位无效索引
        gather_idx = first_vert_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, K)  # 用0占位无效索引
        room_area = area.gather(dim=1, index=gather_area)  # [B, M, A]
        room_type = room_types.gather(dim=1, index=gather_types) # [B, M, H]
        room_ids = room_index.gather(dim=1, index=gather_idx).argmax(dim=-1) # [B, M, K]
        room_features = th.cat([room_type,room_area], dim=-1) # [B,M,A + H]
        # print(room_features.shape)B,M,H
        room_features = room_features * is_valid.float()  # 无效房间特征置0 # [B, M, A+H]
        # print(room_features.shape)

        # 数值特征投影
        value_emb = self.value_projection(room_features)  # [B, M, d_model]
        # 房间位置编码
        room_emb = self.room_embedding(room_ids)  # [B, M, d_model]
        # 特征融合
        combined_emb = value_emb + room_emb
        combined_emb = self.layer_norm(combined_emb)
        encoded = self.transformer_encoder(combined_emb)

        area_cross_att = self.cross_attention_area(x, encoded, encoded)

        return area_cross_att

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0:1, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = th.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 1, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = th.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.door_attn = MultiHeadAttention(heads, d_model)
        self.gen_attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_model * 2, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, door_mask, self_mask, gen_mask):
        assert (gen_mask.max() == 1 and gen_mask.min() == 0), f"{gen_mask.max()}, {gen_mask.min()}"
        x2 = self.norm_1(x)
        x = x + self.dropout(self.door_attn(x2, x2, x2, door_mask)) \
            + self.dropout(self.self_attn(x2, x2, x2, self_mask)) \
            + self.dropout(self.gen_attn(x2, x2, x2, gen_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class TransformerModel(nn.Module):
    """
    The full Transformer model with timestep embedding.
    """

    def __init__(
            self,
            in_channels,
            condition_channels,
            model_channels,
            out_channels,
            dataset,
            use_checkpoint,
            use_unet,
            analog_bit,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.time_channels = model_channels
        self.use_checkpoint = use_checkpoint
        self.analog_bit = analog_bit
        self.use_unet = use_unet
        self.num_layers = 4

        # self.pos_encoder = PositionalEncoding(model_channels, 0.001)
        # self.activation = nn.SiLU()
        self.activation = nn.ReLU()

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.time_channels),
        )
        self.input_emb = nn.Linear(self.in_channels, self.model_channels)
        self.condition_emb = nn.Linear(self.condition_channels, self.model_channels)

        if use_unet:
            self.unet = UNet(self.model_channels, 1)

        self.transformer_layers = nn.ModuleList(
            [EncoderLayer(self.model_channels, 4, 0.1, self.activation) for x in range(self.num_layers)])
        # self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(self.model_channels, 4, self.model_channels*2, 0.1, self.activation, batch_first=True) for x in range(self.num_layers)])

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.model_channels // 2)
        self.output_linear3 = nn.Linear(self.model_channels // 2, self.out_channels)

        if not self.analog_bit:
            self.output_linear_bin1 = nn.Linear(162 + self.model_channels, self.model_channels)
            self.output_linear_bin2 = EncoderLayer(self.model_channels, 1, 0.1, self.activation)
            self.output_linear_bin3 = EncoderLayer(self.model_channels, 1, 0.1, self.activation)
            self.output_linear_bin4 = nn.Linear(self.model_channels, 16)

        print(f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

        # TODO boundary
        self.effNet = EfficientNetFeatureExtractor(self.model_channels)
        # self.resBlock = ResBlock(self.in_channels, self.model_channels, 30)
        self.cross_attention = MultiHeadAttention(4, self.model_channels)
        self.input_linear = nn.Linear(self.model_channels, self.model_channels)
        self.input_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.resBlock = ResBlock(self.in_channels, self.model_channels, 32)
        # self.area_emb = nn.Linear(30, self.model_channels)
        self.cross_attention_area = MultiHeadAttention(4, self.model_channels)
        self.block = Block(self.model_channels)

        # self.area_emb = nn.Sequential(
        #     nn.Linear(1, self.model_channels//2),
        #     nn.ReLU(),
        #     nn.Linear(self.model_channels//2, self.model_channels)
        # )

        self.area_emb = Area_emb(self.model_channels)
        self.area_cross_emb = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.LayerNorm(self.model_channels),
            nn.SiLU()
        )
        # self.tabnet = RoomFeatureTabNet(self.model_channels,self.model_channels,self.model_channels)
        self.alpha = nn.Parameter(th.tensor(1.0))

    def expand_points(self, points, connections):
        def average_points(point1, point2):
            points_new = (point1 + point2) / 2
            return points_new

        p1 = points
        p1 = p1.view([p1.shape[0], p1.shape[1], 2, -1])
        p5 = points[th.arange(points.shape[0])[:, None], connections[:, :, 1].long()]
        p5 = p5.view([p5.shape[0], p5.shape[1], 2, -1])
        p3 = average_points(p1, p5)
        p2 = average_points(p1, p3)
        p4 = average_points(p3, p5)
        p1_5 = average_points(p1, p2)
        p2_5 = average_points(p2, p3)
        p3_5 = average_points(p3, p4)
        p4_5 = average_points(p4, p5)
        points_new = th.cat((p1.view_as(points), p1_5.view_as(points), p2.view_as(points),
                             p2_5.view_as(points), p3.view_as(points), p3_5.view_as(points), p4.view_as(points),
                             p4_5.view_as(points), p5.view_as(points)), 2)
        return points_new.detach()

    def create_image(self, points, connections, room_indices, img_size=256, res=200):
        img = th.zeros((points.shape[0], 1, img_size, img_size), device=points.device)
        points = (points + 1) * (img_size // 2)
        points[points >= img_size] = img_size - 1
        points[points < 0] = 0
        p1 = points
        p2 = points[th.arange(points.shape[0])[:, None], connections[:, :, 1].long()]

        slope = (p2[:, :, 1] - p1[:, :, 1]) / ((p2[:, :, 0] - p1[:, :, 0]))
        slope[slope.isnan()] = 0
        slope[slope.isinf()] = 1

        m = th.linspace(0, 1, res, device=points.device)
        new_shape = [p2.shape[0], res, p2.shape[1], p2.shape[2]]

        new_p2 = p2.unsqueeze(1).expand(new_shape)
        new_p1 = p1.unsqueeze(1).expand(new_shape)
        new_room_indices = room_indices.unsqueeze(1).expand([p2.shape[0], res, p2.shape[1], 1])

        inc = new_p2 - new_p1

        xs = m.view(1, -1, 1) * inc[:, :, :, 0]
        xs = xs + new_p1[:, :, :, 0]
        xs = xs.long()

        x_inc = th.where(inc[:, :, :, 0] == 0, inc[:, :, :, 1], inc[:, :, :, 0])
        x_inc = m.view(1, -1, 1) * x_inc
        ys = x_inc * slope.unsqueeze(1) + new_p1[:, :, :, 1]
        ys = ys.long()

        img[th.arange(xs.shape[0])[:, None], :, xs.view(img.shape[0], -1),
        ys.view(img.shape[0], -1)] = new_room_indices.reshape(img.shape[0], -1, 1).float()
        return img.detach()

    def get_tanh_time_weight(self,timesteps, T_max, center=0.5, scale=5.0):
        """
        使用 tanh 曲线为扩散模型的时间步生成加权系数。

        Args:
            timesteps (torch.Tensor): 当前的时间步 (例如，从 T_max 递减到 1)。
            T_max (float): 最大的时间步。
            center (float): 归一化时间步的中心点，用于调整 tanh 曲线的“拐点”。
                            例如，如果设置为 0.5，则当 (T_max - t)/(T_max - 1) = 0.5 时，权重为 0.5。
            scale (float): 控制 tanh 曲线的陡峭程度。值越大，权重变化越快。

        Returns:
            torch.Tensor: 对应每个时间步的权重，范围在 (0, 1)。
        """
        # 确保 T_max 是浮点数
        T_max = float(T_max)

        # 归一化并反转 timesteps，使得 timesteps=1 对应 1，timesteps=T_max 对应 0
        # 防止 T_max - 1.0 为 0
        normalized_t_reversed = (T_max - timesteps.float()) / (T_max - 1.0)

        # 映射到 tanh 的输入范围
        input_for_tanh = (normalized_t_reversed - center) * scale

        # 应用 tanh 并缩放到 (0, 1)
        tanh_output = th.tanh(input_for_tanh)
        time_weight = (tanh_output + 1.0) / 2.0

        return time_weight


    def forward(self, x, timesteps, xtalpha, epsalpha, is_syn=False, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x S x C] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x S x C] Tensor of outputs.
        """
        # prefix = 'syn_' if is_syn else ''
        prefix = 'syn_' if is_syn else ''
        x = x.permute([0, 2, 1]).float()  # -> convert [N x C x S] to [N x S x C]

        if not self.analog_bit:
            x = self.expand_points(x, kwargs[f'{prefix}connections'])
        # Different input embeddings (Input, Time, Conditions)
        # time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # time_emb = time_emb.unsqueeze(1)
        input_emb = self.input_emb(x)
        if self.condition_channels > 0:
            cond = None
            for key in [f'{prefix}corner_indices', f'{prefix}room_types', f'{prefix}room_indices']: # ,f'{prefix}room_areas'
                if cond is None:
                    cond = kwargs[key]
                else:
                    # tmp = kwargs[key]
                    # if key in [f'{prefix}room_types', f'{prefix}corner_indices', f'{prefix}room_indices']:
                    #     tmp = kwargs[key].argmax(dim=-1,keepdim=True)
                    cond = th.cat((cond, kwargs[key]), 2)
            cond_emb = self.condition_emb(cond.float())
            # room_embedding = self.tabnet(cond.float())
        # cond_emb = self.cross_attention(input_emb,room_embedding,room_embedding)
        obj_features, room_index = None, None
        input_emb_x = self.resBlock(x, timesteps, obj_features, room_index)

        # TODO 边界特征
        boundary_emb = self.effNet(kwargs['boundary'])  # [B,64,512]
        input_linear = self.input_linear(input_emb_x)
        boundary_cross_att = self.cross_attention(input_emb_x, boundary_emb, boundary_emb)  # [B,100,2]
        # TODO end 边界特征

        # TODO 面积特征
        # area_emb = self.area_emb(input_emb_x,kwargs[f'{prefix}room_areas'],kwargs[f'{prefix}room_indices'],kwargs[f'{prefix}room_types'],timesteps)

        # time_mask = (timesteps / 1000).unsqueeze(-1).unsqueeze(-1)
        input_emb = input_linear + boundary_cross_att #+ self.alpha * area_emb


        # PositionalEncoding and DM model
        out = input_emb + cond_emb  #+ time_emb.repeat((1, input_emb.shape[1], 1))
        for layer in self.transformer_layers:
            out = layer(out, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'], kwargs[f'{prefix}gen_mask'])

        out_dec = self.output_linear1(out)
        out_dec = self.activation(out_dec)
        out_dec = self.output_linear2(out_dec)
        out_dec = self.output_linear3(out_dec)

        if not self.analog_bit:
            out_bin_start = x * xtalpha.repeat([1, 1, 9]) - out_dec.repeat([1, 1, 9]) * epsalpha.repeat([1, 1, 9])
            out_bin = (out_bin_start / 2 + 0.5)  # -> [0,1]
            out_bin = out_bin * 256  # -> [0, 256]
            out_bin = dec2bin(out_bin.round().int(), 8)
            out_bin_inp = out_bin.reshape([x.shape[0], x.shape[1], 16 * 9])
            out_bin_inp[out_bin_inp == 0] = -1

            out_bin = th.cat((out_bin_start, out_bin_inp, cond_emb), 2)
            out_bin = self.activation(self.output_linear_bin1(out_bin))
            out_bin = self.output_linear_bin2(out_bin, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'],
                                              kwargs[f'{prefix}gen_mask'])
            out_bin = self.output_linear_bin3(out_bin, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'],
                                              kwargs[f'{prefix}gen_mask'])
            out_bin = self.output_linear_bin4(out_bin)

            out_bin = out_bin.permute([0, 2, 1])  # -> convert back [N x S x C] to [N x C x S]

        out_dec = out_dec.permute([0, 2, 1])  # -> convert back [N x S x C] to [N x C x S]

        if not self.analog_bit:
            return out_dec, out_bin
        else:
            return out_dec, None
