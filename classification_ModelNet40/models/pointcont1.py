'''
Date: 2022-03-12 11:47:58
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 14:05:49
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from einops import rearrange


# MLPBlockFC
class MLPBlockFC(nn.Module):
    def __init__(self, d_points, d_model, p_dropout):
        super(MLPBlockFC, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(d_points, d_model, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Dropout(p=p_dropout))

    def forward(self, x):
        # print('***********************', x.shape)
        return self.mlp(x)


# ResMLPBlock1D
class ResMLPBlock1D(nn.Module):
    def __init__(self, d_points, d_model):
        super(ResMLPBlock1D, self).__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_model),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Conv1d(d_model, d_points, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(d_points))
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.mlp2(self.mlp1(x)) + x)


class MLPBlock1D(nn.Module):
    def __init__(self, d_points, d_model):
        super(MLPBlock1D, self).__init__()
        self.mlp = nn.Sequential(nn.Conv1d(d_points, d_model, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(d_model),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.mlp(x)


class ConT(nn.Module):
    '''
    Content-based Transformer
    Args:
        dim (int): Number of input channels.
        local_size (int): The size of the local feature space.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''

    def __init__(self, dim, local_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 kmeans=False):
        super().__init__()
        self.dim = dim
        self.ls = local_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        Input: [B, S, D]
        Return: [B, S, D]
        '''

        B, S, D = x.shape
        nl = S // self.ls
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1,
                                                                                        4)  # [3, B, h, S, d]

        q_pre = qkv[0].reshape(B * self.num_heads, S, D // self.num_heads).permute(0, 2, 1)  # [B*h, d, S]
        ntimes = int(math.log(nl, 2))
        q_idx_last = torch.arange(S).cuda().unsqueeze(0).expand(B * self.num_heads, S)

        # balanced binary clustering
        for _ in range(ntimes):
            bh, d, n = q_pre.shape  # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.reshape(bh, d, 2, n // 2)  # [B*h*2^n, d, 2, S/2^n]
            q_avg = q_pre_new.mean(dim=-1)  # [B*h*2^n, d, 2]

            q_avg = torch.nn.functional.normalize(q_avg.permute(0, 2, 1), dim=-1)
            q_norm = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1)

            q_scores = square_distance(q_norm, q_avg)  # [B*h*2^n, S/2^n, 2]
            q_ratio = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)  # [B*h*2^n, S/2^n]
            q_idx = q_ratio.argsort()

            q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)  # [B*h*2^n, S/2^n]
            q_idx_new = q_idx.unsqueeze(1).expand(q_pre.size())  # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.gather(dim=-1, index=q_idx_new).reshape(bh, d, 2, n // 2)  # [B*h*2^n, d, 2, S/(2^(n+1))]
            q_pre = rearrange(q_pre_new, 'b d c n -> (b c) d n')  # [B*h*2^(n+1), d, S/(2^(n+1))]

        # clustering is performed independently in each head
        q_idx = q_idx_last.view(B, self.num_heads, S)  # [B, h, S]
        q_idx_rev = q_idx.argsort()  # [B, h, S]

        # cluster query, key, value
        q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())  # [3, B, h, S, d]
        qkv_pre = qkv.gather(dim=-2, index=q_idx)  # [3, B, h, S, d]
        q, k, v = rearrange(qkv_pre, 'qkv b h (nl ls) d -> qkv (b nl) h ls d', ls=self.ls)

        # MSA
        attn = (q - k) * self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out = torch.einsum('bhld, bhld->bhld', attn, v)  # [B*(nl), h, ls, d]

        # merge and reverse
        out = rearrange(out, '(b nl) h ls d -> b h d (nl ls)', h=self.num_heads, b=B)  # [B, h, d, S]
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        res = out.gather(dim=-1, index=q_idx_rev).reshape(B, D, S).permute(0, 2, 1)  # [B, S, D]

        res = self.proj(res)  # [B, S, D]
        res = self.proj_drop(res)

        res = x + res  # [B, S, D]

        return res


# index_points
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


# square_distance
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


# Point2Patch
def Point2Patch(num_patches, patch_size, xyz):
    """
    Patch Partition in 3D Space
    Input:
        num_patches: number of patches, S
        patch_size: number of points per patch, k
        xyz: input points position data, [B, N, 3]
    Return:
        centroid: patch centroid, [B, S, 3]
        knn_idx: [B, S, k]
    """
    # FPS the patch centroid out
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_patches).long()  # [B, S]
    centroid_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
    # knn to group per patch
    dists = square_distance(centroid_xyz, xyz)  # [B, S, N]
    knn_idx = dists.argsort()[:, :, :patch_size]  # [B, S, k]

    return centroid_xyz, fps_idx, knn_idx


class PatchAbstraction(nn.Module):
    def __init__(self, num_patches, patch_size, in_channel, mlp):
        super(PatchAbstraction, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_act = nn.ModuleList()
        self.mlp_res = ResMLPBlock1D(mlp[-1], mlp[-1])

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_act.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, xyz, feature):
        """
        Input: xyz [B, S_, 3]
               features [B, S_, C]
        Return: [B, S, 3+D]
        """
        B, _, C = feature.shape
        centroid_xyz, centroid_idx, knn_idx = Point2Patch(self.num_patches, self.patch_size, xyz)

        centroid_feature = index_points(feature, centroid_idx)  # [B, S, C]
        grouped_feature = index_points(feature, knn_idx)  # [B, S, k, C]

        k = grouped_feature.shape[2]

        # Normalize
        grouped_norm = grouped_feature - centroid_feature.view(B, self.num_patches, 1, C)  # [B, S, k, C]
        groups = torch.cat((centroid_feature.unsqueeze(2).expand(B, self.num_patches, k, C), grouped_norm),
                           dim=-1)  # [B, S, k, 2C]

        groups = groups.permute(0, 3, 2, 1)  # [B, Channel, k, S]
        # print('------', groups.shape)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            act = self.mlp_act[i]
            groups = act(bn(conv(groups)))  # [B, D, k, S]

        max_patches = torch.max(groups, 2)[0]  # [B, D, S]
        max_patches = self.mlp_res(max_patches).transpose(1, 2)  # [B, S, D]

        avg_patches = torch.mean(groups, 2).transpose(1, 2)  # [B, S, D]

        return centroid_xyz, max_patches, avg_patches


class Backbone(nn.Module):
    def __init__(self, patch_dim=[3, 64, 128, 256, 512, 1024], dropout=0.5, num_classes=40, num_points=1024,
                 down_ratio=[2, 4, 8, 16, 32], patch_size=[16, 16, 16, 16, 16], local_size=[16, 16, 16, 16, 16],
                 num_heads=4):
        super().__init__()
        self.nblocks = 4
        self.patch_abstraction = nn.ModuleList()
        self.patch_transformer = nn.ModuleList()
        self.patch_embedding = nn.ModuleList()
        for i in range(self.nblocks):
            self.patch_abstraction.append(PatchAbstraction(int(num_points / down_ratio[i]),
                                                           patch_size[i],
                                                           2 * patch_dim[i],
                                                           [patch_dim[i + 1], patch_dim[i + 1]]))
            self.patch_transformer.append(ConT(patch_dim[i + 1], local_size[i], num_heads))
            self.patch_embedding.append(MLPBlock1D(patch_dim[i + 1] * 2, patch_dim[i + 1]))

    def forward(self, x):
        if x.shape[-1] == 3:
            pos = x
        else:
            pos = x[:, :, :3].contiguous()
        features = x
        # print(pos.shape)
        pos_and_feats = []
        pos_and_feats.append([pos, features])

        for i in range(self.nblocks):
            pos, max_features, avg_features = self.patch_abstraction[i](pos, features)
            avg_features = self.patch_transformer[i](avg_features)
            features = torch.cat([max_features, avg_features], dim=-1)
            features = self.patch_embedding[i](features.transpose(1, 2)).transpose(1, 2)
            pos_and_feats.append([pos, features])

        return features, pos_and_feats


class PointConT_cls(nn.Module):
    def __init__(self, patch_dim=[3, 64, 128, 256, 512, 1024], dropout=0.5, num_classes=40, num_points=1024,
                 down_ratio=[2, 4, 8, 16, 32], patch_size=[16, 16, 16, 16, 16], local_size=[16, 16, 16, 16, 16],
                 num_heads=4):
        super().__init__()
        self.backbone = Backbone(patch_dim=[3, 64, 128, 256, 512, 1024], dropout=0.5, num_classes=40, num_points=1024,
                                 down_ratio=[2, 4, 8, 16, 32], patch_size=[16, 16, 16, 16, 16],
                                 local_size=[16, 16, 16, 16, 16], num_heads=4)
        self.mlp1 = MLPBlockFC(512, 512, dropout)
        self.mlp2 = MLPBlockFC(512, 256, dropout)
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        # print(x.shape)
        patches, _ = self.backbone(x)  # [B, num_patches[-1], patch_dim[-1]]
        res = torch.max(patches, dim=1)[0]  # [B, patch_dim[-1]]
        res = self.mlp2(self.mlp1(res))
        res = self.output_layer(res)

        return res


if __name__ == '__main__':
    data = torch.rand(20, 1024, 3).cuda()
    model = PointConT_cls().cuda()
    out = model(data)
    print(out.shape)
