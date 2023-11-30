import math
from torch import nn
from einops import pack, rearrange, repeat
# from mmengine.registry import MODELS
# from mmengine.model import BaseModule
# from apes.models.utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample
# from .utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample


# ops
import torch


def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def select_neighbors(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors


def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()

# layers
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.K = 32
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))

    def forward(self, x):
        x_list = []
        x = group(x, self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
        return x


class N2PAttention(nn.Module):
    def __init__(self):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = 32
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x


class GlobalDownSample(nn.Module):
    def __init__(self, npts_ds):
        super(GlobalDownSample, self).__init__()
        self.npts_ds = npts_ds
        self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q_conv(x)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(x)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(x)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        selection = torch.sum(attention, dim=-2)  # (B, N, N) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M N', N=attention.shape[-1]))  # (B, N, N) -> (B, M, N)
        v = scores @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, M, N) @ (B, N, C) -> (B, M, C)
        out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
        return out


class LocalDownSample(nn.Module):
    def __init__(self, npts_ds):
        super(LocalDownSample, self).__init__()
        self.npts_ds = npts_ds  # number of downsampled points
        self.K = 32  # number of neighbors
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = rearrange(q, 'B C N 1 -> B N 1 C').contiguous()  # (B, C, N, 1) -> (B, N, 1, C)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = rearrange(k, 'B C N K -> B N C K').contiguous()  # (B, C, N, K) -> (B, N, C, K)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = rearrange(v, 'B C N K -> B N K C').contiguous()  # (B, C, N, K) -> (B, N, K, C)
        energy = q @ k  # (B, N, 1, C) @ (B, N, C, K) -> (B, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, N, 1, K) -> (B, N, 1, K)
        selection = rearrange(torch.std(attention, dim=-1, unbiased=False), 'B N 1 -> B N').contiguous()  # (B, N, 1, K) -> (B, N, 1) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M 1 K', K=attention.shape[-1]))  # (B, N, 1, K) -> (B, M, 1, K)
        v = torch.gather(v, dim=1, index=repeat(self.idx, 'B M -> B M K C', K=v.shape[-2], C=v.shape[-1]))  # (B, N, K, C) -> (B, M, K, C)
        out = rearrange(scores@v, 'B M 1 C -> B C M').contiguous()  # (B, M, 1, K) @ (B, M, K, C) -> (B, M, 1, C) -> (B, C, M)
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.skip_link = nn.Conv1d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pcd_up, pcd_down):
        q = self.q_conv(pcd_up)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        v = self.v_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, M) -> (B, N, M)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, M) -> (B, N, M)
        x = attention @ rearrange(v, 'B C M -> B M C').contiguous()  # (B, N, M) @ (B, M, C) -> (B, N, C)
        x = rearrange(x, 'B N C -> B C N').contiguous()  # (B, N, C) -> (B, C, N)
        x = self.skip_link(pcd_up) + x  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x


# @MODELS.register_module()
class APESClsBackbone(nn.Module):
    def __init__(self, which_ds):
        super(APESClsBackbone, self).__init__()
        self.embedding = Embedding()
        if which_ds == 'global':
            self.ds1 = GlobalDownSample(1024)  # 2048 pts -> 1024 pts
            self.ds2 = GlobalDownSample(512)  # 1024 pts -> 512 pts
        elif which_ds == 'local':
            self.ds1 = LocalDownSample(1024)  # 2048 pts -> 1024 pts
            self.ds2 = LocalDownSample(512)  # 1024 pts -> 512 pts
        else:
            raise NotImplementedError
        self.n2p_attention = N2PAttention()
        self.n2p_attention1 = N2PAttention()
        self.n2p_attention2 = N2PAttention()
        self.conv = nn.Conv1d(128, 1024, 1)
        self.conv1 = nn.Conv1d(128, 1024, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)

    def forward(self, x):
        self.res_link_list = []
        x = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        x = self.n2p_attention(x)  # (B, 128, 2048) -> (B, 128, 2048)
        self.res_link_list.append(self.conv(x).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        x = self.ds1(x)  # (B, 128, 2048) -> (B, 128, 1024)
        x = self.n2p_attention1(x)  # (B, 128, 1024) -> (B, 128, 1024)
        self.res_link_list.append(self.conv1(x).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 1024, 1024) -> (B, 1024)
        x = self.ds2(x)  # (B, 128, 1024) -> (B, 128, 512)
        x = self.n2p_attention2(x)  # (B, 128, 512) -> (B, 128, 512)
        self.res_link_list.append(self.conv2(x).max(dim=-1)[0])  # (B, 128, 512) -> (B, 1024, 512) -> (B, 1024)
        x, ps = pack(self.res_link_list, 'B *')  # (B, 3072)
        return x
class APESClsHead(nn.Module):
    def __init__(self):
        super(APESClsHead, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(3072, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.linear2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.linear3 = nn.Linear(256, 40)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)  # (B, 3072) -> (B, 1024)
        x = self.dp1(x)  # (B, 1024) -> (B, 1024)
        x = self.linear2(x)  # (B, 1024) -> (B, 256)
        x = self.dp2(x)  # (B, 256) -> (B, 256)
        x = self.linear3(x)  # (B, 256) -> (B, 40)
        return x

class APES(nn.Module):
    def __init__(self):
        super(APES, self).__init__()
        self.backbone = APESClsBackbone(which_ds='global')
        self.head = APESClsHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    data = torch.rand(2, 3, 1024).cuda()
    # model1 = APESClsBackbone(which_ds='global').cuda()
    # model2 = APESClsHead().cuda()
    # out1 = model1(data)
    # out2 = model2(out1)
    # print(out2.shape)
    model = APES().cuda()
    out = model(data)
    print(out.shape)

