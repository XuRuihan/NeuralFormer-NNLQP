import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter

import wandb

from .utils import to_2tuple

# from torch_geometric.utils import to_dense_adj


def gen_Khop_adj(edge_index, n_tokens, k=1):
    value = torch.ones(edge_index.size(1)).to(
        edge_index.device
    )  # edge_index(2, num_edges)
    temp = torch.sparse_coo_tensor(edge_index, value, size=(n_tokens, n_tokens))
    matrix = temp.to_dense()

    if k == 1:
        return matrix


def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return
    if init_type == "thomas":
        size = tensor.size(-1)
        stdv = 1.0 / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == "kaiming_normal_in":
        nn.init.kaiming_normal_(tensor, mode="fan_in", nonlinearity=nonlinearity)
    elif init_type == "kaiming_normal_out":
        nn.init.kaiming_normal_(tensor, mode="fan_out", nonlinearity=nonlinearity)
    elif init_type == "kaiming_uniform_in":
        nn.init.kaiming_uniform_(tensor, mode="fan_in", nonlinearity=nonlinearity)
    elif init_type == "kaiming_uniform_out":
        nn.init.kaiming_uniform_(tensor, mode="fan_out", nonlinearity=nonlinearity)
    elif init_type == "orthogonal":
        nn.init.orthogonal_(tensor, gain=nn.init.calculate_gain(nonlinearity))
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class GNN_LinearAttn(nn.Module):
    def __init__(self, dim: int, degree: bool, dropout: int = 0.0):
        super().__init__()
        self.degree = degree

        self.lin_qk = nn.Linear(dim, dim)
        self.lin_l = nn.Linear(dim, dim)
        self.lin_r = nn.Linear(dim, dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if degree:
            self.lin_d = nn.Linear(1, dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.degree:
            self.lin_d.reset_parameters()

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        if self.degree:
            degree = adj.sum(dim=-1, keepdim=True)  # (B, N, 1)
            degree = torch.sigmoid(self.lin_d(degree))
            x = x * degree

        QK = torch.sigmoid(self.lin_qk(x))
        scores = torch.matmul(QK, QK.transpose(-2, -1)) / math.sqrt(x.size(-1))
        scores = scores * adj

        attn = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.matmul(attn, x)
        out = self.lin_l(out)

        # Adj Feat + Root Feat
        out = out + self.lin_r(x)

        return self.dropout(self.relu(out))


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        degree: bool,
        n_head: int,
        dropout: float = 0.0,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.degree = degree
        self.n_head = n_head
        self.head_size = dim // self.n_head  # default: 32

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.ReLU()  # nn.Linear(dim, dim)
        self.attn_dropout = nn.Identity()  # nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = rel_pos_bias
        if degree:
            self.lin_d = nn.Linear(1, dim)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        if self.degree:
            degree = adj.sum(dim=-1, keepdim=True)  # (B, N, 1)
            degree = torch.sigmoid(self.lin_d(degree))
            x = x * degree

        B, L, C = x.shape

        qk, res, value = self.qkv(x).chunk(3, -1)
        qk = qk.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        res = res.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        qk = torch.sigmoid(qk)
        # (b, n_head, l_q, d_per_head) * (b, n_head, d_per_head, l_k)
        attn = torch.matmul(qk, qk.mT) / math.sqrt(self.head_size)

        if self.rel_pos_bias:
            # attn_mask = adj + torch.eye(
            #     adj.shape[-1], dtype=adj.dtype, device=adj.device
            # )
            attn_mask = adj
            # attn = attn * (1 + attn_mask)
            # attn = attn * attn_mask
            attn = attn.masked_fill(attn_mask == 0, 0)

        # attn = F.softmax(attn, dim=-1)
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-6)
        # attn = attn + torch.eye(attn.shape[-1], dtype=attn.dtype, device=attn.device)
        # attn = F.relu(attn)
        # attn = attn / (attn.sum(-1, True) + 1e-6)
        attn = self.attn_dropout(attn)  # (b, n_head, l_q, l_k)
        x = torch.matmul(attn, value) + res

        x = x.transpose(1, 2).contiguous().view(B, L, self.dim)
        return self.resid_dropout(self.proj(x))

    def extra_repr(self) -> str:
        return f"n_head={self.n_head}, degree={self.degree}"


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        degree: bool = False,
        n_head: int = 8,
        dropout: float = 0.0,
        droppath: float = 0.0,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.Identity()  # nn.LayerNorm(dim)
        # The larger the dataset, the better rel_pos_bias works
        # probably due to the overfitting of rel_pos_bias
        self.attn = MultiHeadAttention(
            dim,
            degree,
            n_head,
            dropout,
            rel_pos_bias=rel_pos_bias,
        )
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()
        self.layer_scale = Scale(dim, 1e-4)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.attn(x_, adj)
        return self.layer_scale(self.drop_path(x_)) + x


class GCNMlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gcn = nn.Linear(in_features, hidden_features)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif act_layer.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        x1 = self.fc1(x)
        gcn_x1, gcn_x2 = self.gcn(x).chunk(2, dim=-1)
        x = x1 + torch.cat([adj @ gcn_x1, adj.mT @ gcn_x2], dim=-1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        act_layer: str = "relu",
        dropout: float = 0.0,
        droppath: float = 0.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.feed_forward = GCNMlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()
        self.layer_scale = Scale(dim, 1e-4)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.feed_forward(x_, adj)
        return self.layer_scale(self.drop_path(x_)) + x


class PreReduction(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.norm(x) * 0.01
        x = self.net(x) + x
        return x


# reduce_func: "sum", "mul", "mean", "min", "max"
class Net(nn.Module):
    def __init__(
        self,
        dataset,
        feat_shuffle,
        glt_norm,
        n_attned_gnn=2,
        num_node_features=44,
        gnn_hidden=512,
        fc_hidden=512,
        use_degree=False,
        reduce_func="sum",
        norm_sf=False,
        ffn_ratio=4,
        init_value=1e-4,
        real_test=False,
        dropout=0.05,
        rel_pos_bias=True,
    ):
        super().__init__()
        self.dataset = dataset
        self.real_test = real_test
        self.reduce_func = reduce_func
        self.norm_sf = norm_sf
        self.n_attned_gnn = n_attned_gnn

        self.embedding = nn.Linear(num_node_features, gnn_hidden)

        self.gnn_layers = nn.ModuleList()
        # self.gnn_drops = nn.ModuleList()
        # self.gnn_relus = nn.ModuleList()
        self.FFN_layers = nn.ModuleList()

        for j in range(n_attned_gnn):
            self.gnn_layers.append(
                # SelfAttentionBlock(
                #     gnn_hidden, use_degree, dropout=dropout, rel_pos_bias=True
                # )
                GNN_LinearAttn(gnn_hidden, use_degree, dropout)
            )
            # self.gnn_drops.append(nn.Dropout(p=dropout))
            # self.gnn_relus.append(nn.ReLU())

            self.FFN_layers.append(
                FeedForwardBlock(
                    gnn_hidden,
                    ffn_ratio,
                    dropout=dropout,
                )
            )

        if self.dataset == "nasbench101" or self.dataset == "nasbench201":
            total_dim = n_attned_gnn * gnn_hidden
            self.dim_out = gnn_hidden
            self.fusion_mlp = nn.Sequential(
                nn.Linear(total_dim, total_dim // 4),
                nn.ReLU(),
                nn.Linear(total_dim // 4, n_attned_gnn),
                nn.Sigmoid(),
            )

        out_dim = 1
        if self.norm_sf:
            self.norm_sf_linear = nn.Linear(40, gnn_hidden)
            self.norm_sf_drop = nn.Dropout(p=dropout)
            self.norm_sf_relu = nn.ReLU()
            sf_hidden = gnn_hidden
        else:
            sf_hidden = 4 if dataset == "nnlqp" else 0
        self.pre_reduction = PreReduction(
            gnn_hidden, gnn_hidden * 2, gnn_hidden, dropout
        )

        self.predictor = nn.Sequential(
            nn.Linear(gnn_hidden + sf_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, out_dim),
            nn.Softplus(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data1, data2, n_edges):
        gnn_feat = []

        if self.dataset == "nnlqp":
            data, static_feature = data1, data2

            # feat (B*N, D); N=21, 124, ...
            # adj (2, B*n_edges)
            # n_edges (B): (n_edges1, n_edges2, ....)
            feat, adj = data.x, data.edge_index

            idx = torch.ones(data.batch.size(0)).to(feat.device)
            idx = scatter(
                idx, data.batch, dim=0, reduce="sum"
            )  # (N1, N2, ..., Ni...) length of each sample in batch

            for i in range(idx.size(0)):
                if i == idx.size(0) - 1:
                    n_nodes_i = int(sum(idx[:i]))
                    n_edges_i = int(sum(n_edges[:i]))
                    x_i = feat[n_nodes_i:].unsqueeze(0)
                    adj_i = gen_Khop_adj(
                        adj[:, n_edges_i:] - n_nodes_i, int(idx[i])
                    ).unsqueeze(0)
                else:
                    n_nodes_i, n_nodes_ii = int(sum(idx[:i])), int(sum(idx[: i + 1]))
                    n_edges_i, n_edges_ii = int(sum(n_edges[:i])), int(
                        sum(n_edges[: i + 1])
                    )
                    x_i = feat[n_nodes_i:n_nodes_ii].unsqueeze(0)  # x_i (1, Ni, D)
                    adj_i = gen_Khop_adj(
                        adj[:, n_edges_i:n_edges_ii] - n_nodes_i, int(idx[i])
                    ).unsqueeze(0)

                x = x_i
                x = self.embedding(x)

                for gnn, ffn in zip(self.gnn_layers, self.FFN_layers):
                    x = gnn(x, adj_i)  # x (1, Ni, D2)
                    x = ffn(x, adj_i)

                x_ = x.detach()
                wandb.log(
                    {
                        "extractor mean": x_.mean().item(),
                        "extractor stds": x_.std().item(),
                    },
                    commit=False,
                )
                # 要考虑一下feature_norm加在sum之前还是之后比较好
                # 加在前面：每个位置的输入平衡，输出不平衡
                # 加在后面：每个位置的输入不平衡，输出平衡
                # 这里需要每个token近似相等，然后通过sum计算最后推理延时
                # 因此加在前面更为合理
                x = self.pre_reduction(x)
                if self.reduce_func == "sum":
                    x = x.sum(dim=1, keepdim=False)  # x (1, D)
                else:
                    raise NotImplementedError

                gnn_feat.append(x)

            gnn_feat = torch.cat(gnn_feat, dim=0)  # x(B, D)

            if self.norm_sf:
                static_feature = self.norm_sf_linear(static_feature)
                static_feature = self.norm_sf_drop(static_feature)
                static_feature = self.norm_sf_relu(static_feature)
            x = torch.cat([gnn_feat, static_feature], dim=1)

        elif self.dataset == "nasbench101" or self.dataset == "nasbench201":
            netcode, adj = data1, data2
            n_samples = netcode.size(0) if not self.real_test else 1
            for i in range(n_samples):
                layer_feats = []
                if self.real_test:
                    # For testing stage
                    adj_i = adj[:, : n_edges[i], : n_edges[i]]
                    x = netcode[:, : n_edges[i], :]
                else:
                    adj_i = adj[i, : n_edges[i], : n_edges[i]].unsqueeze(0)
                    x = netcode[i, : n_edges[i], :].unsqueeze(0)

                for gnn, ffn in zip(self.gnn_layers, self.FFN_layers):
                    x = gnn(x, adj_i)  # x (1, Ni, D2)
                    x = ffn(x, adj_i)
                    layer_feats.append(x.mean(dim=1, keepdim=False))  # [(1, D)]

                gnn_feat.append(torch.cat(layer_feats, dim=-1))  # (1, D*num_layers)

            gnn_feat = torch.cat(gnn_feat, dim=0)  # x(B, D*num_layers)

            layer_weights = self.fusion_mlp(gnn_feat)  # (B, num_layers)

            # (B, num_layers, D) * (B, num_layers, 1) --> sum(1, False) --> (B, D)
            x = (
                gnn_feat.contiguous().view(-1, self.n_attned_gnn, self.dim_out)
                * layer_weights.unsqueeze(2)
            ).sum(dim=1, keepdim=False)

        else:
            raise NotImplementedError

        pred = self.predictor(x)
        return pred


class SRLoss(nn.Module):
    def __init__(self):
        super(SRLoss, self).__init__()
        self.cal_loss = nn.L1Loss()

    def forward(self, predicts, target):
        B = predicts.shape[0]
        ori_pre = predicts
        ori_tar = target
        index = list(range(B))
        random.shuffle(index)
        predicts = predicts[index]
        target = target[index]
        v1 = ori_pre - predicts
        v2 = ori_tar - target
        loss = self.cal_loss(v1, v2)
        return loss
