import argparse
import os
import sys
import unittest

import torch
from torch_geometric.loader import DataLoader

sys.path.extend([".", ".."])

from neuralformer.dataset import GraphLatencyDataset
from neuralformer.model import Net


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = self.init_args()
        self.device = "cpu"

        path = "output/neuralformer_2_checkpoints/ckpt_latest.pth"
        ckpt = torch.load(path, map_location="cpu")
        self.model = Net(
            "nnlqp", True, "LN", num_node_features=152, use_degree=True, norm_sf=True
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument(
            "--gpu", type=int, default=0, help="gpu id, < 0 means no gpu"
        )
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--warmup_rate", type=float, default=0.1)

        parser.add_argument("--norm_sf", action="store_true")
        parser.add_argument("--use_degree", action="store_true")

        parser.add_argument(
            "--dataset", type=str, default="nnlqp", help="nnlqp|nasbench101|nasbench201"
        )
        parser.add_argument(
            "--data_root", type=str, default="dataset/unseen_structure/data"
        )
        parser.add_argument(
            "--all_latency_file",
            type=str,
            default="dataset/unseen_structure/gt_stage.txt",
        )
        parser.add_argument("--test_model_type", type=str)
        parser.add_argument("--train_model_types", type=str)
        parser.add_argument("--train_test_stage", action="store_true")
        parser.add_argument("--train_num", type=int, default=-1)

        parser.add_argument("--onnx_dir", type=str, default="dataset/unseen_structure")
        parser.add_argument("--override_data", action="store_true")

        parser.add_argument("--log", type=str)
        parser.add_argument("--pretrain", type=str)
        parser.add_argument("--resume", type=str)
        parser.add_argument("--model_dir", type=str)

        parser.add_argument("--print_freq", type=int, default=10)
        parser.add_argument("--ckpt_save_freq", type=int, default=20)
        parser.add_argument("--test_freq", type=int, default=10)
        parser.add_argument("--only_test", action="store_true")
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--num_node_features", type=int, default=44)

        # args for NAR-Former V2
        parser.add_argument("--embed_type", type=str)
        parser.add_argument("--n_attned_gnn", type=int, default=2)
        parser.add_argument("--feat_shuffle", type=bool, default=False)
        parser.add_argument("--ffn_ratio", type=int, default=4)
        parser.add_argument("--glt_norm", type=str, default=None)

        # args for NasBench
        parser.add_argument("--multires_x", type=int, default=0)
        parser.add_argument("--multires_p", type=int, default=0)
        parser.add_argument("--optype", type=str, default="onehot")
        parser.add_argument(
            "--lambda_sr",
            type=float,
            default=0,
            help="only be used in accuracy prediction",
        )
        parser.add_argument(
            "--lambda_cons",
            type=float,
            default=0,
            help="only be used in accuracy prediction",
        )

        args = parser.parse_args()
        args.norm_sf = True if args.norm_sf else False

        return args

    def unpack_batch(self, batch):
        if self.args.dataset == "nnlqp":
            data, static_feature, n_edges, _, plt_id = batch
            y = data.y.view(-1, 1).to(self.device)
            data = data.to(self.device)
            static_feature = static_feature.to(self.device)
            return data, static_feature, n_edges, y
        else:
            if not self.args.only_test and self.args.lambda_cons > 0:
                data1, data2 = batch
                code1, adj1, N1, V_A1, T_A1 = data1
                code2, adj2, N2, V_A2, T_A2 = data2
                code = torch.cat([code1, code2], dim=0).to(self.device)
                adj = torch.cat([adj1, adj2], dim=0).to(self.device)
                N = torch.cat([N1, N2], dim=0)
                V_A = torch.cat([V_A1, V_A2], dim=0)
            else:
                code, adj, N, V_A, T_A = batch
                code = code.to(self.device)
                adj = adj.to(self.device)

            if self.args.only_test:
                return code, adj, N, T_A.view(-1, 1).to(self.device)
            else:
                return code, adj, N, V_A.view(-1, 1).to(self.device)

    @torch.no_grad()
    def test_model(self):
        model_types = set()
        for line in open(self.args.all_latency_file).readlines():
            model_types.add(line.split()[4])
        # assert self.args.test_model_type in model_types
        test_model_types = set([self.args.test_model_type])
        if self.args.train_model_types:
            train_model_types = set(self.args.train_model_types.split(","))
            train_model_types = train_model_types & model_types
        else:
            train_model_types = model_types - test_model_types
        assert len(train_model_types) > 0

        sample_num_tr = [0, 1600] if train_model_types == test_model_types else -1
        sample_num_te = [1600, 2000] if train_model_types == test_model_types else -1

        test_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            self.args.embed_type,
            self.args.multires_p,
            override_data=self.args.override_data,
            model_types=train_model_types,
            sample_num=sample_num_te,
        )
        test_loader = DataLoader(
            dataset=test_set, batch_size=self.args.batch_size, shuffle=True
        )

        xs = []
        edge_indices = []
        ys = []
        for iteration, batch in enumerate(test_loader):
            torch.cuda.empty_cache()

            data1, data2, n_edges, y = self.unpack_batch(batch)

            # NNLQP: data1=data, data2=static feature
            # NasBench101/201: data1=netcode, data2=adjacency matrix
            pred_cost = self.model(data1, data2, n_edges)
            print(pred_cost.mean(), pred_cost.std(), y)
            # print(data1.x.shape, data1.edge_index.shape, y)
            if iteration > 3:
                break
        #     xs.append(torch.tensor(data1.x.shape))
        #     edge_indices.append(torch.tensor(data1.edge_index.shape))
        #     ys.append(y)
        # xs = torch.stack(xs)
        # edge_indices = torch.stack(edge_indices)
        # ys = torch.cat(ys)
        # torch.save(xs, "node.pt")
        # torch.save(edge_indices, "edge.pt")
        # torch.save(ys, "ys.pt")


if __name__ == "__main__":
    unittest.main()
