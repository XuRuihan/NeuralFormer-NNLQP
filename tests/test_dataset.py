import argparse
import os
import sys
import unittest

sys.path.extend([".", ".."])
from neuralformer.dataset import GraphLatencyDataset


class TestDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = self.init_args()

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

    def test_latencydataset(self):
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

        print(train_model_types, test_model_types)
        sample_num_tr = [0, 1600] if train_model_types == test_model_types else -1
        sample_num_te = [1600, 2000] if train_model_types == test_model_types else -1

        train_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            self.args.embed_type,
            self.args.multires_p,
            override_data=self.args.override_data,
            model_types=train_model_types,
            sample_num=sample_num_tr,
        )
        test_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            self.args.embed_type,
            self.args.multires_p,
            override_data=self.args.override_data,
            model_types=test_model_types,
            sample_num=sample_num_te,
        )
        for i in range(len(train_set)):
            data, sf, n_edges, graph_name, plt_id = train_set[i]
            print(data.x.shape)
            print(data.edge_index.shape)
            # print(sf)
            # break
        # for i in range(len(test_set)):
        #     data, sf, n_edges, graph_name, plt_id = test_set[i]
        #     print(data.x.shape)
        #     print(data.edge_index)
        #     print(sf)
        #     break


if __name__ == "__main__":
    unittest.main()
