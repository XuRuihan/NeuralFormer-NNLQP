# import sys
# sys.path.extend(["."])
from neuralformer.feature.graph_feature import parse_from_onnx

onnx_path = "dataset/unseen_structure/onnx/nnmeter_alexnet/nnmeter_alexnet_transform_1992.onnx"
# onnx_path = "dataset/unseen_structure/onnx/nnmeter_resnet18/nnmeter_resnet18_transform_1869.onnx"

batch_size = 1
nx_G, output_shapes, flops, params, macs, node_flops, onnx_G = parse_from_onnx(
    onnx_path, batch_size
)


# for node in nx_G.nodes.data():
#     attrs = node[1]["attr"].attributes
#     node_name = node[0]
#     op_type = attrs["type"]

#     print(op_type)
#     print(attrs)
print(batch_size, flops / 1e9, params / 1e9, macs / 1e9)

print(output_shapes)
