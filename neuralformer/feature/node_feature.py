import numpy as np
from .feature_utils import OPS, ATTRS, FEATURE_LENGTH, FEATURE_DIM
from .position_encoding import get_embedder

# int -> one hot embedding
def embed_op_code(op_type):
    length = FEATURE_LENGTH["op_type"]
    if op_type not in OPS:
        return np.zeros(length, dtype="float32")
    op_code = OPS[op_type]["code"] - 1
    '''
    if op_type == 'Clip':
        raise Exception('Clip')
    if op_type == 'ReduceMean':
        raise Exception('ReduceMean')
    '''
    if op_code >= length:
        raise Exception("op code of {}: {} greater than one-hot length {}!".format(
            op_type, op_code, length))
    return np.eye(length, dtype="float32")[op_code]

class EmbedValue:
    # int value embedding
    @staticmethod
    def embed_int(x, center=0, scale=1):
        x = np.array([int(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    # float value embedding
    @staticmethod
    def embed_float(x, center=0, scale=1):
        x = np.array([float(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    # bool value embedding
    @staticmethod
    def embed_bool(x, center=0, scale=1):
        x = np.array([int(bool(x))], dtype="float32")
        return (x - center) / np.abs(scale)

    # tuple value embedding
    @staticmethod
    def embed_tuple(x, length, center=0, scale=1):
        x = np.array(x, dtype="float32").reshape(-1)
        if x.size > length:
            x = x[:length]
        if x.size < length:
            x = np.concatenate([x, np.zeros(length - x.size, dtype="float32")])
        if not isinstance(center, list):
            center = [center] * x.size
        if not isinstance(scale, list):
            scale = [scale] * x.size
        center = np.array(center, dtype="float32")
        scale = np.array(scale, dtype="float32")
        return (x - center) / np.abs(scale)


# attrs embedding
def embed_attrs(op_type, attrs, embed_type, input_type='np_array'):
    length = FEATURE_LENGTH["attrs"]
    total_dim = FEATURE_DIM["attrs"]
    dim = total_dim // length #feature dim of each attribute
    if op_type not in OPS:
        return np.zeros(total_dim, dtype="float32")

    fn, _ = get_embedder(dim // 2, embed_type, input_type)

    feats = []
    for name in OPS[op_type]["attrs"]:
        assert name in attrs, "attr {} for {} need to be encoded but not included!".format(name, op_type)
        assert name in ATTRS, "attr {} for {} does not defined in ATTRS!".format(name, op_type)

        attr_value = attrs[name]
        attr_def = ATTRS[name]
        feat = getattr(EmbedValue, "embed_" + attr_def[0])(attr_value, *attr_def[1:])
        feat = fn(feat) #(-1,dim)=(1,dim)
        feats.append(feat)

    # concat attr features
    feats = np.concatenate(feats) if len(feats) > 0 else np.zeros(total_dim, dtype="float32")
    feat_len = feats.size
    if feat_len > total_dim:
        raise Exception("tuple length {} is grater than the embed length {}".format(
            feat_len, total_dim))
    if feat_len < total_dim:
        feats = np.concatenate([feats, np.zeros(total_dim - feat_len, dtype="float32")])
    return feats


def embed_shape(shape, embed_type, input_type='np_array'):
    length = FEATURE_LENGTH["output_shape"]
    total_dim = FEATURE_DIM["output_shape"]
    dim = total_dim // length

    #shape_value = EmbedValue.embed_tuple(shape, length)
    processed_shape = EmbedValue.embed_tuple(
    shape,
    FEATURE_LENGTH["output_shape"]
        )
    
    fn, _ = get_embedder(dim // 2, embed_type, input_type)
    
    feats = []
    for val in processed_shape:
        value = EmbedValue.embed_float(val)
        feat = fn(value)
        feats.append(feat)
        
    feats = np.concatenate(feats)
    feat_len = feats.size
    if feat_len < total_dim:
        feats = np.concatenate([feats, np.zeros(total_dim - feat_len, dtype="float32")])
    return feats


# networkx_G -> op_code_embeddings & attrs_embeddings
# output_shapes -> output_shape_embeddings
def extract_node_features(networkx_G, output_shapes, batch_size, embed_type):
    embeddings = {}

    print(embed_type)
    
    for node in networkx_G.nodes.data():
        attrs = node[1]["attr"].attributes
        node_name = node[0]
        op_type = attrs["type"]

        # one hot op_code embedding
        op_code_embedding = embed_op_code(op_type)

        # fixed length embedding for attrs, need normalize?
        attrs_embedding = embed_attrs(op_type, attrs, embed_type)

        # fixed length embedding for output shape, need normalize?
        assert node_name in output_shapes, "could not find output shape for node {}".format(node_name)
        output_shape_embedding = embed_shape(output_shapes[node_name], embed_type)

        # concat to the final node feature
        embeddings[node_name] = np.concatenate([
            op_code_embedding,
            attrs_embedding,
            output_shape_embedding,
        ])
        # print(op_type, len(embeddings[node_name]), embeddings[node_name])

    return embeddings
