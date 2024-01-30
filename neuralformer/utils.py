import collections.abc
from itertools import repeat


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
