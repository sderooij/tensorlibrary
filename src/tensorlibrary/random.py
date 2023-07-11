"""
    Random initialization
"""
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
from tensorly.random import random_tensor
import tensorly as tl
from .Decompositions.TensorTrain import TensorTrain


def tt_random(shape: Optional[int], ranks: Optional[int]):
    """tt_random generates a random orthogonal TensorTrain with norm 1

    Args:
        shape (Optional[int]): shape of the tensor-train (size of dimensions)
        ranks (Optional[int]): ranks of the tensor-train

    Returns:
        TensorTrain: the random tensor-train
    """
    #TODO : add option to generate a random tensor-train with a given norm and norm_index
    cores = list()
    d = len(shape)
    assert len(ranks) == d - 1, "ranks must be of length d-1"
    # if norm_index is None:
    #     norm_index = d - 1

    ranks = ranks.copy()
    ranks.insert(0, 1)
    ranks.append(1)
    ranks = [int(r) for r in ranks]
    shape = [int(s) for s in shape]

    for k in range(0, d):
        core1 = random_tensor((int(ranks[k] * shape[k]), ranks[k + 1]))
        core1, _ = tl.qr(core1, mode="reduced")
        ranks[k + 1] = core1.shape[1]
        cores.append(tl.reshape(core1, (ranks[k], shape[k], ranks[k + 1])))

    return TensorTrain(cores=cores, norm_index=d-1)
