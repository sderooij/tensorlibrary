"""
    Random initialization
"""
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
from tensorly.random import random_tensor
import tensorly as tl
from tensorly.backend import check_random_state
from .Decompositions.TensorTrain import TensorTrain


def tt_random(
    shape: Optional[int], ranks: Optional[int], random_state: int = 42, cores_only=False
):
    """tt_random generates a random orthogonal TensorTrain with norm 1

    Args:
        shape (Optional[int]): shape of the tensor-train (size of dimensions)
        ranks (Optional[int]): ranks of the tensor-train
        random_state (int, optional): random seed. Defaults to 42.
        cores_only (bool, optional): if True, only the cores are returned. Defaults to False.

    Returns:
        TensorTrain: the random tensor-train
    """
    # TODO : add option to generate a random tensor-train with a given norm and norm_index
    # cores = list()
    # d = len(shape)
    # assert len(ranks) == d - 1, "ranks must be of length d-1"
    # # if norm_index is None:
    # #     norm_index = d - 1
    # ranks = ranks.copy()
    # ranks = list(ranks)
    # ranks.insert(0, 1)
    # ranks.append(1)
    # ranks = [int(r) for r in ranks]
    # shape = [int(s) for s in shape]
    d = len(shape)
    cores = list()
    rnd = check_random_state(random_state)

    for k in range(0, d):
        core1 = tl.tensor(rnd.random_sample((int(ranks[k] * shape[k]), ranks[k + 1])))
        core1, _ = tl.qr(core1, mode="reduced")
        ranks[k + 1] = core1.shape[1]
        cores.append(tl.reshape(core1, (ranks[k], shape[k], ranks[k + 1]), order="F"))

    if cores_only:
        return cores
    else:
        return TensorTrain(cores=cores, norm_index=d - 1)
