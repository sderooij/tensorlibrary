"""
    Random initialization
"""
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
from tensorly.random import random_tensor
import tensorly as tl
from tensorly.backend import check_random_state
from .decompositions.TensorTrain import TensorTrain
from scipy import linalg


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

    d = len(shape)
    rnd = check_random_state(random_state)
    cores = [None for _ in range(d)]
    cores[0] = tl.tensor(rnd.random_sample((ranks[0], int(shape[0]), ranks[1])))
    cores[0] = cores[0] / tl.norm(cores[0])

    for k in range(d-1, 0, -1):
        core1 = tl.tensor(rnd.random_sample((int(ranks[k+1] * shape[k]), ranks[k])))
        Q = linalg.orth(core1)
        cores[k] = Q.T.reshape((ranks[k], int(shape[k]), ranks[k+1]))

    if cores_only:
        return cores
    else:
        return TensorTrain(cores=cores, norm_index=d - 1)
