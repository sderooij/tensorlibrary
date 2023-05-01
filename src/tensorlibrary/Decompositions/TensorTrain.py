"""
    TensorTrain class
"""

from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import tensornetwork as tn
import tensorly as tl
from tensorlibrary.linalg import truncated_svd

# from numba import njit
import numpy as np
from tensornetwork.backend_contextmanager import get_default_backend


class TensorTrain:
    """
    TensorTrain class.
    """

    def __init__(
        self,
        tensor=None,
        cores=None,
        max_ranks: Optional[int] = np.infty,
        max_trunc_error: Optional[float] = 0.0,
        svd_method="tt_svd",
        relative: Optional[bool] = False,
        backend="numpy",
    ):
        """Initialize a TensorTrain.
        Args:
          backend: The name of the linalg that should be used to perform
            contractions. Available backends are currently 'numpy', 'tensorflow',
            'pytorch', 'jax'
        """

        if backend is None:
            backend = get_default_backend()

        if cores is not None and tensor is None:
            corelist = cores
            errs = None
            self.norm_index = None
        else:
            corelist, ranks, errs = tt_svd(
                tensor,
                max_ranks=max_ranks,
                max_trunc_error=max_trunc_error,
                relative=relative,
            )
            self.norm_index = len(corelist)

        self.cores_to_nodes(corelist)
        self.ndims = len(self.cores)
        self.ranks = self.get_ranks()
        self.errs = errs
        self.shape = self.get_shape()

    # ================== Operators ========================
    def __add__(self, other):

        if isinstance(other, TensorTrain):
            assert tl.all(self.shape == other.shape), "TTs must be of same shape."

            new_rank = self.ranks + other.ranks
            new_cores = [tl.zeros((1, self.shape[0], new_rank[0]))]
            new_cores[0][0, :, :self.ranks[0]] = self.cores[0].tensor
            new_cores[0][0, :, self.ranks[0]:] = other.cores[0].tensor

            for k in range(1, self.ndims-1):
                new_cores.append(tl.zeros((new_rank[k-1], self.shape[k], new_rank[k])))
                new_cores[k][:self.ranks[k-1], :, :self.ranks[k]] = self.cores[k].tensor
                new_cores[k][self.ranks[k-1]:, :, self.ranks[k]:] = other.cores[k].tensor

            new_cores.append(tl.zeros((new_rank[-1], self.shape[-1], 1)))
            new_cores[-1][:self.ranks[-1], :] = self.cores[-1].tensor
            new_cores[-1][self.ranks[-1]:, :] = other.cores[-1].tensor

            return TensorTrain(cores=new_cores)
        else:
            raise Exception("Addition not defined for objects of this type.")

    def __sub__(self, other):
        return self + ((-1.)*other)

    def __mul__(self, other: Union[int, float]):

        new_cores = [self.cores[k].tensor for k in range(0, self.ndims-1)]
        new_cores.append(self.cores[-1].tensor * other)
        return TensorTrain(cores=new_cores)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other: Union[int, float]):
        return self * (1/other)

    # ============ Methods ====================
    def copy(self):
        cores = [core.tensor for core in self.cores]
        return TensorTrain(cores=cores)

    def contract(self, to_array=False, inplace=False):
        if inplace:
            network = self.cores
        else:
            network = self.cores.copy()

        output_edge_order = [core.edges[1] for core in network]
        if to_array:
            return tn.contractors.auto(
                network, output_edge_order=output_edge_order
            ).tensor
        else:
            return tn.contractors.auto(network, output_edge_order=output_edge_order)

    def rank(self):
        return self.ranks

    def get_ranks(self):
        return tl.tensor([self.cores[k].shape[2] for k in range(0, len(self.cores)-1)])

    def cores_to_nodes(self, corelist):
        self.cores = [tn.Node(core, f"core_{i}") for i, core in enumerate(corelist)]
        self.connections = [
            self.cores[k].edges[2] ^ self.cores[k + 1].edges[0]
            for k in range(-1, len(corelist) - 1)
        ]
        return self

    def get_shape(self):
        return tl.tensor([core.shape[1] for core in self.cores])

    def dot(self, other):
        assert self.shape == other.shape
        if isinstance(other, TensorTrain):
            net1 = tn.replicate_nodes(self.cores)
            net2 = tn.replicate_nodes(other.cores)

            connect = [a.edges[1] ^ b.edges[1] for a, b in zip(net1, net2)]
            return tn.contractors.auto(net1+net2)


def tt_svd(
    tensor,
    max_ranks: Optional[int] = np.inf,
    max_trunc_error: Optional[float] = 0.0,
    relative: Optional[bool] = False,
):
    corelist = []
    errs = []
    ranks = []
    d = len(tensor.shape)
    if not relative:
        max_trunc_error = (
            max_trunc_error * tl.norm(tensor) * (1 / tl.sqrt(d - 1))
        )

    if not hasattr(max_ranks, "__len__"):
        max_ranks = max_ranks * tl.ones(d - 1)
    if not hasattr(max_trunc_error, "__len__"):
        max_trunc_error = max_trunc_error * tl.ones(d - 1)

    sz = tensor.shape
    # first core
    tensor = tl.reshape(tensor, (sz[0], tl.prod(sz[1:])))
    u, s, vh, err = truncated_svd(
        tensor,
        max_rank=max_ranks[0],
        max_trunc_error=max_trunc_error[0],
        relative=relative,
    )
    corelist.append(tl.reshape(u, (1, u.shape[0], u.shape[1])))
    tensor = np.diag(s) @ vh
    errs.append(err)
    ranks.append(u.shape[1])

    for i in range(1, d - 1):
        tensor = tl.reshape(tensor, (sz[i] * ranks[i - 1], np.prod(sz[i + 1 :])))
        u, s, vh, err = truncated_svd(
            tensor,
            max_rank=max_ranks[i],
            max_trunc_error=max_trunc_error[i],
            relative=relative,
        )
        corelist.append(tl.reshape(u, (ranks[i - 1], sz[i], u.shape[1])))
        tensor = np.diag(s) @ vh
        errs.append(err)
        ranks.append(u.shape[1])

    corelist.append(tl.reshape(tensor, (tensor.shape[0], tensor.shape[1], 1)))

    return corelist, ranks, tl.norm(errs)
