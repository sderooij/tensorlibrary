"""
    TensorTrain class
"""

from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import tensornetwork as tn
import tensorly as tl
from tensorlibrary.linalg import tt_svd

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
        norm_index=None,
    ):
        """Initialize a TensorTrain.
        """

        if backend is None:
            backend = get_default_backend()

        if cores is not None and tensor is None:
            corelist = cores
            errs = None
            self.norm_index = norm_index
        else:
            corelist, ranks, errs = tt_svd(
                tensor,
                max_ranks=max_ranks,
                max_trunc_error=max_trunc_error,
                relative=relative,
            )
            self.norm_index = len(corelist)-1

        self.__cores_to_nodes(corelist)
        self.ndims = len(self.cores)
        self.ranks = self.__get_ranks()
        self.errs = errs
        self.shape = self.__get_shape()

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

    def __rtruediv__(self, other):
        return self / other

    def __matmul__(self, other):
        if isinstance(other, TensorTrain):
            return self.dot(other)

    # def __rmatmul__(self, other):
    #     TODO: implement outerproduct

    # =============== Private Methods ===============
    def __get_ranks(self):
        return tl.tensor([self.cores[k].shape[2] for k in range(0, len(self.cores)-1)])

    def __get_shape(self):
        return tuple([core.shape[1] for core in self.cores])

    def __cores_to_nodes(self, corelist):
        self.cores = [tn.Node(core, f"core_{i}") for i, core in enumerate(corelist)]
        self.connections = [
            self.cores[k].edges[2] ^ self.cores[k + 1].edges[0]
            for k in range(-1, len(corelist) - 1)
        ]
        return self

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

    def dot(self, other):
        # assert tl.all(self.shape == other.shape)
        if isinstance(other, TensorTrain):
            net1 = tn.replicate_nodes(self.cores)
            net2 = tn.replicate_nodes(other.cores)

            for a, b in zip(net1, net2):
                a.edges[1] ^ b.edges[1]

            node = tn.contractors.auto(net1+net2)
            return node.tensor
        elif tl.is_tensor(other):
            net1 = tn.replicate_nodes(self.cores)

            if len(other.shape) == 1:   # vector
                other = tl.reshape(other, tuple(self.shape))

            net2 = tn.Node(other)

            for k, core in enumerate(net1):
                core.edges[1] ^ net2.edges[k]

            node = tn.contractors.auto(net1 + [net2])
            return node.tensor

    def norm(self):
        if self.norm_index is None:
            return tl.sqrt(self.dot(self))
        else:
            return tl.norm(self.cores[self.norm_index].tensor)

    def orthogonalize(self, n=None, inplace=True):
        #TODO: implement n-orthogonlization

        assert n < self.ndims
        if n is None:
            n = self.ndims-1

        ranks = tl.concatenate([[1], self.ranks, [1]], axis=0)
        N = self.shape
        d = self.ndims
        # newranks = ranks
        cores = [core.tensor for core in self.cores]
        # newcores = []
        # left orthogonalization
        for k in range(n):
            # left unfolding core
            A_L = cores[k].reshape((ranks[k] * N[k], ranks[k + 1]))
            # perform QR decomposition
            Q, R = tl.qr(A_L)
            # save old rank
            oldRank = ranks[k + 1]
            # calculate new rank
            ranks[k + 1] = Q.shape[1]
            # Replace cores
            cores[k] = Q.reshape((ranks[k], N[k], ranks[k + 1]))
            # 1-mode product
            core = cores[k + 1].reshape((oldRank, N[k + 1] * ranks[k + 2]))
            core = np.dot(R, core)
            cores[k + 1] = core.reshape((R.shape[1], N[k + 1], ranks[k + 2]))

        # right orthogonalization
        for k in reversed(range(n + 1, d)):
            # right unfolding core
            A_R = cores[k].reshape((ranks[k], N[k] * ranks[k + 1]))
            # QR
            Q, R = tl.qr(A_R.T)
            # contract core k-1 with R'
            core = cores[k - 1].reshape((N[k - 1] * ranks[k - 1], ranks[k]))
            core = np.dot(core, R.T)
            # calculate new rank
            ranks[k] = Q.shape[1]
            # Replace cores
            cores[k] = Q.T.reshape((ranks[k], N[k], ranks[k + 1]))
            cores[k - 1] = core.reshape((ranks[k - 1], N[k - 1], ranks[k]))

        return TensorTrain(cores=cores, norm_index=n)





