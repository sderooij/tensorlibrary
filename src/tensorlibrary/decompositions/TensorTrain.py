"""
    TensorTrain class
"""

from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import tensornetwork as tn
import tensorly as tl
from tensorlibrary.linalg import tt_svd
from primefac import primefac
import numpy as np


class TensorTrain:
    """
    TensorTrain class.
    """

    def __init__(
        self,
        tensor=None,
        cores=None,
        max_ranks: Optional[int] = np.inf,
        max_trunc_error: Optional[float] = 0.0,
        svd_method="tt_svd",
        relative: Optional[bool] = False,
        backend=None,
        norm_index=None,
    ):
        """Initialize a TensorTrain."""

        if backend is None:
            backend = tl.get_backend()

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
            self.norm_index = len(corelist) - 1

        self.__cores_to_nodes(corelist)
        self.ndims = len(self.cores)
        self.ranks = self.__get_ranks()
        self.errs = errs
        self.shape = self.__get_shape()

    # ================== Operators ========================
    def __add__(self, other):
        """Add two tensor-trains

        Args:
            other (TensorTrain): tensor-train to add

        Raises:
            Exception: if other is not a Tensor-Train

        Returns:
            TensorTrain: A_TT + B_TT
        """
        if isinstance(other, TensorTrain):
            assert tl.all(self.shape == other.shape), "TTs must be of same shape."

            new_rank = self.ranks + other.ranks
            new_cores = [tl.zeros((1, self.shape[0], new_rank[0]))]
            new_cores[0][0, :, : self.ranks[0]] = self.cores[0].tensor
            new_cores[0][0, :, self.ranks[0] :] = other.cores[0].tensor

            for k in range(1, self.ndims - 1):
                new_cores.append(
                    tl.zeros((new_rank[k - 1], self.shape[k], new_rank[k]))
                )
                new_cores[k][: self.ranks[k - 1], :, : self.ranks[k]] = self.cores[
                    k
                ].tensor
                new_cores[k][self.ranks[k - 1] :, :, self.ranks[k] :] = other.cores[
                    k
                ].tensor

            new_cores.append(tl.zeros((new_rank[-1], self.shape[-1], 1)))
            new_cores[-1][: self.ranks[-1], :] = self.cores[-1].tensor
            new_cores[-1][self.ranks[-1] :, :] = other.cores[-1].tensor

            return TensorTrain(cores=new_cores)
        else:
            raise Exception("Addition not defined for objects of this type.")

    def __sub__(self, other):
        return self + ((-1.0) * other)

    def __mul__(self, other: Union[int, float]):
        new_cores = [self.cores[k].tensor for k in range(0, self.ndims - 1)]
        new_cores.append(self.cores[-1].tensor * other)
        return TensorTrain(cores=new_cores)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other: Union[int, float]):
        return self * (1 / other)

    def __rtruediv__(self, other):
        return self / other

    def __matmul__(self, other):
        if isinstance(other, TensorTrain):
            return self.dot(other)

    def __len__(self):
        return self.ndims

    # def __rmatmul__(self, other):
    #     TODO: implement outerproduct

    # =============== Private Methods ===============
    def __get_ranks(self) -> list:
        return tl.tensor(
            [self.cores[k].shape[2] for k in range(0, len(self.cores) - 1)]
        )

    def __get_shape(self):
        return tuple([core.shape[1] for core in self.cores])

    def __cores_to_nodes(self, corelist: list):
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

    def update_core(self, core_index, new_core):
        assert self.cores[core_index].tensor.shape == new_core.shape
        new_core = tn.Node(new_core, name=f"core_{core_index}")
        self.connections[core_index] = tn.disconnect(self.connections[core_index])
        self.connections[core_index] = (
            self.cores[core_index - 1].edges[2] ^ new_core.edges[0]
        )
        if core_index != self.ndims - 1:
            self.connections[core_index + 1] = tn.disconnect(
                self.connections[core_index + 1]
            )
            self.connections[core_index + 1] = (
                new_core.edges[2] ^ self.cores[core_index + 1].edges[0]
            )
            self.cores[core_index] = new_core

        elif core_index == self.ndims - 1:
            self.connections[0] = tn.disconnect(self.connections[0])
            self.connections[0] = new_core.edges[2] ^ self.cores[0].edges[0]
            self.cores[core_index] = new_core
        else:
            raise Exception(IndexError, "Core index out of bounds")

        self.norm_index = None

        return self

    def contract(self, to_array=False, inplace=False):
        """contract the tensor-train to the full tensor

        Args:
            to_array (bool, optional): to array instead of tensornetwork. Defaults to False.
            inplace (bool, optional): contract the network inplace. Defaults to False.

        Returns:
            tn.network or array_like: full tensor
        """
        if inplace:
            network = self.cores
        else:
            network = self.cores.copy()

        output_edge_order = [core.edges[1] for core in network]
        if to_array:
            return tl.tensor(
                tn.contractors.auto(network, output_edge_order=output_edge_order).tensor
            )
        else:
            return tn.contractors.auto(network, output_edge_order=output_edge_order)

    def rank(self):
        return self.ranks

    def dot(self, other):
        """tensor dot product

        Args:
            other (TensorTrain or Tensor): tensor to contract with

        Raises:
            Exception: if other is not Tensor or TensorTrain object

        Returns:
            value: result of dot product
        """
        # assert tl.all(self.shape == other.shape)
        if isinstance(other, TensorTrain):
            net1 = tn.replicate_nodes(self.cores)
            net2 = tn.replicate_nodes(other.cores)

            for a, b in zip(net1, net2):
                a.edges[1] ^ b.edges[1]

            node = tn.contractors.auto(net1 + net2)
            return node.tensor
        elif tl.is_tensor(other):
            net1 = tn.replicate_nodes(self.cores)

            if len(other.shape) == 1:  # vector
                other = tl.reshape(other, tuple(self.shape), order="F")

            net2 = tn.Node(other)

            for k, core in enumerate(net1):
                core.edges[1] ^ net2.edges[k]

            node = tn.contractors.auto(net1 + [net2])
            return node.tensor
        else:
            raise Exception(TypeError, "dot method not defined for this type")

    def norm(self):
        if self.norm_index is None:
            return tl.sqrt(self.dot(self))
        else:
            return tl.norm(self.cores[self.norm_index].tensor)

    def orthogonalize(self, n=None, inplace=False):
        """orthogonalize the tensor-train

        Args:
            n (int, optional): location of the norm of the tensor-train. Defaults to None.
            inplace (bool, optional): in place processing. Defaults to False.

        Returns:
            TensorTrain: n-orthogonal tensor-train
        """
        assert n < self.ndims
        if n is None:
            n = self.ndims - 1

        ranks = tl.concatenate([[1], self.ranks, [1]], axis=0)
        N = self.shape
        d = self.ndims
        cores = [core.tensor for core in self.cores]

        # left orthogonalization
        for k in range(n):
            # left unfolding core
            A_L = cores[k].reshape((ranks[k] * N[k], ranks[k + 1]), order="F")
            # perform QR decomposition
            Q, R = tl.qr(A_L)
            # calculate new rank
            ranks[k + 1] = Q.shape[1]
            # Replace cores
            cores[k] = Q.reshape((ranks[k], N[k], ranks[k + 1]), order="F")
            # 1-mode product
            cores[k + 1] = tl.tenalg.mode_dot(cores[k + 1], R, 0, transpose=True)
        # right orthogonalization
        for k in reversed(range(n + 1, d)):
            # right unfolding core
            A_R = tl.reshape(cores[k], (ranks[k], N[k] * ranks[k + 1]), order="F")
            # QR
            Q, R = tl.qr(A_R.T)
            # contract core k-1 with R'
            cores[k - 1] = tl.tenalg.mode_dot(cores[k - 1], R, 2)
            # calculate new rank
            ranks[k] = Q.shape[1]
            # Replace cores
            cores[k] = Q.T.reshape((ranks[k], N[k], ranks[k + 1]), order="F")

        return TensorTrain(cores=cores, norm_index=n)

    def shiftnorm(self, new_index: int, inplace=True):
        """shift the norm of the tensor-traing to a new location

        Args:
            new_index (int): new location of the norm
            inplace (bool, optional): in place processing. Defaults to True.

        Returns:
            TensorTrain: tensor-train with shifted norm
        """
        if self.norm_index is None:
            return self.orthogonalize(n=new_index, inplace=inplace)

        if new_index == self.norm_index:
            return self

        d = self.ndims
        ranks = tl.concatenate([[1], self.ranks, [1]], axis=0)
        if inplace:
            if new_index > self.norm_index:
                for k in range(self.norm_index, new_index):
                    # left unfolding core
                    A_L = self.cores[k].tensor.reshape(
                        (ranks[k] * self.shape[k], ranks[k + 1]), order="F"
                    )
                    # perform QR decomposition
                    Q, R = tl.qr(A_L, mode="reduced")
                    # calculate new rank
                    ranks[k + 1] = Q.shape[1]
                    # Replace cores
                    self.update_core(
                        k, Q.reshape((ranks[k], self.shape[k], ranks[k + 1]), order="F")
                    )
                    # 1-mode product
                    self.update_core(
                        k + 1,
                        tl.tenalg.mode_dot(
                            self.cores[k + 1].tensor, R, 0, transpose=True
                        ),
                    )
            elif new_index < self.norm_index:
                for k in reversed(range(new_index + 1, self.norm_index + 1)):
                    # right unfolding core
                    A_R = tl.reshape(
                        self.cores[k].tensor,
                        (ranks[k], self.shape[k] * ranks[k + 1]),
                        order="F",
                    )
                    # QR
                    Q, R = tl.qr(A_R.T, mode="reduced")
                    # contract core k-1 with R'
                    self.update_core(
                        k - 1, tl.tenalg.mode_dot(self.cores[k - 1].tensor, R, 2)
                    )
                    # calculate new rank
                    ranks[k] = Q.shape[1]
                    # Replace cores
                    self.update_core(
                        k,
                        Q.T.reshape((ranks[k], self.shape[k], ranks[k + 1]), order="F"),
                    )
            self.norm_index = new_index
            return self
        else:
            cores = [core.tensor for core in self.cores]
            # ranks = tl.concatenate([[1], self.ranks, [1]], axis=0)
            if new_index > self.norm_index:
                for k in range(self.norm_index, new_index):
                    # left unfolding core
                    A_L = cores[k].reshape(
                        (ranks[k] * self.shape[k], ranks[k + 1]), order="F"
                    )
                    # perform QR decomposition
                    Q, R = tl.qr(A_L, mode="reduced")
                    # calculate new rank
                    ranks[k + 1] = Q.shape[1]
                    # Replace cores
                    cores[k] = Q.reshape(
                        (ranks[k], self.shape[k], ranks[k + 1]), order="F"
                    )
                    # 1-mode product
                    cores[k + 1] = tl.tenalg.mode_dot(
                        cores[k + 1], R, 0, transpose=True
                    )
            elif new_index < self.norm_index:
                for k in reversed(range(new_index, self.norm_index)):
                    # right unfolding core
                    A_R = tl.reshape(
                        cores[k], (ranks[k], self.shape[k] * ranks[k + 1]), order="F"
                    )
                    # QR
                    Q, R = tl.qr(A_R.T, mode="reduced")
                    # contract core k-1 with R'
                    cores[k - 1] = tl.tenalg.mode_dot(cores[k - 1], R, 2)
                    # calculate new rank
                    ranks[k] = Q.shape[1]
                    # Replace cores
                    cores[k] = Q.T.reshape(
                        (ranks[k], self.shape[k], ranks[k + 1]), order="F"
                    )
            return TensorTrain(cores=cores, norm_index=new_index)

    def outer(self, tens):
        if isinstance(tens, "TensorTrain"):
            return

    def tkron(self, tens):
        # TODO: implement tensor kronecker product
        return self


def QTT(
    vec,
    q=None,
    max_ranks: Optional[int] = np.inf,
    max_trunc_error: Optional[float] = 0.0,
    svd_method="tt_svd",
    relative: Optional[bool] = False,
    backend="numpy",
):
    if q is None:
        q_gen = primefac(len(vec))
        q = [fac for fac in q_gen]
    # assert tl.prod(q) == len(vec)
    tensor = tl.reshape(vec, tuple(q))
    return TensorTrain(
        tensor=tensor,
        max_ranks=max_ranks,
        max_trunc_error=max_trunc_error,
        svd_method=svd_method,
        relative=relative,
        backend=backend,
    )

    # def contract(self, to_array=True, inplace=False):
    #     return tl.reshape(super().contract(to_array=to_array, inplace=inplace), tl.prod(self.shape))
