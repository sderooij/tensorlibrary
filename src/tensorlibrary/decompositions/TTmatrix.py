"""
    Tensor-train matrix class
"""

from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import tensornetwork as tn
import tensorly as tl
from tensorlibrary.linalg import tt_svd
from primefac import primefac
import numpy as np


# from tensornetwork.backend_contextmanager import get_default_backend


class TTmatrix:
    def __init__(
        self,
        matrix=None,
        q_row=None,
        q_col=None,
        cores=None,
        max_ranks: Optional[int] = np.infty,
        max_trunc_error: Optional[float] = 0.0,
        svd_method="tt_svd",
        relative: Optional[bool] = False,
        backend=None,
        norm_index=None,
    ):
        if backend is None:
            backend = tl.get_backend()

        if cores is not None and matrix is None:
            corelist = cores
            errs = None
            self.norm_index = norm_index
        else:
            assert len(matrix.shape) == 2
            sz = matrix.shape
            if q_row is not None:
                assert tl.prod(q_row) == sz[0]
                assert isinstance(q_row, "list")
            else:
                q_gen = primefac(sz[0])
                q_row = [fac for fac in q_gen]
            if q_col is not None:
                assert tl.prod(q_col) == sz[1]
                assert isinstance(q_col, "list")
            else:
                q_gen = primefac(sz[1])
                q_col = [fac for fac in q_gen]

            d_row = len(q_row)
            d_col = len(q_col)
            d_diff = d_row - d_col

            if d_diff > 0:
                q_col += [1] * d_diff
                d_col = d_col + d_diff
            elif d_diff < 0:
                q_row += [1] * abs(d_diff)
                d_row = d_row - d_diff

            # assert len(d_row) == len(d_col)

            q_tot = q_row + q_col
            tensor = tl.reshape(matrix, tuple(q_tot))
            d_tot = len(q_tot)
            row_index = [2 * i for i in range(int((d_tot + 1) / 2))]
            col_index = [2 * i + 1 for i in range(int(d_tot / 2))]
            old_dims = [i for i in range(0, d_tot)]
            tensor = tl.moveaxis(tensor, old_dims, row_index + col_index)
            tensor = tl.reshape(tensor, tuple(tl.prod([q_row, q_col], axis=0)))

            corelist, ranks, errs = tt_svd(
                tensor,
                max_ranks=max_ranks,
                max_trunc_error=max_trunc_error,
                relative=relative,
            )
            self.norm_index = len(corelist) - 1
            for i, core in enumerate(corelist):
                sz = core.shape
                corelist[i] = tl.reshape(core, (sz[0], q_row[i], q_col[i], sz[2]))

        self.row_shape = tuple(q_row)
        self.col_shape = tuple(q_col)
        self.__cores_to_nodes(corelist)
        self.ndims = len(self.cores)
        self.ranks = self.__get_ranks()
        self.errs = errs
        self.shape = self.__get_shape()

    # ================== Operators ========================

    # =============== Private Methods ===============
    def __get_ranks(self) -> list:
        return tl.tensor(
            [self.cores[k].shape[3] for k in range(0, len(self.cores) - 1)]
        )

    def __get_shape(self):
        return [tuple([core.shape[1], core.shape[2]]) for core in self.cores]

    def __cores_to_nodes(self, corelist: list):
        self.cores = [tn.Node(core, f"core_{i}") for i, core in enumerate(corelist)]
        self.connections = [
            self.cores[k].edges[3] ^ self.cores[k + 1].edges[0]
            for k in range(-1, len(corelist) - 1)
        ]
        return self

    # ============ Methods ====================
    def contract(self, *, to_array=True, inplace=False):
        if inplace:
            network = self.cores
        else:
            network = self.cores.copy()

        row_edges = [core.edges[1] for core in network]
        col_edges = [core.edges[2] for core in network]
        output_edge_order = row_edges + col_edges
        if to_array:
            tensor = tl.tensor(
                tn.contractors.auto(network, output_edge_order=output_edge_order).tensor
            )
            return tl.reshape(
                tensor, (tl.prod(self.row_shape), tl.prod(self.col_shape))
            )
        else:
            return tn.contractors.auto(network, output_edge_order=output_edge_order)
