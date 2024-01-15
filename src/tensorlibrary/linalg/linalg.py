"""
	Linear algebra helper functions
"""

import numpy as np
import tensorly as tl
from numba import njit
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence


def truncated_svd(
    mat,
    max_rank: int = np.inf,
    max_trunc_error: Optional[float] = 0.0,
    relative: Optional[bool] = False,
):
    u, s, vh = tl.svd(mat, full_matrices=False)
    if np.isinf(max_rank) and max_trunc_error == 0.0:
        return u, s, vh, 0.0

    if max_trunc_error != 0.0:
        err = 0.0
        k = len(s) - 1
        if relative:
            max_trunc_error = max_trunc_error * np.max(s)
        while err <= max_trunc_error:
            err = err + s[k]
            k -= 1
        max_rank = min([k + 2, max_rank])

    max_rank = int(min([max_rank, len(s)]))
    err = tl.norm(s[max_rank:])
    u = u[:, :max_rank]
    s = s[:max_rank]
    vh = vh[:max_rank, :]

    return u, s, vh, err


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
        max_trunc_error = max_trunc_error * tl.norm(tensor) * (1 / tl.sqrt(d - 1))

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


def dot_kron(mat1, mat2):
    """
    Computes the row-wise right kronecker product of two matrices.
    Args:
        mat1: first matrix (N x M)
        mat2: second matrix (N x P)

    Returns:
        matrix of size N x (M*P)
    """
    return tl.tenalg.khatri_rao([mat2.T, mat1.T]).T


def multi_dot_kron(matlist):
    """
    Computes the row-wise right kronecker product of a list of matrices.
    Args:
        matlist: list of matrices

    Returns:
        matrix of size N x (M1*M2*...*MP)
    """
    return tl.tenalg.khatri_rao([mat.T for mat in matlist]).T


@njit(parallel=True)
def dot_kron_numba(a, b):
    """
    Computes the row-wise right kronecker product of two matrices.
    Args:
        a: first matrix (N x M)
        b: second matrix (N x P)

    Returns:
        matrix of size N x (M*P)
    """
    at = np.reshape(a, (a.shape[0], 1, a.shape[1]))
    bt = np.reshape(b, (b.shape[0], b.shape[1], 1))
    temp = at * bt
    return np.reshape(temp, (a.shape[0], -1))


