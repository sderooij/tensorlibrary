"""
	Linear algebra helper functions
"""

import numpy as np
import tensorly as tl
from numba import njit
from numba.typed import List
from typing import Any, Optional, Text, Type, Union, Dict, Sequence


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


@njit()
def dot_kron_numba(a, b):
    """
    Computes the right face-splitting product of two matrices.
    Args:
        a: first matrix (N x M)
        b: second matrix (N x P)

    Returns:
        matrix of size N x (P*M)
    """
    at = np.reshape(a, (a.shape[0], 1, a.shape[1]))
    bt = np.reshape(b, (b.shape[0], b.shape[1], 1))
    temp = at * bt
    return np.reshape(temp, (a.shape[0], -1))


def dot_kron_numpy(a, b):
    """
    Computes the right face-splitting product of two matrices.
    Args:
        a: first matrix (N x M)
        b: second matrix (N x P)

    Returns:
        matrix of size N x (P*M)
    """
    at = np.reshape(a, (a.shape[0], 1, a.shape[1]))
    bt = np.reshape(b, (b.shape[0], b.shape[1], 1))
    temp = at * bt
    return np.reshape(temp, (a.shape[0], -1))


def khatrao(a, b):
    """
    Computes the khatri-rao product of two matrices. Transpose of face-splitting product.
    Args:
        a: first matrix (M x N)
        b: second matrix (P x N)
    Returns:
        matrix of size (M x P) x N
    """
    at = np.reshape(a, (a.shape[0], 1, a.shape[1]))
    bt = np.reshape(b, (1, b.shape[0], b.shape[1]))
    temp = at * bt
    return np.reshape(temp, (-1, a.shape[1]))


def dot_r1(a, b):
    """
    Dot product of two rank-1 tensors. (kronecker products)
    Args:
        a: list of d vectors that define the rank-1 tensor (can be of different lengths)
        b: list of d vectors that define the rank-1 tensor (same dimensions as a)

    Returns:
        scalar: dot product of the two rank-1 tensors
    """
    prod = 1.0
    for i, ai in enumerate(a):
        prod *= (ai.T @ b[i]).item()

    return prod


@njit()
def dot_r1_numba(a, b):
    """
    Dot product of two rank-1 tensors. (kronecker products)
    Args:
        a: numba.typed.List of d vectors that define the rank-1 tensor (can be of different lengths)
        b: numba.typed.List of d vectors that define the rank-1 tensor (same dimensions as a)

    Returns:
        scalar: dot product of the two rank-1 tensors
    """
    prod = 1.0
    for i, ai in enumerate(a):
        prod *= (ai.T @ b[i]).item()

    return prod


def cp_dot(tensor1, tensor2, *, engine: Optional[Text] = "numpy"):
    if isinstance(tensor1, tl.cp_tensor.CPTensor):
        factors1 = tensor1.factors.copy()
        factors1[0] = factors1[0] * tensor1.weights
    elif isinstance(tensor1, list):
        factors1 = tensor1.copy()

    if isinstance(tensor2, tl.cp_tensor.CPTensor):
        factors2 = tensor2.factors.copy()
        factors2[0] = factors2[0] * tensor2.weights
    elif isinstance(tensor2, list):
        factors2 = tensor2.copy()

    d = len(factors1)
    assert d == len(factors2), "Both tensors must have the same number of modes"
    assert all(
        factors1[i].shape[0] == factors2[i].shape[0] for i in range(d)
    ), "Dimension mismatch between tensors"

    if engine == "numba":
        r1 = factors1[0].shape[1]
        r2 = factors2[0].shape[1]
        return _cp_dot_numba(factors1, factors2, d, r1, r2)
    else:
        r1 = factors1[0].shape[1]
        r2 = factors2[0].shape[1]
        result = tl.ones((r1, r2))
        for i in range(0, d):
            result *= factors1[i].T @ factors2[i]

        return tl.sum(result)


@njit()
def _cp_dot_numba(factors1, factors2, d, r1, r2):
    results = np.ones((r1, r2))
    for i in range(d):
        results *= factors1[i].T @ factors2[i]
    return np.sum(results)


def cp_dist(w1, w2):
    """
    ||w1 - w2||_F
    Args:
        w1: cp tensor 1
        w2: cp tensor 2

    Returns:
        float: Frobenius norm of the difference between the two cp tensors
    """
    distance = cp_dot(w1, w1) - 2 * cp_dot(w1, w2) + cp_dot(w2, w2)
    return np.sqrt(distance)


def cp_dot_batch(tensor1, tensor2, batch_size=128):
    """
    Batch version of cp_dot for large rank tensors.

    Args:
        tensor1: first cp tensor or list of factors
        tensor2: second cp tensor or list of factors
        batch_size: size of the batches to use for the dot product, defaults to 128.
    Returns:
        float: dot product of the two cp tensors
    """
    if isinstance(tensor1, tl.cp_tensor.CPTensor):
        factors1 = tensor1.factors.copy()
        factors1[0] = factors1[0] * tensor1.weights
    elif isinstance(tensor1, list):
        factors1 = tensor1.copy()

    if isinstance(tensor2, tl.cp_tensor.CPTensor):
        factors2 = tensor2.factors.copy()
        factors2[0] = factors2[0] * tensor2.weights
    elif isinstance(tensor2, list):
        factors2 = tensor2.copy()
    d = len(factors1)
    assert d == len(factors2), "Both tensors must have the same number of modes"

    R1 = factors1[0].shape[1]
    R2 = factors2[0].shape[1]
    result = 0
    for r in range(0, R1, batch_size):
        idx1_end = min(r + batch_size, R1)
        batch1 = [factors1[d][:, r:idx1_end].copy() for d in range(len(factors1))]
        for k in range(0, R2, batch_size):
            idx2_end = min(k + batch_size, R2)
            batch2 = [factors2[d][:, k:idx2_end].copy() for d in range(len(factors2))]
            prod = cp_dot(batch1, batch2)
            result += prod

    return result


def cp_squared_dist(w1, w2, *, engine="numpy", method="fro", batch_size=None):
    """
    ||w1 - w2||_F
    Args:
        w1: cp tensor 1
        w2: cp tensor 2

    Returns:
        float: Frobenius norm of the difference between the two cp tensors
    """
    if batch_size is None:
        distance = (
            cp_dot(w1, w1, engine=engine)
            - 2 * cp_dot(w1, w2, engine=engine)
            + cp_dot(w2, w2, engine=engine)
        )
        return distance
    else:
        distance = (
                cp_dot_batch(w1, w1, batch_size=batch_size)
                - 2*cp_dot_batch(w1, w2, batch_size=batch_size)
                + cp_dot_batch(w2, w2, batch_size=batch_size)
        )
        return distance


def cp_cos_sim(w1, w2):
    """
    Cosine similarity between two cp tensors.
    Args:
        w1: cp  tensor 1
        w2: cp tensor 2

    Returns:
        float: cosine similarity between the two cp tensors (in the range [-1, 1])
    """
    sim = cp_dot(w1, w2) / (np.sqrt(cp_dot(w1, w1)) * np.sqrt(cp_dot(w2, w2)))
    return sim


def cp_angle(w1, w2):
    """
    (principle?) Angle between two cp tensors, using cosine similarity.
    Args:
        w1:
        w2:

    Returns:

    """
    sim = cp_cos_sim(w1, w2)
    angle = np.arccos(sim) * 180 / np.pi
    return angle


def cp_norm(w):
    return np.sqrt(cp_dot(w, w))


def cp_add(tensor1, tensor2):
    """
    Add two CP tensors.
    Args:
        w1: cp tensor 1
        w2: cp tensor 2

    Returns:
        cp tensor: sum of the two cp tensors
    """
    if isinstance(tensor1, tl.cp_tensor.CPTensor):
        factors1 = tensor1.factors.copy()
        factors1[0] = factors1[0] * tensor1.weights
    elif isinstance(tensor1, list):
        factors1 = tensor1.copy()

    if isinstance(tensor2, tl.cp_tensor.CPTensor):
        if isinstance(tensor2, tl.cp_tensor.CPTensor):
            factors2 = tensor2.factors.copy()
            factors2[0] = factors2[0] * tensor2.weights
        elif isinstance(tensor2, list):
            factors2 = tensor2.copy()

    d = len(factors1)
    assert d == len(factors2), "Both tensors must have the same number of modes"
    new_factors = []
    for i in range(d):
        assert (
            factors1[i].shape[0] == factors2[i].shape[0]
        ), "Dimension mismatch between tensors"
        new_factors.append(tl.concatenate([factors1[i], factors2[i]], axis=1))

    return new_factors
