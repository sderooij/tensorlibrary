import numpy as np

from ..Decompositions.TensorTrain import TensorTrain
from ..random import tt_random
from ..linalg import dot_kron, multi_dot_kron
import tensorly as tl
import tensornetwork as tn
import itertools


def tt_krr(x, y, m, ranks, reg_par, num_sweeps, kernel_type="rbf", kernel_param=1.0):
    """
    Apply the TT-KRR agorithm (Tensor-Train Kernel Ridge Regression).

    Args:
        x: data (N x D)
        y: labels (1 x N)
        m: number of basis functions / order of polynomials
        ranks: TT-ranks
        reg_par:
        kernel_param: lengthscale for rbf-kernel
        num_sweeps: number of ALS sweeps

    Returns:
        weights: weights of the TT-KRR
    """

    N, D = x.shape
    if isinstance(ranks, int):
        ranks = ranks * tl.ones(D - 1)
    shape_weights = m * tl.ones(D)  # m^D
    weights_init = tt_random(shape_weights, ranks)

    weights = tt_krr_als(
        weights_init,
        x,
        y,
        m,
        reg_par,
        num_sweeps,
        kernel_type=kernel_type,
        kernel_param=kernel_param,
    )

    return weights


def tt_krr_als(
    weights, x, y, m, reg_par, num_sweeps, kernel_type="rbf", kernel_param=1.0
):
    for ite in range(0, num_sweeps):  # forward and backward sweep
        weights = tt_krr_sweep(
            weights,
            x,
            y,
            m,
            reg_par,
            kernel_type=kernel_type,
            kernel_param=kernel_param,
        )

    return weights


def tt_krr_sweep(
    weights, x, y, m: int, reg_par: float, kernel_param: float = 1.0, kernel_type="rbf"
):
    """
    One sweep of the TT-KRR algorithm.  Forward and backward sweep.
    Args:
        weights: d-dimensional tensor train
        x: data (N x D)
        y: labels (1 x N)
        m: basis functions / order of polynomials
        reg_par: regularization parameter
        kernel_param: kernel parameter (lengthscale for rbf-kernel)
        kernel_type: rbf, poly or chebishev

    Returns:
        weights: updated weights
    """
    # TODO change to include dot_kron
    d = weights.ndims
    # sz = weights.shape
    # ranks = weights.ranks
    N = x.shape[0]
    if weights.norm_index != d - 1:
        weights.orthogonalize(d - 1, inplace=True)

    sweep = list(range(d - 1, 0, -1)) + list(range(0, d - 1))
    z_0 = features(x[:, 0], m, kernel_type=kernel_type, kernel_param=kernel_param)
    z_d = features(x[:, d - 1], m, kernel_type=kernel_type, kernel_param=kernel_param)
    g_left = [tl.ones((N, 1)), update_g(weights.cores[0].tensor, z_0, mode="first")]
    g_right = [None for iter in range(0, d - 2)]
    g_right.append(update_g(weights.cores[d - 1].tensor, z_d, mode="last"))
    g_right.append(tl.ones((N, 1)))  # g_right[k] from k+1 to d-1

    for iter in range(1, d - 1):
        z_k = features(
            x[:, iter], m, kernel_type=kernel_type, kernel_param=kernel_param
        )
        g_left.append(
            update_g(weights.cores[iter].tensor, z_k, g_left[iter], mode="left")
        )

    prev_core = d
    for iter, k_core in enumerate(sweep):
        g = []  # RRm x N matrix of features with respect to the k_core
        sz = weights.cores[k_core].shape
        lft = k_core < prev_core  # direction is left (bool)
        # for x_row in x:  # rewrite to make parallel
        #     # feature map
        #     z_x = features(
        #         x_row, m, kernel_type=kernel_type, kernel_param=kernel_param
        #     )  # D x m
        #     # contract features with weights
        #     g.append(tl.tensor_to_vec(get_g(weights, z_x, k_core).tensor))
        #
        # g = tl.stack(g, axis=1)  # RRm x N
        z_k = features(
            x[:, k_core], m, kernel_type=kernel_type, kernel_param=kernel_param
        )
        g = dot_kron(dot_kron(g_left[k_core], z_k), g_right[k_core])  # N x RRm
        gg = g.T @ g
        gy = g.T @ y
        del g
        new_weight = tl.solve(
            gg + reg_par * N * tl.eye(gg.shape[0]), gy
        )  # (G^T G + reg_par * N * I) w_k = G^T y
        # update core
        weights.update_core(k_core, tl.reshape(new_weight, weights.cores[k_core].shape))
        # shift norm to next core of the sweep
        if iter < len(sweep) - 1:
            weights.shiftnorm(sweep[iter + 1], inplace=True)
        else:
            weights.shiftnorm(sweep[0], inplace=True)

        # update g_right / g_left
        if lft and d - 1 > k_core >= 1:
            g_right[k_core - 1] = update_g(
                weights.cores[k_core].tensor, z_k, g_right[k_core], mode="right"
            )
        elif lft and k_core == d - 1:
            g_right[k_core - 1] = update_g(
                weights.cores[k_core].tensor, z_k, mode="last"
            )
        elif not lft and 1 < k_core < d - 1:
            g_left[k_core + 1] = update_g(
                weights.cores[k_core].tensor, z_k, g_left[k_core], mode="left"
            )
        elif not lft and k_core == 0:
            g_left[k_core + 1] = update_g(
                weights.cores[k_core].tensor, z_k, mode="first"
            )

        prev_core = k_core

    return weights


def features(x_d, m: int, kernel_type="rbf", *, kernel_param=1.0):
    """
    Feature mapping.
    Args:
        x_d: d-th feature N x 1 array (N=datapoints, 1=feature dimension)
        m: number of basis functions or order of polynomial
        kernel_type: kernel type rbf (via deterministic fourier), poly or chebishev
        kernel_param: parameters of the kernel function. lengthscale for rbf

    Returns:
        z_x : mapped features (D x m)
    """
    if kernel_type == "rbf":
        x_d = (x_d + 1 / 2) / 2
        w = np.arange(1, m + 1)
        s = (
            np.sqrt(2 * np.pi)
            * kernel_param
            * np.exp(-((np.pi * w / 2) ** 2) * kernel_param**2 / 2)
        )
        z_x = np.sin(np.pi * x_d[:, np.newaxis] * w) * np.sqrt(s)
    elif kernel_type == "poly":
        # polynomial feature map
        z_x = np.zeros((x_d.shape[0], m))
        # in vectorized form
        for i in range(0, m):
            z_x[:, i] = x_d**i

    elif kernel_type == "chebishev":
        # chebishev feature map
        z_x = np.zeros((x_d.shape[0], m))
        # in vectorized form
        for i in range(0, m):
            z_x[:, i] = np.cos(i * np.arccos(2 * x_d[:, 0] - 1))
    else:
        raise NotImplementedError

    return z_x


def update_g(weight_k, z_k, g_prev=None, mode="right"):
    """
    Update the G tensor with next using the dot_kron function.
    Args:
        weight_k: W^(k) tensor, k-th core of the TT weights
        z_k: feature map of k-th feature (N x m)
        g_prev: previous G tensor (N x R_k-1))
        mode: 'right' or 'left', 'left' for k-1 cores and 'right' for k+1 cores
        first: True if k=1, default False
        last: True if k=d (last core), default False

    Returns:
        g_next: updated G tensor (N x R_k)
    """
    if mode == "first" or mode == "last":
        return tl.unfold(tl.tenalg.mode_dot(weight_k, z_k, mode=1), mode=1)
    elif mode == "left":
        gz = dot_kron(g_prev, z_k)
        return gz @ tl.unfold(weight_k, mode=2).T
    elif mode == "right":
        gz = dot_kron(g_prev, z_k)
        return gz @ tl.unfold(weight_k, mode=0).T


def get_g(weights, z_x, k_d):
    """
    Contract the TT with the features z_x without the k_d-th core.
    Args:
        weights: TT weights of the TT-KRR
        z_x: mapped features (D x m)
        k_d: index of the core to be left out

    Returns:
        g: tensor of shape (R_d-1 x m x R_d+1)
    """

    D = len(weights)
    assert len(weights) == len(z_x)
    w_rest = tn.replicate_nodes(weights.cores[:k_d] + weights.cores[k_d + 1 :])
    z_nodes = [tn.Node(z_x[k, :, None, None]) for k in range(0, D)]
    z_connect = [
        z_nodes[k].edges[2] ^ z_nodes[k + 1].edges[1] for k in range(-1, D - 1)
    ]

    connections = [
        w_rest[k_w].edges[1] ^ z_nodes[k_z].edges[0]
        for k_w, k_z in zip(
            range(0, D - 1), itertools.chain(range(0, k_d), range(k_d + 1, D))
        )
    ]

    if k_d != D - 1:
        output_edge_order = [
            w_rest[k_d - 1].edges[2],
            z_nodes[k_d].edges[0],
            w_rest[k_d].edges[0],
        ]
    elif k_d == D - 1:
        output_edge_order = [
            w_rest[k_d - 1].edges[2],
            z_nodes[k_d].edges[0],
            w_rest[0].edges[0],
        ]
    return tn.contractors.auto(w_rest + z_nodes, output_edge_order=output_edge_order)


def tt_krr_predict(
    weights, new_sample, m: int, reg_par, kernel_type="rbf", *, kernel_param=1.0
):
    """
    Predict the label of a new sample. The new sample is mapped to the feature space and then contracted with the TT weights.
    Args:
        weights: of the TT-KRR
        new_sample: new datapoint to be classified
        m: number of basis functions or order of polynomial
        reg_par: regularization parameter
        kernel_type: kernel type, rbf (via deterministic fourier), poly or chebishev
        kernel_param: kernel parameter, lengthscale for rbf

    Returns:
        labels: predicted labels
        out: output of the TT-KRR
    """
    [N_test, D] = new_sample.shape
    if N_test == 1:
        m = weights.cores[0].shape[1]
        z = features(new_sample, m, kernel_type=kernel_type, kernel_param=kernel_param)
        z = [tl.reshape(z[k], (1, m, 1)) for k in range(len(z))]
        ztt = TensorTrain(cores=z)
        out = weights.dot(ztt) + reg_par * weights.norm() ** 2
        labels = np.sign(out)
    elif N_test < 1e5:
        # Z = multi_dot_kron([features(new_sample[:, k], m, kernel_type=kernel_type, kernel_param=kernel_param)
        #                     for k in range(0, D)])
        wz = tl.ones((N_test, 1))
        z_k = features(
            new_sample[:, 0], m, kernel_type=kernel_type, kernel_param=kernel_param
        )
        wz = update_g(weights.cores[0].tensor, z_k, mode="first")
        for k in range(1, D):
            z_k = features(
                new_sample[:, k], m, kernel_type=kernel_type, kernel_param=kernel_param
            )
            wz = update_g(weights.cores[k].tensor, z_k, g_prev=wz, mode="left")
        wz = tl.tensor_to_vec(wz)
        labels = np.sign(wz)

    return labels, wz