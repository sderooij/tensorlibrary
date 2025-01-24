# TODO check that the reshapes are correct (C row major instead of column major)
import numpy as np

from ..decompositions.TensorTrain import TensorTrain
from ..random import tt_random
from ..linalg import dot_kron, multi_dot_kron
from .features import features
import tensorly as tl
import tensornetwork as tn
import itertools


def tt_krr(x, y, m, ranks, reg_par, num_sweeps, feature_map="rbf", map_param=1.0):
    """
    Apply the TT-KRR agorithm (Tensor-Train Kernel Ridge Regression).

    Args:
        x: data (N x D)
        y: labels (1 x N)
        m: number of basis functions / order of polynomials
        ranks: TT-ranks
        reg_par:
        map_param: lengthscale for rbf-kernel
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
        feature_map=feature_map,
        map_param=map_param,
    )

    return weights


def tt_krr_als(weights, x, y, m, reg_par, num_sweeps, feature_map="rbf", map_param=1.0):
    for ite in range(0, num_sweeps):  # forward and backward _sweep
        weights = tt_krr_sweep(
            weights, x, y, m, reg_par, feature_map=feature_map, map_param=map_param,
        )

    return weights


def tt_krr_sweep(
    weights, x, y, m: int, reg_par: float, map_param: float = 1.0, feature_map="rbf"
):
    """
    One _sweep of the TT-KRR algorithm.  Forward and backward _sweep.
    Args:
        weights: d-dimensional tensor train
        x: data (N x D)
        y: labels (1 x N)
        m: basis functions / order of polynomials
        reg_par: regularization parameter
        map_param: kernel parameter (lengthscale for rbf-kernel)
        feature_map: rbf, poly or chebishev

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
    z_0 = features(x[:, 0], m, feature_map=feature_map, map_param=map_param)
    z_d = features(x[:, d - 1], m, feature_map=feature_map, map_param=map_param)
    g_left = [tl.ones((N, 1)), update_wz_tt(weights.cores[0].tensor, z_0, mode="first")]
    g_right = [None for iter in range(0, d - 2)]
    g_right.append(update_wz_tt(weights.cores[d - 1].tensor, z_d, mode="last"))
    g_right.append(tl.ones((N, 1)))  # g_right[k] from k+1 to d-1

    for iter in range(1, d - 1):
        z_k = features(x[:, iter], m, feature_map=feature_map, map_param=map_param)
        g_left.append(
            update_wz_tt(weights.cores[iter].tensor, z_k, g_left[iter], mode="left")
        )

    prev_core = d
    for iter, k_core in enumerate(sweep):
        g = []  # RRm x N matrix of features with respect to the k_core
        sz = weights.cores[k_core].shape
        lft = k_core < prev_core  # direction is left (bool)
        # for x_row in x:  # rewrite to make parallel
        #     # feature map
        #     z_x = features(
        #         x_row, m, feature_map=feature_map, map_param=map_param
        #     )  # D x m
        #     # contract features with weights
        #     g.append(tl.tensor_to_vec(get_g(weights, z_x, k_core).tensor))
        #
        # g = tl.stack(g, axis=1)  # RRm x N
        z_k = features(x[:, k_core], m, feature_map=feature_map, map_param=map_param)
        g = dot_kron(dot_kron(g_left[k_core], z_k), g_right[k_core])  # N x RRm
        gg = g.T @ g
        gy = g.T @ y
        del g
        new_weight = tl.solve(
            gg + reg_par * N * tl.eye(gg.shape[0]), gy
        )  # (G^T G + reg_par * N * I) w_k = G^T y
        # update core
        sz = [s for s in reversed(weights.cores[k_core].shape)]
        weights.update_core(k_core, tl.reshape(new_weight, tuple(sz)).T)
        # shift norm to next core of the _sweep
        if iter < len(sweep) - 1:
            weights.shiftnorm(sweep[iter + 1], inplace=True)
        else:
            weights.shiftnorm(sweep[0], inplace=True)

        # update g_right / g_left
        if lft and d - 1 > k_core >= 1:
            g_right[k_core - 1] = update_wz_tt(
                weights.cores[k_core].tensor, z_k, g_right[k_core], mode="right"
            )
        elif lft and k_core == d - 1:
            g_right[k_core - 1] = update_wz_tt(
                weights.cores[k_core].tensor, z_k, mode="last"
            )
        elif not lft and 1 < k_core < d - 1:
            g_left[k_core + 1] = update_wz_tt(
                weights.cores[k_core].tensor, z_k, g_left[k_core], mode="left"
            )
        elif not lft and k_core == 0:
            g_left[k_core + 1] = update_wz_tt(
                weights.cores[k_core].tensor, z_k, mode="first"
            )

        prev_core = k_core

    return weights


def update_wz_tt(weight_k, z_k, g_prev=[], mode="right"):
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
        gz = dot_kron(z_k, g_prev)  # different order than matlab
        return gz @ tl.reshape(weight_k, (-1, weight_k.shape[2]))
    elif mode == "right":
        gz = dot_kron(g_prev, z_k)
        return gz @ tl.unfold(weight_k, mode=0).T


def initialize_wz(weights, x, M, feature_map, map_param, k_core):

    N, D = x.shape
    wz_right = [None for iter in range(0, D)]
    wz_left = [None for iter in range(0, D)]

    if k_core == D - 1:
        wz_right[-1] = tl.ones((N, 1))
        wz_left[0] = tl.ones((N, 1))
        for iter, core in enumerate(weights):
            z_x = features(
                x[:, iter + 1], M, feature_map=feature_map, map_param=map_param
            )
            if iter == 0:
                wz_left[iter + 1] = update_wz_tt(core, z_x, mode="first")
            elif iter == D - 1:
                break
            else:
                wz_left[iter + 1] = update_wz_tt(core, z_x, wz_left[iter], mode="left")

    elif k_core == 0:
        wz_left[0] = tl.ones((N, 1))
        wz_right[-1] = tl.ones((N, 1))
        # iter = D-1
        for iter, core in enumerate(reversed(weights)):
            if iter == D - 1:
                break
            ii = -(iter + 2)
            z_x = features(
                x[:, ii + 1], M, feature_map=feature_map, map_param=map_param
            )
            if ii == -2:
                wz_right[ii] = update_wz_tt(core, z_x, mode="last")
            else:
                wz_right[ii] = update_wz_tt(core, z_x, wz_right[ii + 1], mode="right")
    else:
        raise ValueError("k_core must be either 0 or D-1 for initialization.")
    return wz_left, wz_right


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
    weights, new_sample, m: int, reg_par, feature_map="rbf", *, map_param=1.0
):
    """
    Predict the label of a new sample. The new sample is mapped to the feature space and then contracted with the TT weights.
    Args:
        weights: of the TT-KRR
        new_sample: new datapoint to be classified
        m: number of basis functions or order of polynomial
        reg_par: regularization parameter
        feature_map: kernel type, rbf (via deterministic fourier), poly or chebishev
        map_param: kernel parameter, lengthscale for rbf

    Returns:
        labels: predicted labels
        out: output of the TT-KRR
    """
    [N_test, D] = new_sample.shape
    if N_test == 1:
        m = weights.cores[0].shape[1]
        z = features(new_sample, m, feature_map=feature_map, map_param=map_param)
        z = [tl.reshape(z[k], (1, m, 1)) for k in range(len(z))]
        ztt = TensorTrain(cores=z)
        out = weights.dot(ztt) + reg_par * weights.norm() ** 2
        labels = np.sign(out)
    elif N_test < 1e5:
        # Z = multi_dot_kron([features(new_sample[:, k], m, feature_map=feature_map, map_param=map_param)
        #                     for k in range(0, D)])
        wz = tl.ones((N_test, 1))
        z_k = features(
            new_sample[:, 0], m, feature_map=feature_map, map_param=map_param
        )
        wz = update_wz_tt(weights.cores[0].tensor, z_k, mode="first")
        for k in range(1, D):
            z_k = features(
                new_sample[:, k], m, feature_map=feature_map, map_param=map_param
            )
            wz = update_wz_tt(weights.cores[k].tensor, z_k, g_prev=wz, mode="left")
        wz = tl.tensor_to_vec(wz)
        labels = np.sign(wz)

    return labels, wz


def get_tt_rank(shape, max_rank):
    out = [1]
    for k in range(0, len(shape) - 1):
        left = tl.prod(shape[: k + 1], dtype=float)
        right = tl.prod(shape[k + 1 :], dtype=float)
        temp = tl.min([left, right, max_rank])
        out.append(int(temp))
    out.append(1)
    return out
