import numpy as np

from ..Decompositions.TensorTrain import TensorTrain
from ..random import tt_random
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
        ranks = ranks * tl.ones((1, D - 1))
    shape_weights = m * tl.ones((1, D))  # m^D
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
    # TODO: implement the TT_ALS procedure
    d = weights.ndims
    sz = weights.shape
    if weights.norm_index != d - 1:
        weights.orthogonalize(d - 1, inplace=True)

    sweep = list(range(d - 1, 0, -1)) + list(range(0, d - 2))

    for n_sweep in sweep:
        for k_core in range(0, len(weights.cores)):
            g = []
            for x_row, y_row, n in enumerate(zip(x, y)):
                # feature map
                z_x = features(
                    x_row, m, kernel_type=kernel_type, kernel_param=kernel_param
                )  # D x m
                # contract features with weights
                g.append(tl.tensor_to_vec(get_g(weights, z_x, k_core).tensor))

            g = tl.stack(g, axis=1)     # RRm x N
            gg = g @ g.T
            gy = g @ y.T
            new_weight = tl.solve(gg, gy)
            weights.update_core(k_core, tl.reshape(new_weight, weights.cores[k_core].shape))


    return weights


def features(x, m: int, kernel_type="rbf", *, kernel_param=1.0):
    """
    Feature mapping.
    Args:
        x: D x 1 array (1=datapoints, D=features)
        m: number of basis functions or order of polynomial
        kernel_type: kernel type rbf (via deterministic fourier), poly or chebishev
        kernel_param: parameters of the kernel function. lengthscale for rbf

    Returns:
        z_x : mapped features (D x m)
    """
    if kernel_type == "rbf":
        x = (x + 1 / 2) / 2
        w = np.arange(1, m + 1)
        s = (
            np.sqrt(2 * np.pi)
            * kernel_param
            * np.exp(-((np.pi * w / 2) ** 2) * kernel_param**2 / 2)
        )
        z_x = np.sin(np.pi * x[:, np.newaxis] * w) * np.sqrt(s)
    elif kernel_type == "poly":
        z_x = x
    elif kernel_type == "chebishev":
        z_x = x

    return z_x


def get_g(weights, z_x, k_d):

    D = len(weights)
    assert len(weights) == len(z_x)
    w_rest = tn.replicate_nodes(weights.cores[:k_d] + weights.cores[k_d + 1:])
    z_nodes = [tn.Node(z_x[k, :, None]) for k in range(0, D)]
    z_connect = [z_nodes[k].edges[1] ^ z_nodes[k].edges[1] for k in range(0, D - 1)]

    connections = [
        w_rest[k].edges[1] ^ z_nodes[k].edges[0]
        for k in itertools.chain(range(0, k_d), range(k_d + 1, D))
    ]

    if k_d != D:
        output_edge_order = [
            w_rest[k_d-1].edges[2],
            z_nodes[k_d].edges[0],
            w_rest[k_d+1].edges[0]
        ]
    elif k_d == D:
        output_edge_order = [
            w_rest[k_d - 1].edges[2],
            z_nodes[k_d].edges[0],
            w_rest[0].edges[0]
        ]
    return tn.contractors.auto(w_rest + z_nodes, output_edge_order=output_edge_order)
        # connections = [
        #         w_rest[k].edges[2] ^ w_rest[k + 1].edges[0]
        #         for k in range(0, k_d)
        #     ] + [
        #     w_rest[k].edges[2] ^ w_rest[k + 1].edges[0]
        #         for k in range(k_d-1, weights.ndims-1)
        #     ] + [
        #     w_rest[-1].edges[2] ^ w_rest[0].edges[0]
        # ]

    # return
