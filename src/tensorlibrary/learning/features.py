import tensorly as tl
from ..linalg.linalg import cp_squared_dist, cp_dot


def features(
    x_d, m: int, feature_map="rbf", *, map_param=1.0, Ld=1.0, old_version=False
):
    """
    Feature mapping.
    Args:
        x_d: d-th feature N x 1 array (N=datapoints, 1=feature dimension)
        m: number of basis functions or order of polynomial
        feature_map: kernel type rbf (via deterministic fourier), poly or chebishev
        map_param: al_parameters of the kernel function. lengthscale for rbf
        Ld: domain of the input data (default 1) for rbf [-Ld, Ld]

    Returns:
        z_x : mapped features (N x m)
    """
    if feature_map == "rbf" or feature_map == "fourier":
        if old_version:
            x_d = (x_d + 0.5) * 0.5
            w = tl.arange(1, m + 1)
            s = (
                tl.sqrt(2 * tl.pi)
                * map_param  # lengthscale
                * tl.exp(-((tl.pi * w.T / 2) ** 2) * map_param ** 2 / 2)
            )
            z_x = tl.sin(tl.pi * tl.tenalg.outer([x_d, w])) * tl.sqrt(s)
        else:
            z_x = fourier_features(x_d, m, map_param=map_param, Ld=Ld)
    elif feature_map == "poly":
        # polynomial feature map
        z_x = tl.zeros((x_d.shape[0], m))
        # in vectorized form
        for i in range(0, m):
            z_x[:, i] = x_d ** (i)

    elif feature_map == "pure_power":
        z_x = pure_power_features(x_d, m)

    # elif feature_map == "chebyshev":
    #     # chebyshev feature map
    #     z_x = tl.zeros((x_d.shape[0], m))
    #     # z_x[:, 0] = 1
    #     z_x[:, 0] = x_d
    #     for i in range(1, m):
    #         z_x[:, i] = 2 * x_d * z_x[:, i - 1] - z_x[:, i - 2]
    # elif feature_map == "chebyshev2":
    #     # chebyshev feature map of the second kind
    #     z_x = tl.zeros((x_d.shape[0], m))
    #     # z_x[:, 1] = 1
    #     z_x[:, 0] = 2 * x_d
    #     for i in range(1, m):
    #         z_x[:, i] = 2 * x_d * z_x[:, i - 1] - z_x[:, i - 2]

    else:
        raise NotImplementedError

    return z_x


def fourier_features(x_d, m: int, map_param=1.0, Ld=1.0):
    x_d = (x_d + Ld) / (2 * Ld)
    w = tl.arange(1, m + 1)
    s = (
        tl.sqrt(2 * tl.pi)
        * map_param
        * tl.exp(-(((tl.pi * w.T) / (2 * Ld)) ** 2) * map_param ** 2 / 2)
    )
    z_x = (1 / tl.sqrt(Ld)) * tl.sin(tl.pi * tl.tenalg.outer([x_d, w])) * tl.sqrt(s)
    return z_x


def pure_power_features(x_d, m):
    """
    Pure-Power Polynomial Features

    Parameters:
    X : numpy.ndarray
        Input array of shape (n_samples, n_features).
    M : int
        Maximum power (degree) of the polynomial features.

    Returns:
    Mati : numpy.ndarray
        Array of shape (n_features, n_samples, M) containing unit-norm pure-power features.
    """
    # Compute the pure-power features
    # polynomial feature map
    z_x = tl.zeros((x_d.shape[0], m))
    # in vectorized form
    for i in range(0, m):
        z_x[:, i] = x_d ** (i)

    # Normalize each sample's features along the last axis to have a unit norm
    norms = tl.norm(
        z_x, axis=1
    )  # Compute norms along the power axis
    norms = tl.reshape(norms, (-1, 1))  # Reshape norms to match the feature shape
    Mati = z_x / norms  # Normalize features to unit norm

    return Mati + 0.2


def kernel_mat_features(x, y, m=10, feature_map="rbf", *, map_param=1.0, Ld=1.0):
    """
    Compute the kernel matrix for the tensor product features. Phi^T Phi
    Args:
        x: matrix of size N_x x D, containing N_x samples
        y: matrix of size N_y x D, containing N_y samples (can be the same as x, for the kernel matrix)
        m: number of frequency or basis functions for the feature map, degree of polynomial. Default 10
        feature_map: kernel type rbf (via deterministic fourier), poly. Default rbf
        map_param: al_parameters of the kernel function. lengthscale for rbf, none for poly. Default 1.
        Ld: domain of the input data (default 1) for rbf [-Ld, Ld]. Default 1.

    Returns:
        K: kernel matrix of size N_x x N_y
    """
    assert x.shape[1] == y.shape[1], "x and y must have the same number of features"
    D = x.shape[1]
    N_x = x.shape[0]
    N_y = y.shape[0]
    K = tl.ones((N_x, N_y))  # initialize "kernel" matrix Phi^T Phi

    for d in range(D):
        phi_x = features(
            x[:, d], m=m, feature_map=feature_map, map_param=map_param, Ld=Ld
        )
        phi_y = features(
            y[:, d], m=m, feature_map=feature_map, map_param=map_param, Ld=Ld
        )
        K *= phi_x @ phi_y.T

    return K


def MMD(
    x_source,
    x_target,
    feature_fun_source,
    feature_fun_target,
    *,
    y_source=None,
    y_target=None,
    engine="numpy",
    batch_size=128,
):
    """
    Compute the Maximum Mean Discrepancy (MMD) between the source and target distributions using the tensor product features.
    Args:
        x_source: source data N_source x D
        x_target: target data N_target x D
        feature_fun_source: feature map function for the source data
        feature_fun_target: feature map function for the target data
        y_source: (Optional) source labels N_source x 1, used for separate MMD computation for different classes.
                Default None.
        y_target: (Optional) target labels N_target x 1
        engine: (Optional) computation engine, numba or numpy. Default numba.

    Returns:

    """

    if y_source is None:
        return _compute_mmd(
            x_source, x_target, feature_fun_source, feature_fun_target, engine=engine, batch_size=batch_size
        )
    else:
        idx_pos_source = y_source == 1
        idx_neg_source = y_source == -1
        idx_pos_target = y_target == 1
        idx_neg_target = y_target == -1
        mmd_pos = _compute_mmd(
            x_source[idx_pos_source,:],
            x_target[idx_pos_target,:],
            feature_fun_source,
            feature_fun_target,
            engine=engine,
            batch_size=batch_size,
        )
        mmd_neg = _compute_mmd(
            x_source[idx_neg_source,:],
            x_target[idx_neg_target,:],
            feature_fun_source,
            feature_fun_target,
            engine=engine,
            batch_size=batch_size,
        )
        return mmd_pos, mmd_neg


def _compute_mmd(x_source, x_target, feature_fun_source, feature_fun_target, engine, batch_size):
    N_source, D = x_source.shape
    N_target, Dt = x_target.shape
    assert D == Dt, "x_source and x_target must have the same number of features"

    z_source = []
    z_target = []
    for d in range(D):
        z_source.append(feature_fun_source(x_source[:, d]).T)
        z_target.append(feature_fun_target(x_target[:, d]).T)

    loadings_source = (1 / N_source) * tl.ones((1, N_source))
    loadings_target = (1 / N_target) * tl.ones((1, N_target))
    z_source[-1] = z_source[-1] * loadings_source
    z_target[-1] = z_target[-1] * loadings_target
    mmd = cp_squared_dist(z_source, z_target, batch_size=batch_size, engine=engine)
    return mmd


def MMD_kernel(x_source, x_target, kernel, *, y_source=None, y_target=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between the source and target distributions using the kernel function.
    Args:
        x_source: source data N_source x D
        x_target: target data N_target x D
        kernel: kernel function K(Xs, Xt) --> N_source x N_target, function must be symmetric and vectorized
        y_source: (Optional) source labels N_source x 1, used for separate MMD computation for different classes.
                Default None.
        y_target: (Optional) target labels N_target x 1

    Returns:

    """
    if y_source is None:
        return _compute_mmd_kernel(x_source, x_target, kernel)
    else:
        idx_pos_source = y_source == 1
        idx_neg_source = y_source == -1
        idx_pos_target = y_target == 1
        idx_neg_target = y_target == -1
        mmd_pos = _compute_mmd_kernel(
            x_source[idx_pos_source,:], x_target[idx_pos_target,:], kernel
        )
        mmd_neg = _compute_mmd_kernel(
            x_source[idx_neg_source,:], x_target[idx_neg_target,:], kernel
        )
        return mmd_pos, mmd_neg


def _compute_mmd_kernel(x_source, x_target, kernel):
    N_source, D = x_source.shape
    N_target, Dt = x_target.shape
    assert D == Dt, "x_source and x_target must have the same number of features"

    K_ss = kernel(x_source, x_source)
    K_tt = kernel(x_target, x_target)
    K_st = kernel(x_source, x_target)

    mmd = tl.sum(K_ss) / (N_source * N_source) + tl.sum(K_tt) / (N_target * N_target) - 2 * tl.sum(K_st) / (N_source * N_target)
    return mmd


