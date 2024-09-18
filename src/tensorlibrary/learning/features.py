import tensorly as tl


def features(x_d, m: int, feature_map="rbf", *, map_param=1.0, Ld=1.0, old_version=False):
    """
    Feature mapping.
    Args:
        x_d: d-th feature N x 1 array (N=datapoints, 1=feature dimension)
        m: number of basis functions or order of polynomial
        feature_map: kernel type rbf (via deterministic fourier), poly or chebishev
        map_param: parameters of the kernel function. lengthscale for rbf
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
                * tl.exp(-((tl.pi * w.T / 2) ** 2) * map_param**2 / 2)
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
    x_d = (x_d + Ld) / (2*Ld)
    w = tl.arange(1, m + 1)
    s = (
        tl.sqrt(2 * tl.pi)
        * map_param
        * tl.exp(-((tl.pi * w.T)/(2*Ld))**2 * map_param**2 / 2)
    )
    z_x = (1/tl.sqrt(Ld))*tl.sin(tl.pi * tl.outer(x_d, w)) * tl.sqrt(s)
    return z_x


def kernel_mat_features(x, y, m=10, feature_map='rbf', *, map_param=1., Ld=1.):
    """
    Compute the kernel matrix for the tensor product features. Phi^T Phi
    Args:
        x: matrix of size N_x x D, containing N_x samples
        y: matrix of size N_y x D, containing N_y samples (can be the same as x, for the kernel matrix)
        m: number of frequency or basis functions for the feature map, degree of polynomial. Default 10
        feature_map: kernel type rbf (via deterministic fourier), poly. Default rbf
        map_param: parameters of the kernel function. lengthscale for rbf, none for poly. Default 1.
        Ld: domain of the input data (default 1) for rbf [-Ld, Ld]. Default 1.

    Returns:
        K: kernel matrix of size N_x x N_y
    """
    assert x.shape[1] == y.shape[1], "x and y must have the same number of features"
    D = x.shape[1]
    N_x = x.shape[0]
    N_y = y.shape[0]
    K = tl.ones((N_x, N_y)) # initialize "kernel" matrix Phi^T Phi

    for d in range(D):
        phi_x = features(x[:, d], m=m, feature_map=feature_map, map_param=map_param, Ld=Ld)
        phi_y = features(y[:, d], m=m, feature_map=feature_map, map_param=map_param, Ld=Ld)
        K *= (phi_x @ phi_y.T)

    return K
