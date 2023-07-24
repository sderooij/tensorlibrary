import tensorly as tl


def features(x_d, m: int, feature_map="rbf", *, map_param=1.0):
    """
    Feature mapping.
    Args:
        x_d: d-th feature N x 1 array (N=datapoints, 1=feature dimension)
        m: number of basis functions or order of polynomial
        feature_map: kernel type rbf (via deterministic fourier), poly or chebishev
        map_param: parameters of the kernel function. lengthscale for rbf

    Returns:
        z_x : mapped features (D x m)
    """
    if feature_map == "rbf" or "fourier":
        x_d = (x_d + 0.5) * 0.5
        w = tl.arange(1, m + 1)
        s = (
            tl.sqrt(2 * tl.pi)
            * map_param  # lengthscale
            * tl.exp(-((tl.pi * w.T / 2) ** 2) * map_param**2 / 2)
        )
        z_x = tl.sin(tl.pi * tl.tenalg.outer([x_d, w])) * tl.sqrt(s)
    elif feature_map == "poly":
        # polynomial feature map
        z_x = tl.zeros((x_d.shape[0], m))
        # in vectorized form
        for i in range(0, m):
            z_x[:, i] = x_d ** (i)

    elif feature_map == "chebyshev":
        # chebyshev feature map
        z_x = tl.zeros((x_d.shape[0], m))
        # z_x[:, 0] = 1
        z_x[:, 0] = x_d
        for i in range(1, m):
            z_x[:, i] = 2 * x_d * z_x[:, i - 1] - z_x[:, i - 2]
    elif feature_map == "chebyshev2":
        # chebyshev feature map of the second kind
        z_x = tl.zeros((x_d.shape[0], m))
        # z_x[:, 1] = 1
        z_x[:, 0] = 2 * x_d
        for i in range(1, m):
            z_x[:, i] = 2 * x_d * z_x[:, i - 1] - z_x[:, i - 2]

    else:
        raise NotImplementedError

    return z_x