from ..linalg import dot_kron, dot_kron_numba

import tensorly as tl


def get_system_cp_krr(z_x, g, y, *, numba=True, batch_size=8192):
    """Get the objective function for CP KRR ALS sweeps

    Args:
        z_x (tl.tensor): z_x features N x M
        g (tl.tensor): feature multiplied with factors
        y (tl.tensor): target tensor / labels N x 1
        numba (bool, optional): Use numba for faster computation. Only for numpy backend. Defaults to True.
        batch_size (int, optional): Batch size for computation. Defaults to 10000.

    Returns:
        Tuple: (A, b) for Ax = b

    """
    if numba:
        dotkron = dot_kron_numba
    else:
        dotkron = dot_kron

    N, M = z_x.shape
    _, R = g.shape
    A = tl.zeros((R * M, R * M))
    b = tl.zeros(R * M)
    for i in range(0, N, batch_size):
        idx_end = min(i + batch_size, N)
        temp = dotkron(z_x[i:idx_end, :], g[i:idx_end, :])
        A += temp.T @ temp
        b += temp.T @ y[i:idx_end]

    return A, b


def get_system_cp_LMPROJ(z_x, p, gamma, *, numba=True, batch_size=8192):
    """Get the objective function for CP KRR ALS sweeps

    Args:
        z_x (tl.tensor): z_x features N x M, source and target concatenated
        p (tl.tensor): feature multiplied with factors
        gamma (tl.tensor): (1/Ns) or (-1/Nt) for source and target respectively
        numba (bool, optional): Use numba for faster computation. Only for numpy backend. Defaults to True.
        batch_size (int, optional): Batch size for computation. Defaults to 8192.

    Returns:
        Tuple: (A, b) for Ax = b

    """
    if numba:
        dotkron = dot_kron_numba
    else:
        dotkron = dot_kron

    N, M = z_x.shape
    _, R = p.shape
    PP = tl.zeros((R * M, R * M))

    for i in range(0, N, batch_size):
        idx_end = min(i + batch_size, N)
        gamma_z_x = gamma[i:idx_end] * z_x[i:idx_end, :]
        P = dotkron(gamma_z_x, p[i:idx_end, :])
        PP += P.T @ P

    return PP


# below is old code
def init_k(W, V):

    # WV^T
    assert len(W) == len(V)
    WV = tl.ones((W[0].shape[1], V[0].shape[1]))
    for d in range(1, len(W)):
        WV *= W[d] @ V[d].T

    k = V[0] @ WV.T
    return k.reshape((-1, 1), order="F")  # flatten M x R_w


def CPKM_predict(x, w, features):
    """
    Prediction function for CPD based kernel machines.
    Args:
        x: input data/features, shape (N, D)
        w: CPD weights tensor, list of length D containing (M x R) arrays
        features: the feature map function, z_x = features(x)

    Returns:
        (N,1) array with outputs
    """
    N, D = x.shape
    y_pred = tl.ones((N, 1))
    for d in range(0, D):
        z_x = features(x[:, d])
        y_pred = y_pred * (z_x @ w[d])
    y_pred = tl.sum(y_pred, axis=1)
    return y_pred


def CPKM_predict_batchwise(x, w, features, batch_size=8192):
    """
    Prediction function for CPD based kernel machines in a batch-wise manner.
    Args:
        x: input data/features, shape (N, D)
        w: CPD weights tensor, list of length D containing (M x R) arrays
        features: the feature map function, z_x = features(x)
        batch_size: batch size for prediction

    Returns:
        (N,1) array with outputs
    """
    N, D = x.shape
    y_pred = tl.zeros((N, 1))
    for i in range(0, N, batch_size):
        idx_end = min(i + batch_size, N)
        y_pred[i:idx_end] = CPKM_predict(x[i:idx_end, :], w, features)

    return y_pred
