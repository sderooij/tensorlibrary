from ..linalg import dot_kron

import tensorly as tl


def get_system_cp_krr(z_x, g, y):
    """Get the objective function for CP KRR ALS sweeps

    Args:
        z_x (tl.tensor): z_x features N x M
        g (tl.tensor): feature multiplied with factors
        y (tl.tensor): target tensor / labels N x 1

    Returns:
        Tuple: (A, b) for Ax = b

    """

    N, M = z_x.shape
    _, R = g.shape
    A = tl.zeros((R*M, R*M))
    b = tl.zeros(R*M)
    batch_size = 10000
    for i in range(0, N, batch_size):
        idx_end = min(i+batch_size, N)
        temp = dot_kron(z_x[i:idx_end, :], g[i:idx_end, :])
        A += temp.T @ temp
        b += temp.T @ y[i:idx_end]

    return A, b




