"""
Function for CP kernel machines TODO: MUCH
"""

import tensorly as tl
from scipy.optimize import minimize
from copy import deepcopy


def init_CP(w_init, M, D, R, *, random_state=None):
    if w_init is None:
        temp = tl.random.random_cp(
            shape=tuple(int(M) * tl.ones(D, dtype=int)),
            rank=R,
            full=False,
            orthogonal=False,
            random_state=random_state,
            normalise_factors=True,
        )
        w_init = temp.factors
        w = w_init
    elif isinstance(w_init, list):
        w = deepcopy(w_init)
        # for d in range(D):
        #     w[d] /= tl.norm(w[d], order=2, axis=0)
    elif isinstance(w_init, tl.cp_tensor.CPTensor):
        w = w_init.factors
    else:
        raise ValueError("w_init must be a CPTensor or a list of factors")

    return w


def _init_model_params_step(d, x, w, feature_fun, reg, G):
    """
    Initialization step for  model al_parameters for CP Kernel Machines. This function initializes the mapped features and
    the regularizer and the G matrix which is the product of the mapped features and the weights and is used in
    the ALS sweeps. Format to be compatible with jax.

    Args:
        d: step number
        x:
        w:
        feature_fun:
        reg:
        G:

    Returns:

    """
    z_x = feature_fun(x[:, d])
    reg *= w[d].T @ w[d]
    G *= z_x @ w[d]
    return reg, G


def _init_model_params(x, w, feature_fun, *, balanced=False, y=None):
    """
    Initialize model al_parameters for CP Kernel Machines. This function initializes the mapped features and the
    regularizer and the G matrix which is the product of the mapped features and the weights and is used in the
    ALS sweeps.

    Args:
        x: input data/features, shape (N, D)
        w: CPD weights tensor, list of length D containing (M x R) arrays
        feature_fun: the feature map function, z_x = features(x)

    Returns:
        Tuple: z_x, reg, G
    """
    N, D = x.shape
    M, R = w[0].shape
    if balanced:
        idx_p = tl.where(y == 1)[0]
        idx_n = tl.where(y == -1)[0]
        Np = tl.sum(y == 1)
        Nn = tl.sum(y == -1)
        Cp = N / (2 * Np)
        Cn = N / (2 * Nn)

        reg = tl.ones((R, R))
        Gn = tl.ones((Nn, R))
        Gp = tl.ones((Np, R))
        for d in range(D - 1, -1, -1):  # D-1:-1:0
            w_d = w[d]
            reg *= w_d.T @ w_d
            z_x_n = feature_fun(x[idx_n, d])
            z_x_p = feature_fun(x[idx_p, d])
            Gn *= z_x_n @ w_d
            Gp *= z_x_p @ w_d

        return reg, Gn, Gp, Cn, Cp, idx_n, idx_p
    else:
        reg = tl.ones((R, R))
        G = tl.ones((x.shape[0], R))

        for d in range(D - 1, -1, -1):
            (reg, G) = _init_model_params_step(d, x, w, feature_fun, reg, G)

        return (reg, G)


# def _solve_CPKM_step(A, b, reg_mat, solver,  loss):
#     if loss == 'l2':
#         w_d = tl.solve(A + reg_mat, b)
#     elif loss == 'squared_hinge':
#         print("Not implemented yet")
#
#     return w_d


def _fit_CP_KM_step(
    it, x, y, reg_par, solver, feature_fun, penalty, compute_objective, model_parameters
):

    w, reg, G, loadings = model_parameters
    M, R = w[0].shape
    D = len(w)
    d = it % D

    z_x = feature_fun(x[:, d])

    reg /= w[d].T @ w[d]  # remove current factor
    G /= z_x @ w[d]  # remove current factor

    A, b = compute_objective(z_x, G, y)
    if penalty == "l2":
        reg_mat = reg + reg_par * tl.kron(reg, tl.eye(M))
    else:
        raise NotImplementedError

    w_d = solver(A, b, reg_mat)

    w[d] = tl.reshape(w_d, (M, R), order="F")

    loadings = tl.norm(w[d], order=2, axis=0)
    w[d] /= loadings
    reg *= w[d].T @ w[d]  # add current factor
    G *= z_x @ w[d]  # add current factor

    return w, reg, G, loadings


def _fit_CP_KM_balanced_step(
    it, x, y, reg_par, solver, feature_fun, penalty, compute_objective, model_parameters
):

    w, reg, G, loadings = model_parameters
    M, R = w[0].shape
    D = len(w)
    d = it % D

    z_x = feature_fun(x[:, d])

    reg /= w[d].T @ w[d]  # remove current factor
    G /= z_x @ w[d]  # remove current factor

    A, b = compute_objective(z_x, G, y)
    if penalty == "l2":
        reg_mat = reg + reg_par * tl.kron(reg, tl.eye(M))
    else:
        raise NotImplementedError

    w_d = solver(A, b, reg_mat)

    w[d] = tl.reshape(w_d, (M, R), order="F")

    loadings = tl.norm(w[d], order=2, axis=0)
    w[d] /= loadings
    reg *= w[d].T @ w[d]  # add current factor
    G *= z_x @ w[d]  # add current factor

    return w, reg, G, loadings


def _square_hinge_loss(w, G, y, reg_mat):

    margins = 1 - y * (G @ w)
    hinge_loss = (margins > 0) * margins
    squared_hinge_loss = tl.dot(hinge_loss, hinge_loss)
    regularization = w.T @ reg_mat @ w
    return squared_hinge_loss + regularization


def _square_hinge_grad(w, G, y, reg_mat):

    margins = 1 - y * (G @ w)
    hinge_loss = (margins > 0) * margins
    grad_loss = -2 * G.T @ (y * hinge_loss)
    grad_reg = 2 * reg_mat @ w
    return grad_loss + grad_reg


def _solve_TSVM_square_hinge(A, b, reg_mat, w_init=None):
    # TODO: implement with slack variables
    if w_init is None:
        w_init = tl.zeros(A.shape[1])
    w = minimize(
        _square_hinge_loss,
        w_init,
        args=(A, b, reg_mat),
        jac=_square_hinge_grad,
        method="L-BFGS-B",
    ).x
    return w


#
# def _fit_CP_KM(self, x, y, w_init, M, D, R, C, reg_par, num_sweeps, max_iter, random_state, class_weight, mu, Ld,
#               train_loss_flag, loss, penalty):
#
#     N, D = x.shape
#     # initialize factors
#     w = init_CP(w_init, M, D, R, random_state=random_state)
#
#     # initialize mapped features
#     if self.class_weight is None or self.class_weight == 'none':
#         (reg, G) = _init_model_params(x, w, self._features)
#         balanced = False
#     elif self.class_weight == 'balanced':
#         # count class instances in y
#         idx_p = tl.where(y == 1)[0]
#         idx_n = tl.where(y == -1)[0]
#         Np = tl.sum(y == 1)
#         Nn = tl.sum(y == -1)
#         Cp = N / (2 * Np) * C
#         Cn = N / (2 * Nn) * C
#         balanced = True
#
#         (reg_n, G_n) = _init_model_params(x[idx_n, :], w, self._features)
#         (reg_p, G_p) = _init_model_params(x[idx_p, :], w, self._features)
#
#     # ALS sweeps
#     itemax = int(tl.min([self.num_sweeps * D, self.max_iter]))
#     loadings = tl.norm(w[0], order=2, axis=0)
#     if balanced:
#         model_parameters = (w, reg, G, loadings)
#         for it in range(itemax):
#             model_parameters = _fit_CP_KM_step(it, x, y, , solver, feature_fun, penalty, compute_objective,
#                                     model_parameters)
#         w, reg, G, loadings = model_parameters
#         w = w * loadings
#     # else:
#     #     model_parameters = (w, reg_n, reg_p, G_n, G_p, loadings)
#     #     for it in range(itemax):
#     #         model_parameters = _fit_CP_KM_balanced_step(it, x, y, C_p, C_m,  solver, feature_fun, penalty,
#     #                                                     compute_objective,
#     #                                 model_parameters)
#     #
#
#
#
#     return w
