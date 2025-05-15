"""
    Transfer learning with TKMs
"""

import tensorly as tl
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from functools import partial

# from sklearn.utils import check_X_y, check_array
# from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    check_random_state,
    check_is_fitted,
    check_array,
    check_X_y,
)
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, pairwise_kernels
import cvxopt
from cvxopt import matrix, spmatrix, solvers

from numbers import Real
from sklearn.metrics import accuracy_score, hinge_loss

# from tensorly import tensor
from copy import deepcopy

# local imports
from .t_krr import CPKRR
from ._cp_krr import get_system_cp_krr, get_system_cp_LMPROJ
from ._cp_km import init_CP, _init_model_params, _init_model_params_LMPROJ
from .features import features


class CPKRR_Adapt(CPKRR):
    # TODO: use this in the future for the adapt methods
    _parameter_contraints: Dict[str, Dict[str, Any]] = {
        "M": [Interval(Real, 1, None, closed="left")],
        "w_init": [Interval(Real, None, None, closed="neither")],
        "feature_map": [
            StrOptions({"rbf", "fourier", "poly", "chebyshev", "chebyshev2"})
        ],
        "reg_par": [Interval(Real, 0, None, closed="neither")],
        "mu": [Interval(Real, 0, None, closed="neither")],
        "num_sweeps": [Interval(Real, 1, None, closed="left")],
        "map_param": [Interval(Real, 0, None, closed="neither")],
        "max_rank": [Interval(Real, 1, None, closed="left")],
        "random_state": ["random_state"],
        "class_weight": [StrOptions({"balanced"}), dict, None],
    }

    def __init__(
            self,
            M: int = 5,
            w_init=None,
            feature_map="rbf",
            reg_par=1e-5,
            num_sweeps=15,
            map_param=0.1,
            max_rank=5,
            random_state=None,
            mu=0,
            class_weight=None,
            max_iter=tl.inf,
            Ld=1.0,
            train_loss_flag=False,
            loss="l2",
            penalty="l2",
            random_init=True,
            debug=False,
            source_model=None,
            batch_size=8192,
    ):
        super().__init__(
            M=M,
            w_init=w_init,
            feature_map=feature_map,
            reg_par=reg_par,
            num_sweeps=num_sweeps,
            map_param=map_param,
            max_rank=max_rank,
            random_state=random_state,
            class_weight=class_weight,
            max_iter=max_iter,
            mu=mu,
            Ld=Ld,
            train_loss_flag=train_loss_flag,
            loss=loss,
            penalty=penalty,
            random_init=random_init,
            debug=debug,
        )
        self._features = None
        self.source_model = source_model
        self.batch_size = batch_size

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        """
        Solving:
            min_w ||y-<G, w>
        Args:
            x:
            y:
            **kwargs:

        Returns:

        """
        # set parameters
        assert tl.min(y) == -1, "negative class must be -1"
        assert tl.max(y) == 1, "positive class must be 1"
        self.classes_ = tl.tensor([tl.min(y), tl.max(y)])

        self._features = partial(
            features,
            m=self.M,
            feature_map=self.feature_map,
            Ld=self.Ld,
            map_param=self.map_param,
        )
        N, D = x.shape
        rnd = check_random_state(self.random_state)

        # Initialize reference weights
        if self.w_init == "random":
            self.random_init = True
        elif self.w_init == "source":
            self.random_init = False
        w_source = deepcopy(self.source_model.weights_)
        loadings_source, w_source = tl.cp_tensor.cp_normalize(
            (tl.ones(self.max_rank), w_source)
        )

        # Initialize weights with random or source model
        if self.random_init:
            w = init_CP(None, self.M, D, self.max_rank, random_state=rnd)
        else:
            w = deepcopy(w_source)
        self.w_init = deepcopy(w)

        """
        min_w[d] ||y-<G, w[d]>||_F^2 + (reg_par + mu)<W[d]^T W[d], H> - 2*mu <W[d].T, W_source @ K.T>

             where, K = (w[1].T w_source[1] * ... * w[d-1].T w_source[d-1] * w[d+1].T w_source[
             d+1] * ... * w[D].T w_source[D])
        """
        loadings = tl.ones(self.max_rank)
        # Initialize parameters
        if self.class_weight is None or self.class_weight == "none":
            H, G = _init_model_params(x, w, self._features)
            balanced = False
        elif self.class_weight == "balanced":
            balanced = True
            H, Gn, Gp, c_n, c_p, idx_n, idx_p = _init_model_params(
                x, w, self._features, balanced=True, y=y
            )
        else:
            raise ValueError("class_weight must be 'none' or 'balanced'")

        # Initialize K
        K = tl.ones((self.max_rank, self.max_rank))
        for d in range(D):
            K *= w[d].T @ w_source[d]

        # ALS Sweeps
        itemax = int(tl.min([self.num_sweeps * D, self.max_iter]))
        for it in range(itemax):
            # "core" to update
            d = it % D

            # calculate mapped features
            z_x = self._features(x[:, d])

            # remove d-th component of the terms
            H /= w[d].T @ w[d]
            K /= w[d].T @ w_source[d]

            if balanced:
                z_x_n = self._features(x[idx_n, d])
                z_x_p = self._features(x[idx_p, d])
                Gn /= z_x_n @ w[d]
                Gp /= z_x_p @ w[d]
            else:
                z_x = self._features(x[:, d])
                G /= z_x @ w[d]  # remove current factor

            # get system of equations
            if balanced:
                GGn, Gyn = get_system_cp_krr(
                    z_x_n, Gn, y[idx_n], batch_size=self.batch_size
                )
                GGp, Gyp = get_system_cp_krr(
                    z_x_p, Gp, y[idx_p], batch_size=self.batch_size
                )
                GG = c_n * GGn + c_p * GGp
                Gy = c_n * Gyn + c_p * Gyp
            else:
                GG, Gy = get_system_cp_krr(z_x, G, y, batch_size=self.batch_size)

            # Solve system and assign to w[d]
            A = GG + (self.reg_par + self.mu) * N * tl.kron(H.T, tl.eye(self.M))
            b = Gy + self.mu * N * tl.reshape(
                (w_source[d] * loadings_source) @ K.T, (-1,), order="F"
            )
            w_d = tl.solve(A, b)
            w[d] = tl.reshape(w_d, (self.M, self.max_rank), order="F")

            # Normalize
            loadings = tl.norm(w[d], order=2, axis=0)
            w[d] /= loadings

            # Update terms
            H *= w[d].T @ w[d]
            K *= w[d].T @ w_source[d]
            if balanced:
                Gn *= z_x_n @ w[d]
                Gp *= z_x_p @ w[d]
            else:
                G *= z_x @ w[d]

        w[d] = w[d] * (loadings)
        self.weights_ = w
        return self


class CPKRR_LMPROJ(CPKRR):
    def __init__(
        self,
        M: int = 5,
        w_init=None,
        feature_map="rbf",
        reg_par=1e-5,
        num_sweeps=15,
        map_param=0.1,
        max_rank=5,
        random_state=None,
        mu=0,
        class_weight=None,
        max_iter=tl.inf,
        Ld=1.0,
        train_loss_flag=False,
        loss="l2",
        penalty="l2",
        random_init=True,
        debug=False,
        batch_size=8192,
    ):
        super().__init__(
            M=M,
            w_init=w_init,
            feature_map=feature_map,
            reg_par=reg_par,
            num_sweeps=num_sweeps,
            map_param=map_param,
            max_rank=max_rank,
            random_state=random_state,
            class_weight=class_weight,
            max_iter=max_iter,
            mu=mu,
            Ld=Ld,
            train_loss_flag=train_loss_flag,
            loss=loss,
            penalty=penalty,
            random_init=random_init,
            debug=debug,
        )
        self._features = None
        self.batch_size = batch_size

    def fit(self, x: tl.tensor, y: tl.tensor, x_target=None):
        """

        Args:
            x:
            y:
            sample_weight: array of shape (n_samples,) should be 1 for the source and -1 for the negative class

        Returns:

        """
        if x_target is None:
            Warning("x_target is None, using standard CPKRR")
            return super().fit(x, y)

        self.classes_ = tl.tensor([-1, 1])

        N_s, D = x.shape
        N_t, Dt = x_target.shape
        assert D == Dt, "source and target data must have the same number of features"
        N = N_s + N_t
        rnd = check_random_state(self.random_state)

        # set feature fun
        self._features = partial(
            features,
            m=self.M,
            feature_map=self.feature_map,
            Ld=self.Ld,
            map_param=self.map_param,
        )

        # Initialize reference weights
        if self.w_init == "random":
            self.random_init = True

        # Initialize weights with random or source model
        if self.random_init:
            w = init_CP(None, self.M, D, self.max_rank, random_state=rnd)
        else:
            w = deepcopy(self.w_init)

        """
        min_w[d] ||y-<G, w[d]>||_F^2 + (reg_par)<W[d]^T W[d], H> + mu ||<gamma*Q, w[d]>||_F^2

             where, K = (w[1].T w_source[1] * ... * w[d-1].T w_source[d-1] * w[d+1].T w_source[
             d+1] * ... * w[D].T w_source[D])
        """
        loadings = tl.ones(self.max_rank)
        # Initialize parameters
        if self.class_weight is None or self.class_weight == "none":
            H, G, G_target = _init_model_params_LMPROJ(x, w, self._features, x_target)
            balanced = False
            sample_weights=None
        elif self.class_weight == "balanced":
            balanced = True
            H, G, G_target, Cn, Cp, idx_n, idx_p = _init_model_params_LMPROJ(
                x, w, self._features, x_target, balanced=True, y=y
            )
            sample_weights = tl.ones((N_s, 1))  # for source
            sample_weights[idx_n] = Cn
            sample_weights[idx_p] = Cp
        else:
            raise ValueError("class_weight must be 'none' or 'balanced'")

        # set gamma
        gamma = tl.tensor([1/N_s, -(1/N_t)])#tl.concatenate([(1/N_s) * tl.ones((N_s, 1)), -(1/N_t) * tl.ones((N_t, 1))])

        # ALS Sweeps
        itemax = int(tl.min([self.num_sweeps * D, self.max_iter]))
        for it in range(itemax):
            # "core" to update
            d = it % D

            # calculate mapped features
            z_x = self._features(x[:, d])
            z_x_target = self._features(x_target[:, d])
            # remove d-th component of the terms
            H /= w[d].T @ w[d]
            G /= z_x @ w[d]  # remove current factor
            G_target /= z_x_target @ w[d]

            # get system of equations
            GG, Gy = get_system_cp_krr(z_x, G, y, batch_size=self.batch_size, sample_weights=sample_weights)
            QQ = get_system_cp_LMPROJ(
                z_x, z_x_target, G, G_target, gamma, batch_size=self.batch_size
            )

            # Solve system and assign to w[d]
            A = GG + self.reg_par * N_s * tl.kron(H.T, tl.eye(self.M)) + self.mu * N_s * QQ       #TODO: N should be N_s, but this seems to work???
            b = Gy
            w_d = tl.solve(A, b)
            w[d] = tl.reshape(w_d, (self.M, self.max_rank), order="F")

            # Normalize
            loadings = tl.norm(w[d], order=2, axis=0)
            w[d] /= loadings

            # Update terms
            H *= w[d].T @ w[d]
            G *= z_x @ w[d]
            G_target *= z_x_target @ w[d]

        w[d] = w[d] * (loadings)
        self.weights_ = w
        return self


class SVC_LMPROJ(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        mu=1e-5,
        gamma="scale",
        reg_par=1e-5,
        degree=3,
        coef0=1,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        random_state=None,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.reg_par = reg_par
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        self.mu = mu

    def _compute_kernel_mat(self, x, x_target):
        """
        Compute Omega and Lambda matrices for the LMPROJ method
        Args:
            x: source data \in R^{N_s x D}
            x_target: target data \in R^{N_t x D}

        Returns:
            Omega, Lambda, K_s
        """
        N_s = x.shape[0]
        N_t = x_target.shape[0]
        N = N_s + N_t
        Z = tl.concatenate([x, x_target], axis=0)

        # Compute the kernel matrix for the source data
        K_s = rbf_kernel(Z, x, gamma=self.gamma) # N x N_s

        # Compute the kernel matrix for the target data
        K_t= rbf_kernel(Z, x_target, gamma=self.gamma)

        # Combine the kernel matrices to form Omega
        Omega = ((1/(N_s**2) * (K_s @ tl.ones((N_s, N_s)) @ K_s.T)
                 + 1/(N_t**2) * (K_t @ tl.ones((N_t, N_t)) @ K_t.T))
                 - 1/(N_s*N_t) * (
                         K_s @ tl.ones((N_s, N_t)) @ K_t.T
                         + K_t @ tl.ones((N_t, N_s)) @ K_s.T
                 ))

        # compute Lambda K(Z,Z) from K_s and K_t
        Lambda = rbf_kernel(Z, Z, gamma=self.gamma)
        return Omega, Lambda, K_s

    def optimize(self, Omega, Lambda, K_s, y, N_t):
        """
        Quadratic optimization algorithm
        Args:
            Omega: Omega matrix
            Lambda: Lambda matrix
            x: source data \in R^{N_s x D}
            y: source labels \in R^{N_s x 1}

        Returns:
            alpha: dual variables
        """
        N_s = y.shape[0]
        N = N_s + N_t

        n_vars = N + 1 + N_s

        # Define the quadratic programming problem
        P = np.zeros((n_vars, n_vars))
        P[:N, :N] = self.mu*Omega + Lambda + self.reg_par * np.eye(N)
        P = matrix(P)

        q = np.zeros((n_vars, 1))
        q[N+1:] = self.C*np.ones((N_s, 1))
        q = matrix(q)

        G = np.zeros((2*N_s, n_vars))
        G[:N_s, :N] = -y[:, np.newaxis] * K_s.T
        G[:N_s, N] = -y
        G[:N_s, N+1:] = -np.eye(N_s)
        G[N_s:, N+1:] = -np.eye(N_s)
        G = matrix(G)

        h = np.zeros((2*N_s, 1))
        h[:N_s] = -np.ones((N_s, 1))
        h = matrix(h)

        solvers.options['show_progress'] = self.verbose

        sol = solvers.qp(P, q, G, h)
        alpha = np.array(sol["x"][:N]).flatten()
        b = np.array(sol["x"][N])
        return alpha, b


    def fit(self, x, y, x_target=None):
        """
        Fit the model
        """
        if x_target is None:
            Warning("x_target is None, using standard SVC")
            return super().fit(x, y)

        # check if x and y are valid
        x, y = check_X_y(x, y)
        x_target = check_array(x_target)
        N_s, D = x.shape
        N_t, Dt = x_target.shape

        # compute kernel matrices
        Omega, Lambda, K_s = self._compute_kernel_mat(x, x_target)
        # optimize
        alpha, b = self.optimize(Omega, Lambda, K_s, y, N_t)
        # set alpha and b
        # remove zeros from alpha
        z = np.concatenate([x, x_target], axis=0)
        self.support_ = np.where(alpha > 1e-6)[0]
        self.dual_coef_ = alpha[self.support_]
        self.intercept_ = b
        # set support vectors
        self.support_vectors_ = z[self.support_,:]
        self.n_features_in_ = D

        return self

    def decision_function(self, x):
        """
        Compute the decision function for the given data
        Args:
            x: data to predict

        Returns:
            decision function
        """
        check_is_fitted(self)
        x = check_array(x)
        K = rbf_kernel(x, self.support_vectors_, gamma=self.gamma)
        return K @ self.dual_coef_ + self.intercept_

    def predict(self, x):
        """
        Predict the labels for the given data
        Args:
            x: data to predict

        Returns:
            predicted labels
        """
        return np.sign(self.decision_function(x))