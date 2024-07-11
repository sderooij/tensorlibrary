"""

"""
import tensorly as tl
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# from sklearn.utils import check_X_y, check_array
# from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    check_random_state,
    check_is_fitted,
    check_array,
    check_X_y,
)
from numbers import Real
from sklearn.metrics import accuracy_score

# from tensorly import tensor


from ._cp_krr import get_system_cp_krr
from .tt_krr import get_tt_rank, update_wz_tt, initialize_wz
from .features import features
from ..random import tt_random
from ..linalg import dot_kron, truncated_svd
from abc import ABC, abstractmethod, ABCMeta


class BaseTKRR(BaseEstimator, metaclass=ABCMeta):
    """Abstract class for Tensor Kernel Ridge Regression. Format compatible with scikit-learn.

    Args:
        ABC (ABC): Abstract Base Class
    """

    @abstractmethod
    def __init__(
        self,
        M: int = 1,
        w_init=None,
        feature_map="rbf",
        reg_par=1e-5,
        num_sweeps=2,
        map_param=1.0,
        max_rank=1,
        mu=0,
        random_state=None,
        class_weight=None,
        max_iter=tl.inf,
    ):
        self.M = M
        self.w_init = w_init
        self.feature_map = feature_map
        self.reg_par = reg_par
        self.num_sweeps = num_sweeps
        self.map_param = map_param
        self.max_rank = max_rank
        self.random_state = random_state
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.mu = mu

    @abstractmethod
    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        return self

    @abstractmethod
    def predict(self, x: tl.tensor, **kwargs):
        pass


class TTKRR(BaseTKRR, ClassifierMixin):
    """Tensor Train Kernel Ridge Regression

    Args:
        BaseTKRR (BaseTKRR): Abstract Base Class
    """
    def __init__(
        self,
        M: int = 5,
        w_init=None,
        feature_map="rbf",
        reg_par=1e-5,
        num_sweeps=15,
        map_param=0.1,
        max_rank=5,
        mu=0,
        random_state=None,
        class_weight="balanced",
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
        )
        self.mu = mu

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        self.classes_ = tl.tensor([-1, 1])  # TODO based on y

        if self.mu != 0 and self.w_init is not None:
            self._extra_reg = True
        else:
            self._extra_reg = False

        # check that x and y have correct shape
        x, y = check_X_y(x, y, y_numeric=True, multi_output=False)
        N, D = x.shape
        shape_weights = self.M * tl.ones(D)  # m^D

        if isinstance(self.max_rank, int):
            ranks = get_tt_rank(shape_weights, self.max_rank)
        elif isinstance(self.max_rank, list):
            ranks = self.max_rank
        else:
            raise TypeError("Unsupported max_rank type")

        # Initialize weights
        if self.w_init is None:
            w = tt_random(
                shape_weights, ranks, random_state=self.random_state, cores_only=True
            )

        elif isinstance(self.w_init, list):
            w = self.w_init
            ranks = [w[i].shape[0] for i in range(len(w))] + [1]
            assert tl.all(tl.tensor([w[i].shape[1] for i in range(len(w))]) == self.M), "w_init does not match M"
        else:
            raise TypeError("Unsupported w_int type")

        # Initialize feature map
        WZ_left, WZ_right = initialize_wz(w, x, self.M, self.feature_map, self.map_param, 0)
        sweep = list(range(0, D-1)) + list(range(D-1, 0, -1))
        ltr = True  # left to right _sweep
        for ite in range(self.num_sweeps):
            for d in sweep:
                # update ltr
                if d == D - 1:
                    ltr = False
                elif d == 0:
                    ltr = True
                z_d = features(x[:, d], m=self.M, feature_map=self.feature_map, map_param=self.map_param)
                # construct linear subsystem matrix

                WZ = dot_kron(
                    WZ_left[d],
                    dot_kron(z_d, WZ_right[d])
                )
                # %% Solve the sytem
                # TODO : make solver a variable + implement batch mode
                # with solve, regularization needed
                # CC = tl.dot(WZ.T, WZ)
                # CC += self.reg_par * tl.eye(CC.shape[0])
                # yy = tl.dot(WZ.T, y)
                # new_weight = tl.solve(CC, yy)

                # with pinv, no regularization needed
                # new_weight = np.linalg.pinv(WZ) @ y

                # with truncated svd
                u, s, v, _ = truncated_svd(WZ, max_trunc_error=0.05, relative=True)
                new_weight = v.T @ np.diag(1 / s) @ u.T @ y

                # %% orthogonalize
                if ltr:
                    new_weight = new_weight.reshape(
                        (tl.prod([self.M, ranks[d]], dtype=int), ranks[d+1]),
                        order='F'
                    )
                    Q, R = tl.qr(new_weight, 'reduced')
                    # shift norm to next core
                    w[d] = Q.reshape(
                        (ranks[d], self.M, ranks[d + 1]),
                        order='F'
                    )
                    w[d+1] = tl.tenalg.mode_dot(w[d+1], R, 0)
                    # update WZ left
                    WZ_left[d+1] = update_wz_tt(w[d], z_d, WZ_left[d], mode="left")
                else:
                    new_weight = new_weight.reshape(
                        (ranks[d], tl.prod([self.M, ranks[d + 1]], dtype=int)),
                        order='F'
                    )
                    Q, R = tl.qr(new_weight.T, 'reduced')
                    # shift norm to next core
                    w[d] = Q.T.reshape((ranks[d], self.M, ranks[d + 1]), order='F')
                    w[d-1] = tl.tenalg.mode_dot(w[d-1], R, 2)
                    # update WZ right
                    WZ_right[d-1] = update_wz_tt(w[d], z_d, WZ_right[d], mode="right")


        self.weights_ = w
        return self

    def decision_function(self, x: tl.tensor, **kwargs):
        """Compute the decision function for the input samples"""
        D = x.shape[1]
        y_pred = update_wz_tt(
            self.weights_[0],
            features(x[:, 0], m=self.M, feature_map=self.feature_map, map_param=self.map_param),
            None,
            mode="first")
        for d in range(1, D):
            y_pred = update_wz_tt(
                self.weights_[d],
                features(x[:, d], m=self.M, feature_map=self.feature_map, map_param=self.map_param),
                y_pred,
                mode="left")
        return y_pred[:,0]

    def predict(self, x: tl.tensor, **kwargs):
        """Predict class labels for samples in x"""
        y_pred = self.decision_function(x)
        return tl.sign(y_pred)


class CPKRR(BaseTKRR, ClassifierMixin):
    """CP Kernel Ridge Regression (KRR)

    Args:
        BaseTKRR (BaseTKRR): Abstract Base Class
    """

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
        class_weight="balanced",
        max_iter=tl.inf,
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
        )

        # self.weights_ = w_init

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        self.classes_ = tl.tensor([-1, 1])  # TODO based on y

        if self.mu != 0 and self.w_init is not None:
            self._extra_reg = True
            self.reg_par += self.mu
        else:
            self._extra_reg = False

        rnd = check_random_state(self.random_state)
        # check that x and y have correct shape
        x, y = check_X_y(x, y, y_numeric=True, multi_output=False)
        N, D = x.shape
        # initialize factors
        if self.w_init is None:
            temp = tl.random.random_cp(
                shape=tuple(int(self.M) * tl.ones(D, dtype=int)),
                rank=self.max_rank,
                full=False,
                orthogonal=False,
                random_state=self.random_state,
                normalise_factors=True,
            )
            self.w_init = temp.factors
            w = self.w_init
        elif isinstance(self.w_init, list):
            w = self.w_init
            for d in range(D):
                w[d] /= tl.norm(w[d], order=2, axis=0)
        elif isinstance(self.w_init, tl.cp_tensor.CPTensor):
            w = self.w_init.factors
        else:
            raise ValueError("w_init must be a CPTensor or a list of factors")

        # initialize mapped features
        if self.class_weight is None:
            reg = 1
            G = 1
            for d in range(D - 1, -1, -1):  # D-1:-1:0
                w_d = w[d]
                reg *= w_d.T @ w_d
                z_x = features(x[:, d], self.M, self.feature_map, map_param=self.map_param)
                G = (z_x @ w_d) * G
            balanced = False
        elif self.class_weight == 'balanced':
            # count class instances in y
            idx_p = tl.where(y == 1)[0]
            idx_n = tl.where(y == -1)[0]
            Np = tl.sum(y == 1)
            Nn = tl.sum(y == -1)
            Cp = N / (2 * Np)
            Cn = N / (2 * Nn)

            reg = 1
            Gn = 1
            Gp = 1
            for d in range(D - 1, -1, -1):  # D-1:-1:0
                w_d = w[d]
                reg *= w_d.T @ w_d
                z_x_n = features(x[idx_n, d], self.M, self.feature_map, map_param=self.map_param)
                z_x_p = features(x[idx_p, d], self.M, self.feature_map, map_param=self.map_param)
                Gn = (z_x_n @ w_d) * Gn
                Gp = (z_x_p @ w_d) * Gp
            balanced = True

        if self._extra_reg:
            extra_reg = reg

        # ALS sweeps
        itemax = int(tl.min([self.num_sweeps * D, self.max_iter]))
        for it in range(itemax):
            d = it % D

            if not balanced:
                z_x = features(
                    x[:, d], self.M, feature_map=self.feature_map, map_param=self.map_param
                )

                reg /= w[d].T @ w[d]  # remove current factor
                G /= z_x @ w[d]  # remove current factor
                CC, Cy = get_system_cp_krr(z_x, G, y)
            else:
                z_x_n = features(x[idx_n, d], self.M, self.feature_map, map_param=self.map_param)
                z_x_p = features(x[idx_p, d], self.M, self.feature_map, map_param=self.map_param)
                reg /= w[d].T @ w[d]
                Gn /= z_x_n @ w[d]
                Gp /= z_x_p @ w[d]
                CCn, Cyn = get_system_cp_krr(z_x_n, Gn, y[idx_n])
                CCp, Cyp = get_system_cp_krr(z_x_p, Gp, y[idx_p])
                CC = Cn*CCn + Cp*CCp
                Cy = Cn*Cyn + Cp*Cyp

            if self._extra_reg:
                extra_reg /= w[d].T @ self.w_init[d]
                Cy += self.mu * tl.reshape(self.w_init[d] @ extra_reg.T, (-1,), order='F')

            w_d = tl.solve(CC + self.reg_par * N * tl.kron(reg, tl.eye(self.M)), Cy)
            del CC, Cy
            w[d] = tl.reshape(
                w_d, (self.M, self.max_rank), order="F"
            )
            # weights_ = tl.cp_tensor.cp_normalize(weights_)
            loadings = tl.norm(w[d], order=2, axis=0)
            w[d] /= loadings
            reg *= w[d].T @ w[d]  # add current factor
            if not balanced:
                G *= z_x @ w[d]  # add current factor
            else:
                Gn *= z_x_n @ w[d]
                Gp *= z_x_p @ w[d]

            if self._extra_reg:
                extra_reg *= w[d].T @ self.w_init[d]

        w[d] = w[d] * loadings
        self.weights_ = w
        return self

    def decision_function(self, x: tl.tensor, **kwargs):
        check_is_fitted(self, ["weights_"])
        x = check_array(x)
        N, D = x.shape
        y_pred = tl.ones((N, 1))
        for d in range(0, D):
            z_x = features(x[:, d], self.M, self.feature_map, map_param=self.map_param)
            y_pred = y_pred * (z_x @ self.weights_[d])
        y_pred = tl.sum(y_pred, axis=1)
        return y_pred

    def predict(self, x: tl.tensor, **kwargs):
        # check_is_fitted(self, ["weights_"])
        y_pred = self.decision_function(x, **kwargs)
        return tl.sign(y_pred)

