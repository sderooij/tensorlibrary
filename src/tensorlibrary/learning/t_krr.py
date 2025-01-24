"""

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
from numbers import Real
from sklearn.metrics import accuracy_score, hinge_loss

# from tensorly import tensor
from copy import deepcopy


from ._cp_krr import get_system_cp_krr, CPKM_predict, CPKM_predict_batchwise
from ._cp_km import init_CP, _solve_TSVM_square_hinge, _init_model_params
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
        Ld=1.0,
        train_loss_flag=False,
        loss="l2",
        penalty="l2",
        random_init=False,
        debug=False,
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
        self.Ld = Ld
        self.train_loss_flag = train_loss_flag
        self.train_loss = []
        self.loss = loss
        self.penalty = penalty
        self.random_init = random_init
        self.debug = debug

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
            assert tl.all(
                tl.tensor([w[i].shape[1] for i in range(len(w))]) == self.M
            ), "w_init does not match M"
        else:
            raise TypeError("Unsupported w_int type")

        # Initialize feature map
        WZ_left, WZ_right = initialize_wz(
            w, x, self.M, self.feature_map, self.map_param, 0
        )
        sweep = list(range(0, D - 1)) + list(range(D - 1, 0, -1))
        ltr = True  # left to right _sweep
        for ite in range(self.num_sweeps):
            for d in sweep:
                # update ltr
                if d == D - 1:
                    ltr = False
                elif d == 0:
                    ltr = True
                z_d = features(
                    x[:, d],
                    m=self.M,
                    feature_map=self.feature_map,
                    map_param=self.map_param,
                    Ld=self.Ld,
                )
                # construct linear subsystem matrix

                WZ = dot_kron(WZ_left[d], dot_kron(z_d, WZ_right[d]))
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
                        (tl.prod([self.M, ranks[d]], dtype=int), ranks[d + 1]),
                        order="F",
                    )
                    Q, R = tl.qr(new_weight, "reduced")
                    # shift norm to next core
                    w[d] = Q.reshape((ranks[d], self.M, ranks[d + 1]), order="F")
                    w[d + 1] = tl.tenalg.mode_dot(w[d + 1], R, 0)
                    # update WZ left
                    WZ_left[d + 1] = update_wz_tt(w[d], z_d, WZ_left[d], mode="left")
                else:
                    new_weight = new_weight.reshape(
                        (ranks[d], tl.prod([self.M, ranks[d + 1]], dtype=int)),
                        order="F",
                    )
                    Q, R = tl.qr(new_weight.T, "reduced")
                    # shift norm to next core
                    w[d] = Q.T.reshape((ranks[d], self.M, ranks[d + 1]), order="F")
                    w[d - 1] = tl.tenalg.mode_dot(w[d - 1], R, 2)
                    # update WZ right
                    WZ_right[d - 1] = update_wz_tt(w[d], z_d, WZ_right[d], mode="right")

        self.weights_ = w
        return self

    def decision_function(self, x: tl.tensor, **kwargs):
        """Compute the decision function for the input samples"""
        D = x.shape[1]
        y_pred = update_wz_tt(
            self.weights_[0],
            features(
                x[:, 0],
                m=self.M,
                feature_map=self.feature_map,
                map_param=self.map_param,
                Ld=self.Ld,
            ),
            None,
            mode="first",
        )
        for d in range(1, D):
            y_pred = update_wz_tt(
                self.weights_[d],
                features(
                    x[:, d],
                    m=self.M,
                    feature_map=self.feature_map,
                    map_param=self.map_param,
                    Ld=self.Ld,
                ),
                y_pred,
                mode="left",
            )
        return y_pred[:, 0]

    def predict(self, x: tl.tensor, **kwargs):
        """Predict class labels for samples in x"""
        y_pred = self.decision_function(x)
        return tl.sign(y_pred)


class CPKRR(ClassifierMixin, BaseTKRR):
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
        class_weight=None,
        max_iter=tl.inf,
        Ld=1.0,
        train_loss_flag=False,
        loss="l2",
        penalty="l2",
        random_init=False,
        debug=False,
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

        # self.weights_ = w_init

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        self.classes_ = tl.tensor([-1, 1])  # TODO based on y
        self._features = partial(
            features,
            m=self.M,
            feature_map=self.feature_map,
            Ld=self.Ld,
            map_param=self.map_param,
        )

        if self.mu != 0 and self.w_init is not None:
            self._extra_reg = True
            self.reg_par += self.mu
            # loadings_w_init = tl.norm(self.w_init[-1], order=2, axis=0)
            # self.w_init[-1] /= loadings_w_init
        else:
            self._extra_reg = False

        rnd = check_random_state(self.random_state)
        # check that x and y have correct shape
        x, y = check_X_y(x, y, y_numeric=True, multi_output=False)
        N, D = x.shape
        # initialize factors
        # if isinstance(self.w_init, list):
        #     for d in range(D):
        #         self.w_init[d] /= tl.norm(self.w_init[d], order=2, axis=0)
        if self.random_init:
            w = init_CP(None, self.M, D, self.max_rank, random_state=rnd)
        else:
            w = init_CP(
                deepcopy(self.w_init), self.M, D, self.max_rank, random_state=rnd
            )

        # if isinstance(self.w_init, list):
        #     for d in range(D):
        #         self.w_init[d] /= tl.norm(self.w_init[d], order=2, axis=0)

        # initialize mapped features
        if self.class_weight is None or self.class_weight == "none":
            reg = 1
            G = 1
            for d in range(D - 1, -1, -1):  # D-1:-1:0
                w_d = w[d]
                reg *= w_d.T @ w_d
                z_x = self._features(x[:, d])
                G = (z_x @ w_d) * G
            balanced = False
        elif self.class_weight == "balanced":
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
                z_x_n = self._features(x[idx_n, d])
                z_x_p = self._features(x[idx_p, d])
                Gn = (z_x_n @ w_d) * Gn
                Gp = (z_x_p @ w_d) * Gp
            balanced = True

        if self._extra_reg:
            extra_reg = tl.ones((self.max_rank, self.max_rank))
            for d in range(D - 1, -1, -1):
                extra_reg *= w[d].T @ self.w_init[d]

        if self.train_loss_flag:
            self.train_loss.append(self._loss_fun(x, y, w))

        # ALS sweeps
        itemax = int(tl.min([self.num_sweeps * D, self.max_iter]))
        for it in range(itemax):
            d = it % D

            if not balanced:
                z_x = self._features(x[:, d])

                reg /= w[d].T @ w[d]  # remove current factor
                G /= z_x @ w[d]  # remove current factor
                CC, Cy = get_system_cp_krr(z_x, G, y)
            else:
                z_x_n = self._features(x[idx_n, d])
                z_x_p = self._features(x[idx_p, d])
                reg /= w[d].T @ w[d]
                Gn /= z_x_n @ w[d]
                Gp /= z_x_p @ w[d]
                CCn, Cyn = get_system_cp_krr(z_x_n, Gn, y[idx_n])
                CCp, Cyp = get_system_cp_krr(z_x_p, Gp, y[idx_p])
                CC = Cn * CCn + Cp * CCp
                Cy = Cn * Cyn + Cp * Cyp

            if self._extra_reg:
                extra_reg /= w[d].T @ self.w_init[d]
                Cy += (
                    self.mu
                    * N
                    * tl.reshape(self.w_init[d] @ extra_reg.T, (-1,), order="F")
                )
            reg_mat = self.reg_par * N * tl.kron(reg, tl.eye(self.M))

            if self.loss == "l2":
                if self.debug:
                    # check that the loss matrix is positive definite
                    if not tl.all(np.linalg.eigvals(CC + reg_mat) > 0):
                        raise ValueError("Loss matrix is not positive definite")
                w_d = tl.solve(CC + reg_mat, Cy)

            del CC, Cy
            w[d] = tl.reshape(w_d, (self.M, self.max_rank), order="F")

            if self.train_loss_flag:
                self.train_loss.append(self._loss_fun(x, y, w))
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

                # self.train_loss.append(1-accuracy_score(y, tl.sign(CPKM_predict(x, w, self._features))))

        if self.train_loss_flag:
            self.train_loss = tl.tensor(self.train_loss)

        w[d] = w[d] * loadings
        self.weights_ = w
        return self

    def decision_function(self, x: tl.tensor):
        check_is_fitted(self, ["weights_"])
        x = check_array(x)
        if not hasattr(self, "_features"):
            self._features = partial(
                features,
                m=self.M,
                feature_map=self.feature_map,
                Ld=self.Ld,
                map_param=self.map_param,
            )
        # if batch_size is None:
        try:
            return CPKM_predict_batchwise(x, self.weights_, self._features)
        except:
            return CPKM_predict(x, self.weights_, self._features)
        # TODO: implement batch mode
        # else:
        # batch_size = 1024
        # for i in range(0, x.shape[0],1024):
        #     if i == 0:
        #         y_pred = CPKM_predict(x[i : i + batch_size], self.weights_, self._features)
        #     else:
        #         y_pred = tl.concatenate(
        #             [y_pred, CPKM_predict(x[i : i + batch_size], self.weights_, self._features)]
        #         )

    def predict(self, x: tl.tensor, **kwargs):
        # check_is_fitted(self, ["weights_"])
        return tl.sign(self.decision_function(x))

    def _loss_fun(self, x: tl.tensor, y: tl.tensor, w: tl.tensor):
        return (tl.norm(y - CPKM_predict(x, w, self._features)) ** 2) / x.shape[
            0
        ]  # normalized loss


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
