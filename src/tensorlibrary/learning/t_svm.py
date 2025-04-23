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


from ._cp_krr import get_system_cp_krr, CPKM_predict
from ._cp_km import init_CP, _solve_TSVM_square_hinge, _init_model_params
from .tt_krr import get_tt_rank, update_wz_tt, initialize_wz
from .features import features
from ..random import tt_random
from ..linalg import dot_kron, truncated_svd
from abc import ABC, abstractmethod, ABCMeta


class BaseTSVM(BaseEstimator, metaclass=ABCMeta):
    """Abstract class for Tensor Support Vector Machine. Format compatible with scikit-learn.

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
        loss="squared_hinge",
        penalty="l2",
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
        self.train_loss_flag = (train_loss_flag,)
        self.train_loss = []
        self._features = partial(
            features,
            m=self.M,
            feature_map=self.feature_map,
            Ld=self.Ld,
            map_param=self.map_param,
        )
        self.loss = loss
        self.penalty = penalty

    @abstractmethod
    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        return self

    @abstractmethod
    def predict(self, x: tl.tensor, **kwargs):
        pass


class CPSVM(BaseTSVM, ClassifierMixin):

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
        loss="squared_hinge",
        penalty="l2",
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
        )

        # self.weights_ = w_init

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        self.classes_ = tl.tensor([-1, 1])  # TODO based on y

        # if self.mu != 0 and self.w_init is not None:
        #     self._extra_reg = True
        #     self.reg_par += self.mu
        # else:
        #     self._extra_reg = False

        rnd = check_random_state(self.random_state)
        # check that x and y have correct shape
        x, y = check_X_y(x, y, y_numeric=True, multi_output=False)
        N, D = x.shape
        # initialize factors
        w = init_CP(self.w_init, self.M, D, self.max_rank, random_state=rnd)

        # initialize mapped features
        if self.class_weight is None or self.class_weight == "none":
            (reg, G) = _init_model_params(x, w, self._features)

        if self.train_loss_flag:
            self.train_loss.append(hinge_loss(y, CPKM_predict(x, w, self._features)))

        # ALS sweeps
        itemax = int(tl.min([self.num_sweeps * D, self.max_iter]))
        for it in range(itemax):
            d = it % D

            z_x = self._features(x[:, d])

            reg /= w[d].T @ w[d]  # remove current factor
            G /= z_x @ w[d]  # remove current factor
            reg_mat = self.reg_par * tl.kron(reg, tl.eye(self.M))
            CC = dot_kron(z_x, G)
            w_d = _solve_TSVM_square_hinge(CC, y, reg_mat, w[d].ravel())

            w[d] = tl.reshape(w_d, (self.M, self.max_rank), order="F")
            # weights_ = tl.cp_tensor.cp_normalize(weights_)
            loadings = tl.norm(w[d], order=2, axis=0)
            w[d] /= loadings
            reg *= w[d].T @ w[d]  # add current factor
            G *= z_x @ w[d]  # add current factor

            if self.train_loss_flag:
                self.train_loss.append(
                    hinge_loss(y, CPKM_predict(x, w, self._features))
                )

        if self.train_loss_flag:
            self.train_loss = tl.tensor(self.train_loss)
        w[d] = w[d] * loadings
        self.weights_ = w
        return self

    def decision_function(self, x: tl.tensor):
        check_is_fitted(self, ["weights_"])
        x = check_array(x)
        return CPKM_predict(x, self.weights_, self._features)

    def predict(self, x: tl.tensor, **kwargs):
        # check_is_fitted(self, ["weights_"])
        return tl.sign(self.decision_function(x))
