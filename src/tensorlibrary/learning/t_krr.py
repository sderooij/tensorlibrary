"""

"""
import tensorly as tl
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence
import numpy as np
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.utils import check_X_y, check_array
# from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
# from tensorly import tensor


from .cp_krr import get_system_cp_krr

from abc import ABC, abstractmethod


class BaseTKRR(ABC):
    """Abstract class for Tensor Kernel Ridge Regression. Format compatible with scikit-learn.

    Args:
        ABC (ABC): Abstract Base Class
    """

    def __init__(
        self,
        M: int = 1,
        w_init=None,
        feature_map="det_fourier",
        reg_par=1e-5,
        num_sweeps=2,
        map_param=1.0,
        max_rank=None,
        random_state=None,
        class_weight=None,
        **kwargs,
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
        self.w = w_init
        # self._init(**kwargs)

    # @abstractmethod
    # def _init(self, **kwargs):
    #     pass

    @abstractmethod
    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        return self

    @abstractmethod
    def predict(self, x: tl.tensor, **kwargs):
        pass

    @abstractmethod
    def score(self, **kwarg):
        pass

    @abstractmethod
    def get_params(self, **kwargs):
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass


class TTKRR(BaseTKRR):
    """Tensor Train Kernel Ridge Regression

    Args:
        BaseTKRR (BaseTKRR): Abstract Base Class
    """

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        return self

    def predict(self, x: tl.tensor, **kwargs):
        return self

    def score(self, **kwarg):
        pass

    def get_params(self, **kwargs):
        pass

    def set_params(self, **kwargs):
        pass


class CPKRR(BaseTKRR):
    """CP Kernel Ridge Regression (KRR)

    Args:
        BaseTKRR (BaseTKRR): Abstract Base Class
    """

    def _init(
            self,
            M: int = 1,
            w_init=None,
            feature_map="det_fourier",
            reg_par=1e-5,
            num_sweeps=2,
            map_param=1.0,
            max_rank=None,
            random_state=None,
            class_weight=None,
            **kwargs,
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
        self.w = w_init
        #TODO: add validation of parameters

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        N, D = x.shape
        # initialize factors
        if self.w_init is None:
            temp = tl.random.random_cp(
                shape=tuple(int(self.M) * tl.ones(D, dtype=int)),
                rank=self.max_rank,
                full=False,
                orthogonal=False,
                random_state=self.random_state,
                normalise_factors=True)
            w = temp.factors
            # w = []
            # for d in range(D):
            #     w_d = np.random.randn(self.M, self.max_rank, random_state=self.random_state)
            #     loadings = tl.norm(w_d, order=2, axis=0)
            #     w_d = w_d / loadings
            #     w.append(w_d)
        elif isinstance(self.w_init, list):
            w = self.w_init
        else:
            raise ValueError("w_init must be a CPTensor or a list of factors")

        # initialize mapped features
        reg = 1
        G = 1
        for d in range(D-1, -1, -1):   #D-1:-1:0
            w_d = w[d]
            reg *= (w_d.T @ w_d)
            z_x = features(x[:, d], self.M, self.feature_map, map_param=self.map_param)
            G = (z_x @ w_d) * G

        # ALS sweeps
        itemax = self.num_sweeps * 2
        for it in range(itemax):
            for d in range(0,D):
                z_x = features(x[:, d], self.M, self.feature_map, map_param=self.map_param)

                reg /= (w[d].T @ w[d])  # remove current factor
                G /= (z_x @ w[d])  # remove current factor
                CC, Cy = get_system_cp_krr(z_x, G, y)
                w_d = tl.solve(CC + self.reg_par*N*tl.kron(reg, tl.eye(self.M)), Cy)
                del CC, Cy
                w[d] = tl.reshape(w_d, (self.max_rank, self.M)).T   # to match C row-major order
                # w = tl.cp_tensor.cp_normalize(w)
                loadings = tl.norm(w[d], order=2, axis=0)
                w[d] /= loadings
                reg *= (w[d].T @ w[d])  # add current factor
                G *= (z_x @ w[d])  # add current factor

        w[d] = w[d] * loadings
        self.w = w
        return self

    def decision_function(self, x: tl.tensor, **kwargs):
        N, D = x.shape
        y_pred = tl.ones((N, 1))
        for d in range(0, D):
            z_x = features(x[:, d], self.M, self.feature_map, map_param=self.map_param)
            y_pred = y_pred * (z_x @ self.w[d])
        y_pred = tl.sum(y_pred, axis=1)
        return y_pred

    def predict(self, x: tl.tensor, **kwargs):
        y_pred = self.decision_function(x, **kwargs)
        return tl.sign(y_pred)

    def score(self, x: tl.tensor, y: tl.tensor, **kwargs):
        y_pred = self.predict(x, **kwargs)
        return accuracy_score(y, y_pred)

    def get_params(self, **kwargs):
        pass

    def set_params(self, **kwargs):
        pass

    def get_weights(self, **kwargs):
        return self.w


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
    if feature_map == "rbf":
        x_d = (x_d + 0.5) * 0.5
        w = tl.arange(1, m + 1)
        s = (
                tl.sqrt(2 * tl.pi)
                * map_param     # lengthscale
                * tl.exp(-((tl.pi * w.T / 2) ** 2) * map_param ** 2 / 2)
        )
        z_x = tl.sin(tl.pi * tl.tenalg.outer([x_d, w])) * tl.sqrt(s)
    elif feature_map == "poly":
        # polynomial feature map
        z_x = tl.zeros((x_d.shape[0], m))
        # in vectorized form
        for i in range(0, m):
            z_x[:, i] = x_d ** i

    elif feature_map == "chebishev":
        # chebishev feature map
        z_x = tl.zeros((x_d.shape[0], m))
        # in vectorized form
        for i in range(0, m):
            z_x[:, i] = tl.cos(i * tl.arccos(2 * x_d - 1))
    else:
        raise NotImplementedError

    return z_x