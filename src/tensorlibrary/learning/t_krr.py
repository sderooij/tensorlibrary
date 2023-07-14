"""

"""
import tensorly as tl
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence

# from tensorly import tensor

from tensorlibrary import TensorTrain
from tensorlibrary.linalg import dot_kron

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
        self.max_rank = max_rank
        self.random_state = random_state
        self.class_weight = class_weight
        self._init(**kwargs)

    @abstractmethod
    def _init(self, **kwargs):
        pass

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

    def _init(
        self,
        M,
        w_init,
        feature_map,
        reg_par,
        num_sweeps,
        max_rank,
        random_state,
        class_weight,
        **kwargs,
    ):
        self.M = M
        self.w_init = w_init
        self.feature_map = feature_map
        self.reg_par = reg_par
        self.num_sweeps = num_sweeps
        self.max_rank = max_rank
        self.random_state = random_state
        self.class_weight = class_weight
        self._init(**kwargs)

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
        M,
        w_init,
        feature_map,
        reg_par,
        num_sweeps,
        max_rank,
        random_state,
        class_weight,
        **kwargs,
    ):
        self.M = M
        self.w_init = w_init
        self.feature_map = feature_map
        self.reg_par = reg_par
        self.num_sweeps = num_sweeps
        self.max_rank = max_rank
        self.random_state = random_state
        self.class_weight = class_weight
        self._init(**kwargs)

    def fit(self, x: tl.tensor, y: tl.tensor, **kwargs):
        pass

    def predict(self, x: tl.tensor, **kwargs):
        pass

    def score(self, **kwarg):
        pass

    def get_params(self, **kwargs):
        pass

    def set_params(self, **kwargs):
        pass
