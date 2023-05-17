from tensorlibrary import TTmatrix
import numpy as np
import tensorly as tl
from tensorly.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_,
    assert_array_almost_equal,
    assert_raises,
)
import pytest


def test_init():
    mat = np.random.rand(16,20)

    # -- first case -------
    tt = TTmatrix(mat, max_ranks=10)
    assert tl.all(tt.ranks <= 10)
    # ------- second case -----
    tt = TTmatrix(mat, max_trunc_error=0.)
    tback = tt.contract()
    assert_allclose(mat, tback)
    assert_allclose(tt.errs, 0.)


