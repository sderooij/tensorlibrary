from tensorlibrary import TensorTrain
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


def test_init_contract():
    tens = np.random.rand(10, 30, 10, 15, 16)

    # -- first case -------
    tt = TensorTrain(tens, max_ranks=100)
    assert tl.all(tt.ranks <= 100)
    tback = tt.contract(to_array=True)
    assert np.isclose(tt.errs, tl.norm(tens - tback))
    # ----- second case ---------
    tt = TensorTrain(tens, max_trunc_error=0.01, relative=False)
    tback = tt.contract().tensor
    assert tt.errs / tl.norm(tens) <= 0.01
    assert np.isclose(tt.errs, tl.norm(tens - tback))
    # ----- third case ---------
    tt = TensorTrain(tens, max_trunc_error=0.1, relative=True)
    tback = tt.contract().tensor
    # assert tt.errs/np.linalg.norm(tens) <= 0.01
    assert np.isclose(tt.errs, tl.norm(tens - tback))
    assert len(tt.cores) == tt.ndims

    r = [tt.cores[k].edges[2].dimension for k in range(0, tt.ndims - 1)]
    assert_allclose(r, tt.ranks)


def test_add():
    tens = np.random.rand(10, 30, 10, 15, 16)

    # ------ first case =======
    tt = TensorTrain(tens)
    tt2 = tt + tt
    tt2b = tt2.contract(to_array=True)
    assert np.isclose(tl.norm(tt2b), tl.norm(tens+tens))
    assert_array_almost_equal(tt2b, tens+tens)


def test_sub():
    tens = np.random.rand(10, 28, 10, 15, 16)

    # ------ first case =======
    tt = TensorTrain(tens)
    tt2 = 3*tt - tt
    tt2b = tt2.contract(to_array=True)
    assert np.isclose(tl.norm(tt2b), tl.norm(tens+tens))
    assert_array_almost_equal(tt2b, tens+tens)


def test_mul():
    tens = np.random.rand(10, 30, 10, 15, 16)

    # ------ first case =======
    tt = TensorTrain(tens)
    tt2 = tt * 2.
    tt2b = tt2.contract(to_array=True)
    assert np.isclose(tl.norm(tt2b), tl.norm(tens+tens))
    assert_array_almost_equal(tt2b, tens+tens)


def test_div():
    tens = np.random.rand(10, 30, 10, 15, 16)

    # ------ first case =======
    tt = TensorTrain(tens)
    tt2 = tt / 0.5
    tt2b = tt2.contract(to_array=True)
    assert np.isclose(tl.norm(tt2b), tl.norm(tens+tens))
    assert_array_almost_equal(tt2b, tens+tens)






