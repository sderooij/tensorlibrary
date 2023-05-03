from tensorlibrary import QTT
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


def test_init_contract():
    tens = np.random.rand(10*30*10*15*16)

    # -- first case -------
    tt = QTT(tens, max_ranks=100)
    assert tl.all(tt.ranks <= 100)
    tback = tl.tensor_to_vec(tt.contract(to_array=True))
    assert np.isclose(tt.errs, tl.norm(tens - tback))
    # ----- second case ---------
    tt = QTT(tens, max_trunc_error=0.01, relative=False)
    tback = tl.tensor_to_vec(tt.contract(to_array=True))
    assert tt.errs / tl.norm(tens) <= 0.01
    assert np.isclose(tt.errs, tl.norm(tens - tback))
    # ----- third case ---------
    tt = QTT(tens, max_trunc_error=0.1, relative=True)
    tback = tl.tensor_to_vec(tt.contract(to_array=True))
    # assert tt.errs/np.linalg.norm(tens) <= 0.01
    assert np.isclose(tt.errs, tl.norm(tens - tback))
    assert len(tt.cores) == tt.ndims

    r = [tt.cores[k].edges[2].dimension for k in range(0, tt.ndims - 1)]
    assert_allclose(r, tt.ranks)


def test_add():
    tens = np.random.rand(10*30*10*15*16)

    # ------ first case =======
    tt = QTT(tens)
    tt2 = tt + tt
    tt2b = tl.tensor_to_vec(tl.tensor_to_vec(tt2.contract(to_array=True)))
    assert np.isclose(tl.norm(tt2b), tl.norm(tens + tens))
    assert_array_almost_equal(tt2b, tens + tens)


def test_sub():
    tens = np.random.rand(10*28*10*15*16)

    # ------ first case =======
    tt = QTT(tens)
    tt2 = 3 * tt - tt
    tt2b = tl.tensor_to_vec(tl.tensor_to_vec(tt2.contract(to_array=True)))
    assert np.isclose(tl.norm(tt2b), tl.norm(tens + tens))
    assert_array_almost_equal(tt2b, tens + tens)


def test_mul():
    tens = np.random.rand(10*30*10*15*16)

    # ------ first case =======
    tt = QTT(tens)
    tt2 = tt * 2.0
    tt2b = tl.tensor_to_vec(tt2.contract(to_array=True))
    assert np.isclose(tl.norm(tt2b), tl.norm(tens + tens))
    assert_array_almost_equal(tt2b, tens + tens)


def test_div():
    tens = np.random.rand(10*30*10*15*16)

    # ------ first case =======
    tt = QTT(tens)
    tt2 = tt / 0.5
    tt2b = tl.tensor_to_vec(tt2.contract(to_array=True))
    assert np.isclose(tl.norm(tt2b), tl.norm(tens + tens))
    assert_array_almost_equal(tt2b, tens + tens)

#
# def test_dot():
#     tens = np.random.rand(10, 30, 10, 15, 16, 3)
#     tt = QTT(tens)
#     assert np.isclose(tt.dot(tt), tl.norm(tens) ** 2)
#     assert np.isclose(tt.dot(tens), tl.norm(tens) ** 2)
#     assert np.isclose(tt.dot(tl.reshape(tens, tens.size)), tl.norm(tens) ** 2)
#     with pytest.raises(ValueError):
#         tens = np.random.rand(10, 20, 30, 10)
#         tt.dot(tens)
#
#
# def test_norm():
#     tens = np.random.rand(10, 30, 10, 15, 16, 3)
#     tt = QTT(tens)
#     assert np.isclose(tt.norm(), tl.norm(tens))
#     assert np.isclose(tt.copy().norm(), tl.norm(tens))
#
#
# def test_orthogonalize():
#     tens = np.random.rand(10, 30, 10, 15, 16, 3)
#     tt = QTT(tens)
#     tt_ortho = tt.orthogonalize(3)
#     assert np.isclose(tt_ortho.norm(), tl.norm(tens))
#     assert np.isclose(tl.norm(tt_ortho.cores[3].tensor), tl.norm(tens))
#     assert np.isclose(tt.dot(tt), tt_ortho.norm() ** 2)
#     # assert np.isclose(tl.norm(tt_ortho.cores[1].tensor),1)
#     tback_ortho = tt_ortho.contract(to_array=True)
#     assert_array_almost_equal(tback_ortho, tens)