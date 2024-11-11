
import tensorly as tl
from tensorly import tenalg
from tensorlibrary import learning as tl_learn
from tensorlibrary.linalg import dot_kron
# import tensorlibrary.learning.t_krr as t_krr
from tensorlibrary.random import tt_random

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# get some data
# data = load_breast_cancer()
# X = data.data[:2,:]
# y = data.target[:2]
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# # change the labels to -1 and 1
# y[y == 0] = -1
# N, D = X.shape
# z_d = tl_learn.features(X[:, 0], m=3, feature_map="poly", map_param=0.1)
# shape_weights = 3 * tl.ones(D, dtype=int)
# ranks = tl_learn.get_tt_rank(shape_weights, 4)
# weights = tt_random(shape_weights, ranks, random_state=42, cores_only=True)
#
# # %% test update_wz_tt
# wz_left, wz_right = tl_learn.initialize_wz(weights, X, M=3, feature_map="poly", map_param=0.1, k_core=D-1)
# # %% get wz
# z_d = tl_learn.features(X[:, D-1], m=3, feature_map="poly", map_param=0.1)
# wz = dot_kron(dot_kron(wz_left[D-1], z_d), wz_right[D-1])
#
# # TODO: check with matlab
# # %%
# from tensorlibrary import TensorTrain
# w = TensorTrain(cores=weights)
# g = []
# z = tl.ones((N, 3, D))
#
# for d in range(D-1):
#     z[:, :, d] = tl_learn.features(X[:, d], m=3, feature_map="poly", map_param=0.1)
#
# for n in range(N):
#     g.append(tl.reshape(tl_learn.get_g(w, z[n, :, :].T, D-1).tensor, (-1,), order="F"))
#
# g = tl.stack(g, axis=0)

from scipy.io import loadmat
import numpy as np
tenalg.set_backend('core')
from tensorly import plugins
plugins.use_default_einsum()

data = loadmat("../data/tt_init.mat")
w = data["w"]
wz_2 = data['WZ_2'][0]
wz_1 = data['WZ_1'][0]
data = loadmat("../data/breast_cancer.mat")
X_train = data["X_train"]
y_train = data["y_train"].ravel()
X_test = data["X_test"]
y_test = data["y_test"].ravel()

# w = [core for core in w.T]
w = list(w[0])
# wz_left, wz_right = tl_learn.initialize_wz(w, X_train, M=3, feature_map="rbf", map_param=0.1, k_core=0)
from tensorlibrary.learning.t_krr import TTKRR

# ttkrr = TTKRR(max_rank=30, M=3,feature_map="rbf", map_param=0.1, reg_par=1e-10, num_sweeps=5)
#
# ttkrr = ttkrr.fit(X_train, y_train)
#
# #%% test
# yhat = ttkrr.predict(X_test)

print(np.mean(yhat == y_test))
