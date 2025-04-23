import sklearn
import numpy as np
import tensorly as tl

from tensorlibrary.learning.active import (
    combined_strategy,
    diversity_strategy,
    cos_sim_map,
)

# get some test classification data (binary) from breast cancer dataset
from sklearn.datasets import load_breast_cancer
from sklearn.metrics.pairwise import rbf_kernel

data = load_breast_cancer()
X = data.data
y = data.target
# convert y to -1, 1
y = 2 * y - 1
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# import svm
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import time

# create train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print class distribution in train set (amount of samples per class)
print(f"Class 1: {np.sum(y_train == 1)}, Class -1: {np.sum(y_train == -1)}")

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %% jkk
X_train = X_train[:, :3]
gamma = 2
K1 = rbf_kernel(X_train, gamma=gamma)
sigma = np.sqrt(1 / (2 * gamma))
K2 = cos_sim_map(X_train, X_train, map_param=sigma, m=50)
#%% features test
def features(x_d, m: int, map_param=1.0, Ld=1.0):

    x_d = (x_d + Ld) / (2 * Ld)
    w = tl.arange(1, m + 1)
    s = (
        np.sqrt(2 * np.pi)
        * map_param
        * np.exp(-(((np.pi * w.T) / (2 * Ld)) ** 2) * map_param ** 2 / 2)
    )
    z_x = (1 / np.sqrt(Ld)) * np.sin(np.pi * np.outer(x_d, w)) * np.sqrt(s)
    return z_x


# from tensorlibrary.learning.features import features
m = 20
x_1 = X_train[0, :]
x_2 = X_train[1, :]
D = X_train.shape[1]
dot12 = 1
ZD1 = 1
ZD2 = 1
for d in range(D):
    zd_1 = features(x_1[d], m, map_param=sigma, Ld=2.0)
    zd_2 = features(x_2[d], m, map_param=sigma, Ld=2.0)
    print(zd_1.shape)
    print(zd_2.shape)
    dot12 = dot12 * np.dot(zd_1, zd_2.T)
    ZD1 = np.outer(ZD1, zd_1)
    ZD2 = np.outer(ZD2, zd_2)

ZD1 = ZD1.flatten()
ZD2 = ZD2.flatten()
print(np.dot(ZD1, ZD2))
print(dot12)


#%%
# fit a model
sigma = 0.3
model = SVC(kernel="rbf", C=1.0, gamma=sigma)
model.fit(X_train, y_train)

# get accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# get indices for combined strategy
outputs = model.decision_function(X_train)

# only on positive samples
x_pos = X_train[y_train == 1]
# indices = combined_strategy(x_pos, outputs[y_train==1], int(0.7*x_pos.shape[0]), l =0., map_param=sigma)

start_time = time.time()
indices = diversity_strategy(
    x_pos,
    int(0.5 * x_pos.shape[0]),
    sim_measure="cos",
    feature_map="rbf",
    map_param=sigma,
    m=10,
    approx=True,
    min_div_max=0.0,
)
indices2 = diversity_strategy(
    x_pos,
    int(0.5 * x_pos.shape[0]),
    sim_measure="cos",
    feature_map="rbf",
    map_param=sigma,
    approx=False,
)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")

# sort the indices
indices = np.sort(indices)
indices2 = np.sort(indices2)
print(f"Diff: {np.sum(indices != indices2)}")

# get the selected samples
X_selected = x_pos[indices]
y_selected = np.ones(X_selected.shape[0])

# add negative samples
x_neg = X_train[y_train == -1]
y_neg = -1 * np.ones(x_neg.shape[0])
X_selected = np.concatenate((X_selected, x_neg), axis=0)
y_selected = np.concatenate((y_selected, y_neg), axis=0)
# check if both classes are represented
print(f"Class 1: {np.sum(y_selected == 1)}, Class -1: {np.sum(y_selected == -1)}")

# train model on selected samples
new_model = SVC(kernel="rbf", C=1.0, gamma=sigma)
new_model.fit(X_selected, y_selected)

# get accuracy
accuracy = new_model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
