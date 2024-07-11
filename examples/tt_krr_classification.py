from tensorlibrary import TensorTrain, TTmatrix
from tensorlibrary.learning.t_krr import CPKRR, TTKRR
import numpy as np
import tensorly as tl

# %% import data to classify
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = load_breast_cancer()
X = data.data
y = data.target
# change the labels to -1 and 1
y[y == 0] = -1
# make data imbalanced
idx_pos = np.where(y == 1)[0]
# randomly select 50% of the positive class
np.random.seed(42)
idx_pos = np.random.choice(idx_pos, int(len(idx_pos) * 0.3), replace=False)
idx_neg = np.where(y == -1)[0]
X = np.concatenate((X[idx_pos], X[idx_neg]), axis=0)
y = np.concatenate((y[idx_pos], y[idx_neg]), axis=0)
# %% split data into train and test set
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    make_scorer,
    accuracy_score,
)

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# get class counts
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
#
# # # %% save to mat file
# import scipy.io as sio
# # sio.savemat("../data/breast_cancer.mat", {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test})
# data = sio.loadmat("../data/breast_cancer.mat") # load the data
# X_train = data["X_train"]
# y_train = data["y_train"].ravel()
# X_test = data["X_test"]
# y_test = data["y_test"].ravel()
# # X_train = X_train[:, :5]


# %% lengtscale


# %% train the model
m = 15
rank = 30
random_state = 42
lengthscale = 0.3
parameters = {"max_rank": [20, 30]}

scoring = {"F1": make_scorer(f1_score)}
# w0 = sio.loadmat("../data/init_val.mat")["W"][0]
cpkrr = CPKRR(
    M=m,
    num_sweeps=20,
    reg_par=1e-3,
    feature_map="rbf",
    map_param=lengthscale,
    random_state=random_state,
    class_weight="balanced",
    max_rank=30,
)
# model = GridSearchCV(
#     cpkrr, parameters, cv=5, n_jobs=1, verbose=1, scoring=make_scorer(accuracy_score)
# )
model = cpkrr
# %% Train
model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy CP-KRR: ", acc)

# %% TT-KRR
# tt = tt_krr(
#     X_train,
#     y_train,
#     m=m,
#     ranks=2,
#     reg_par=0,
#     num_sweeps=10,
#     feature_map="rbf",
#     map_param=lengthscale,
# )
# y_pred = tt_krr_predict(
#     tt, X_test, m=m, reg_par=0, feature_map="rbf", map_param=lengthscale
# )
tt_krr = TTKRR(
    M=3,
    num_sweeps=2,
    reg_par=0,
    feature_map="rbf",
    map_param=lengthscale,
    random_state=random_state,
    max_rank=30
)
tt_krr = tt_krr.fit(X_train, y_train)
y_pred = tt_krr.predict(X_test)
acc = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy TT-KRR: ", acc)

# %% compare to sklearn
from sklearn.svm import SVC

clf = SVC(kernel="rbf", gamma=lengthscale, class_weight=None)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy SVM: ", acc)


