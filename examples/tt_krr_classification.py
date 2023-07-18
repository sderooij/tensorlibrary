from tensorlibrary import TensorTrain, TTmatrix
from tensorlibrary.learning.t_krr import CPKRR
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

# %% split data into train and test set
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
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
m = 50
rank = 20
random_state = 42
lengthscale = 0.1

# tt = tt_krr(
#     X_train, y_train, m=m, ranks=3, reg_par=0, num_sweeps=50, feature_map="rbf", map_param=0.01
# )
# w0 = sio.loadmat("../data/init_val.mat")["W"][0]
model = CPKRR(M=m, num_sweeps=10, reg_par=1e-4, max_rank=rank, feature_map="rbf", map_param=lengthscale,
              random_state=random_state)
#%% Train
model = model.fit(X_train, y_train)
# %% predict the labels of the test set
y_pred = model.predict(X_test)

# %% calculate the accuracy
acc = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy CP-KRR: ", acc)

# %% compare to sklearn
from sklearn.svm import SVC

clf = SVC(kernel="rbf", gamma=lengthscale)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
acc = np.sum(y_pred == y_train) / len(y_train)
print("Accuracy SVM: ", acc)
