from tensorlibrary import TensorTrain, TTmatrix
from tensorlibrary.learning.tt_krr import tt_krr, tt_krr_predict
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

# scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# %% train the model
m = 3
tt = tt_krr(
    X_train, y_train, m=m, ranks=1, reg_par=0, num_sweeps=50, kernel_type="poly"
)
# %% predict the labels of the test set
y_pred, _ = tt_krr_predict(tt, X_train, m=m, reg_par=0)
# %% calculate the accuracy
acc = np.sum(y_pred == y_train) / len(y_train)
print("Accuracy tt-krr: ", acc)

# %% compare to sklearn
from sklearn.svm import SVC

clf = SVC(kernel="poly", gamma="scale")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
acc = np.sum(y_pred == y_train) / len(y_train)
print("Accuracy SVM: ", acc)
