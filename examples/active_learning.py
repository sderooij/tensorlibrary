import sklearn
import numpy as np
import tensorly as tl

from tensorlibrary.learning.active import combined_strategy, diversity_strategy

# get some test classification data (binary) from breast cancer dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
# convert y to -1, 1
y = 2 * y - 1
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# import svm
from sklearn.svm import SVC
import time

# create train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print class distribution in train set (amount of samples per class)
print(f'Class 1: {np.sum(y_train == 1)}, Class -1: {np.sum(y_train == -1)}')


# fit a model
sigma = 0.3
model = SVC(kernel='rbf', C=1.0, gamma=sigma)
model.fit(X_train, y_train)

# get accuracy
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# get indices for combined strategy
outputs = model.decision_function(X_train)

# only on positive samples
x_pos = X_train[y_train == 1]
# indices = combined_strategy(x_pos, outputs[y_train==1], int(0.7*x_pos.shape[0]), l =0., map_param=sigma)

start_time = time.time()
indices = diversity_strategy(x_pos, int(0.5*x_pos.shape[0]), sim_measure='cos', feature_map='rbf', map_param=sigma,
                             m=10, approx=True, min_div_max=0.0)
indices2 = diversity_strategy(x_pos, int(0.5*x_pos.shape[0]), sim_measure='cos', feature_map='rbf', map_param=sigma,
                             approx=False)
end_time = time.time()
print(f'Time taken: {end_time - start_time}')

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
print(f'Class 1: {np.sum(y_selected == 1)}, Class -1: {np.sum(y_selected == -1)}')

# train model on selected samples
new_model = SVC(kernel='rbf', C=1.0, gamma=sigma)
new_model.fit(X_selected, y_selected)

# get accuracy
accuracy = new_model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')