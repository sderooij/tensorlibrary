#%%
import pickle
import numpy as np
import tensorlibrary as tl 
from tensorlibrary.learning.t_krr import CPKRR
from tensorlibrary.learning.t_svm import CPSVM

with open('test_model.pkl', 'rb') as f:
    model_params = pickle.load(f)
f.close()

X = np.array(model_params['X'])
y = np.array(model_params['y'])
W = np.array(model_params['W'])
W_init = model_params['W_init']
W_init = [np.array(W_init[i]) for i in range(0,len(W_init))]

features = model_params['feature_map']
l = model_params['l']
M = model_params['M']
num_sweeps = model_params['num_sweeps']
R = model_params['R']

N = len(y)
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# model = CPKRR(
#     M=5,
#     # w_init=W_init,
#     feature_map=features,
#     reg_par=l/N,
#     num_sweeps=num_sweeps,
#     max_rank=R
#     )
model = CPSVM(
    M=2,
    w_init=W_init,
    feature_map=features,
    reg_par=l/N,
    num_sweeps=num_sweeps,
    max_rank=R
    )

model = model.fit(X, y)
accuracy = model.score(X, y)
print(accuracy)
# assert jnp.allcl

#%% 
# new_model_params= {}
# for key, value in model_params.items():
#     new_model_params[key] = np.array(value)
    
# new_model_params['W'] = model.weights_

# #%%
# with open('../data/test_model.pkl', 'wb') as f:
#     pickle.dump( new_model_params, f)
    
# f.close()