"""
Active Learning with tensor kernel machines
"""

# library imports
import tensorly as tl
from sklearn.metrics.pairwise import rbf_kernel
import random

from numba import njit
import numpy as np

# local imports
from tensorlibrary.learning.features import features
from tensorlibrary.linalg.linalg import dot_r1


# def cos_sim_feat(phi_x: list, phi_y: list) -> float:
#     """
#     Determine the cosine similarity between two mapped features.
#     Args:
#         phi_x: mapped features x, d-dimensional list (1D tensor)
#         phi_y: mapped features y, d-dimensional list (1D tensor)
#
#     Returns:
#         scalar: cosine similarity measure between the two mapped features
#
#     """
#     norm_x = tl.sqrt(dot_r1(phi_x, phi_x))
#     norm_y = tl.sqrt(dot_r1(phi_y, phi_y))
#     return dot_r1(phi_x, phi_y) / (norm_x * norm_y)

def cos_sim_map(x, y, m=10, *, feature_map='rbf', map_param=1.0):
    # x (N_x x d), y (N_y x d)
    # compute the cosine similarity between samples
    # TODO: precompute the norms

    assert x.shape[1] == y.shape[1]
    D = x.shape[1]
    N_x = x.shape[0]
    N_y = y.shape[0]
    K = tl.ones((N_x, N_y)) # initialize "kernel" matrix Phi^T Phi

    norm_x = tl.ones((N_x, 1))
    norm_y = tl.ones((N_y, 1))
    ones_m = tl.ones((m, 1))
    for d in range(D):
        phi_x = features(x[:, d], m=m, feature_map=feature_map, map_param=map_param)
        phi_y = features(y[:, d], m=m, feature_map=feature_map, map_param=map_param)
        K = (phi_x @ phi_y.T) * K

        norm_x *= (phi_x * phi_x) @ ones_m
        norm_y *= (phi_y * phi_y) @ ones_m

    # take sqrt to get norms
    norm_x = tl.sqrt(norm_x)
    norm_y = tl.sqrt(norm_y)
    # divide rows by norm_x
    K = K / (norm_x @ norm_y.T)
    return K


def uncertainty_strategy(outputs, n_samples=-1, thresh=-1):
    """
    Perform uncertainty sampling to select the most uncertain samples.
    Either select the n_samples most uncertain samples or select samples below a certain threshold.
    Args:
        outputs: model outputs
        n_samples: number of samples to select
        thresh: threshold for uncertainty sampling

    Returns:
        indices: indices of the most uncertain samples
    """
    if n_samples == -1 and thresh == -1:
        raise ValueError("Either n_samples or thresh must be set")
    elif n_samples != -1 and thresh != -1:
        raise ValueError("Only one of n_samples or thresh must be set")
    elif n_samples != -1:
        indices = tl.argsort(outputs, axis=0)[:n_samples]
    else:
        indices = tl.where(tl.abs(outputs) < thresh)
    return indices


def combined_strategy(x_feat, outputs, max_samples, l=0.5, sim_measure='cos', feature_map='rbf', map_param=1.0, \
    approx=False, min_div_max=0.):
    """
    Perform combined strategy for active learning.
    Args:
        x_feat: features of the data
        outputs: model outputs
        max_samples: maximum number of samples to select for the batch
        l: trade-off parameter between uncertainty and diversity (0.5 by default)
        sim_measure: similarity measure, default is cosine similarity: 'cos'
        feature_map: feature map to use, default is rbf kernel: 'rbf'
        map_param: parameter for the feature map (or kernel function), default is 1.0
        approx: whether to use the approximate feature map instead of kernel function
        min_div_max: minimum value the maximum diversity measure, break if all values are below this threshold

    Returns:
        indices: indices of the most uncertain samples
    """
    outputs = tl.abs(outputs)
    # initialize indices
    indices = tl.zeros(max_samples, dtype=int)
    # select first sample as most uncertain (closest to boundary)
    if l != 0:
        indices[0] = tl.argmin(outputs)
    else: # choose at random
        indices[0] = random.randint(0, x_feat.shape[0])

    outputs[indices[0]] = 1e10  # set high to avoid re-selection
    # compute similarity measure to the first sample
    sim = tl.zeros((x_feat.shape[0], max_samples-1))

    for k in range(0, max_samples-1):
        if not approx:
            if feature_map == 'rbf' and sim_measure == 'cos':
                sim[:, k] = (rbf_kernel(x_feat, x_feat[indices[k], :].reshape(1, -1), gamma=map_param)).reshape(-1)
            else: # TODO: use the feature map function.
                raise NotImplementedError
        else:
            raise NotImplementedError

        # div[indices[k], k] = 0  # for max to work
        # take max over the columns o
        sim_max = tl.max(sim, axis=1)  # results in N_feats x 1
        if tl.all(sim_max[~indices[k]] < min_div_max):
            indices = indices[:k]
            break

        indices[k+1] = tl.argmin(l*outputs + (1-l)*sim_max)
        outputs[indices[k+1]] = 1e10 # set high to avoid re-selection

    return indices


def diversity_strategy(x_feat, max_samples, sim_measure='cos', feature_map='rbf', map_param=1.0, \
    approx=False, min_div_max=0.):
    """
    Perform combined strategy for active learning.
    Args:
        x_feat: features of the data
        outputs: model outputs
        max_samples: maximum number of samples to select for the batch
        l: trade-off parameter between uncertainty and diversity (0.5 by default)
        sim_measure: similarity measure, default is cosine similarity: 'cos'
        feature_map: feature map to use, default is rbf kernel: 'rbf'
        map_param: parameter for the feature map (or kernel function), default is 1.0
        approx: whether to use the approximate feature map instead of kernel function
        min_div_max: minimum value the maximum diversity measure, break if all values are below this threshold

    Returns:
        indices: indices of the most uncertain samples
    """
    # initialize indices
    indices = np.zeros(max_samples, dtype=np.int32)
    indices[0] = random.randint(0, x_feat.shape[0])
    # compute diversity measure to the first sample
    sim = np.zeros((x_feat.shape[0], max_samples-1))
    if approx:
        # phi_x = features(x_feat,  feature_map, map_param)
        raise NotImplementedError

    for k in range(0, max_samples-1):
        if not approx:
            if feature_map == 'rbf' and sim_measure == 'cos': # cosine similarity for rbf kernel is the same as rbf kernel
                # sim[:, k] = (rbf_kernel(x_feat, x_feat[indices[k], :].reshape(1, -1), gamma=map_param)).reshape(-1)
                sim[:,k] = rbf(x_feat, x_feat[indices[k], :], map_param)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError   #TODO: use the feature map function.
        # take max over the columns
        sim_max = np.max(sim, axis=1)  # results in N_feats x 1
        if np.all(sim_max[~indices[k]] < min_div_max):
            indices = indices[:k]
            break

        indices[k+1] = np.argmin(sim_max)

    return indices


def rbf(x,y, sigma):
    """
    Compute the RBF kernel function.
    Args:
        x: first input
        y: second input
        sigma: kernel parameter

    Returns:
        scalar: kernel function value
    """
    return np.exp(-0.5*np.linalg.norm(x - y, axis=1, ord=2)**2 / sigma**2)