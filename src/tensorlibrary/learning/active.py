"""
Active Learning with tensor kernel machines
"""

# library imports
import tensorly as tl
from sklearn.metrics.pairwise import rbf_kernel
import random

# local imports
from tensorlibrary.learning.features import features
from tensorlibrary.linalg.linalg import dot_r1


def cos_sim_feat(phi_x: list, phi_y: list) -> float:
    """
    Determine the cosine similarity between two mapped features.
    Args:
        phi_x: mapped features x, d-dimensional list (1D tensor)
        phi_y: mapped features y, d-dimensional list (1D tensor)

    Returns:
        scalar: cosine similarity measure between the two mapped features

    """
    norm_x = tl.sqrt(dot_r1(phi_x, phi_x))
    norm_y = tl.sqrt(dot_r1(phi_y, phi_y))
    return dot_r1(phi_x, phi_y) / (norm_x * norm_y)


def uncertainty_samp(outputs, n_samples=-1, thresh=-1):
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
    # compute diversity measure to the first sample
    div = tl.zeros((x_feat.shape[0], max_samples-1))

    for k in range(0, max_samples-1):
        if not approx:
            if feature_map == 'rbf' and sim_measure == 'cos':
                div[:, k] = (rbf_kernel(x_feat, x_feat[indices[k], :].reshape(1, -1), gamma=map_param)).reshape(-1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # div[indices[k], k] = 0  # for max to work
        # take max over the columns o
        div_max = tl.max(div, axis=1)  # results in N_feats x 1
        if tl.all(div_max[~indices[k]] < min_div_max):
            indices = indices[:k]
            break

        indices[k+1] = tl.argmin(l*outputs + (1-l)*div_max)
        outputs[indices[k+1]] = 1e10 # set high to avoid re-selection

    return indices




