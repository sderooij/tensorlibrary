"""
Active Learning with tensor kernel machines
"""

# library imports
import tensorly as tl
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, ClassifierMixin, clone, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import random
from functools import partial, reduce
from copy import deepcopy

from numba import njit
import numpy as np

# local imports
from tensorlibrary.learning.features import features
from tensorlibrary.linalg.linalg import dot_r1
from abc import abstractmethod, ABCMeta


class BaseActiveLearnClassifier(BaseEstimator, ClassifierMixin, metaclass=ABCMeta):
    def __init__(self, init_model, n_samples, strategy, al_parameters=None, model_type="SVC", model_params=None,
                 groups=None,
                 ratio=None, random_state=42, random_init=False):

        if al_parameters is None:
            al_parameters = {}
        self.init_model =init_model
        self.n_samples = n_samples
        self.strategy = strategy
        self.al_parameters = al_parameters
        self.model_params = model_params.copy() if model_params is not None else {}
        self.labeling_count = 0
        self.model_type = model_type
        self.groups = groups
        self.ratio = ratio
        self.random_state = random_state
        self.X = None
        self.y = None
        self.train_indices = None
        self.test_indices = None
        self.sample_indices_ = None
        self.model = None
        self.random_init = random_init
        self._initiate_model()

    # @property
    # def test_indices(self):
    #     return np.setdiff1d(np.arange(len(self.y)), self.train_indices)

    @property
    def included_groups(self):
        ug = np.unique(self.groups[self.train_indices])
        return ug[~np.isnan(ug)]

    def _initiate_model(self):
        # TODO: keep the scaler the same
        self.model = clone(self.init_model)
        # if self.random_init:        # random initialization
        #     if isinstance(self.init_model, ClassifierMixin):
        #         self.model.set_params(**self.model_params)
        #     elif isinstance(self.init_model, Pipeline):
        #         self.model['clf'].set_params(**self.model_params)
        #     return self
        if self.model_type == "CPKRR":
            self.model_params['w_init'] = self.init_model['clf'].weights_
            if self.random_init:
                self.model_params['random_init'] = True
        else:
            raise ValueError("Invalid model type")

        if isinstance(self.init_model, ClassifierMixin):
            # if self.model_type == "CPKRR":
            #     self.model_params['w_init'] = self.init_model.weights_
            #     if self.random_init:
            #         self.model_params['random_init'] = True
            self.model.set_params(**self.model_params)
        elif isinstance(self.init_model, Pipeline):
            # if self.model_type == "CPKRR":
            #     self.model_params['w_init'] = self.init_model['clf'].weights_
            #     if self.random_init:
            #         self.model_params['random_init'] = True
            self.model['clf'].set_params(**self.model_params)
            self.model.named_steps['scaler'] = deepcopy(self.init_model.named_steps['scaler'])

        return self

    def _update_model_params(self, params):
        if isinstance(self.init_model, BaseEstimator):
            self.model.set_params(**params)
        elif isinstance(self.init_model, Pipeline):
            self.model['clf'].set_params(**params)


    def _adapt_params(self, strategy, **kwargs):
        if strategy == 'uncertainty':
            self.al_parameters['l'] = 1
            self.algorithm = partial(combined_strategy, **self.al_parameters)
        elif strategy == 'combined':
            self.algorithm = partial(combined_strategy, **self.al_parameters)
        else:
            raise ValueError("Invalid strategy")

    @abstractmethod
    def select_samples(self, X, y=None, **kwargs):
        return self

    @abstractmethod
    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)


class ActiveLearnClassifier(BaseActiveLearnClassifier):
    def __init__(self, init_model, n_samples, strategy, batch_size=2048, break_at_pos=False, min_n_samples=0,
                 similarity='rbf', pos_only=False, al_parameters=None, model_type="SVC", model_params=None, groups=None,
                 ratio=None, random_state=42, random_init=False):
        super().__init__(init_model, n_samples, strategy, al_parameters, model_type, model_params, groups, ratio,
                         random_state, random_init)

        self.batch_size = batch_size
        self.break_at_pos = break_at_pos
        self.min_n_samples = min_n_samples
        self.pos_only = pos_only
        self.similarity = similarity
        if self.strategy == 'uncertainty':
            self.al_parameters['l'] = 1
            self.algorithm = partial(combined_strategy, **self.al_parameters)
        elif self.strategy == 'combined':
            self.algorithm = partial(combined_strategy, **self.al_parameters)
        else:
            raise ValueError("Invalid strategy")

    def select_samples(self, X, y=None,*, model_outputs=None):
        """
                Select the most informative samples. If batch_size is set, select samples in batches.

                Args:
                    X: input X
                    model_outputs: model outputs
                    y: correct labels, can only be None if break_at_pos is False

                Returns:
                    indices: indices of the most uncertain samples
                """
        self.X = X
        self.y = y
        n_input = len(y)

        if self.pos_only:
            pos_indices = np.where(model_outputs > 0)[0]
            X = X[pos_indices]
            model_outputs = model_outputs[pos_indices]
            y = y[pos_indices]

        if self.batch_size is None:
            self.train_indices = self.algorithm(X, model_outputs, self.n_samples, break_at_pos=self.break_at_pos,
                                          labels=y, min_n_samples=self.min_n_samples)
        else:
            total_samples = len(model_outputs)
            indices = np.arange(0, total_samples)
            batch_indices = [indices[i:i + self.batch_size] for i in range(0, total_samples, self.batch_size)]
            n_batches = len(batch_indices)
            n_samples_per_batch = int(np.ceil(self.n_samples / n_batches))
            # n_samples_last_batch = np.min(len(batch_indices[-1]), self.n_samples - n_samples_per_batch * (n_batches -
            #                                                                                               1))
            indices_select = np.empty((0,), dtype=int)
            min_selected_flag = False
            total_selected = 0
            for k, idx in enumerate(batch_indices):
                al_indices_batch = self.algorithm(X[idx, :], model_outputs[idx],
                                                  n_samples_per_batch,
                                                  break_at_pos=self.break_at_pos, labels=y[idx],
                                                  min_n_samples=self.min_n_samples * (1 - min_selected_flag),
                                                  prev_batch=self.X[indices_select, :])

                indices_select = np.append(indices_select, idx[al_indices_batch])
                total_selected += len(al_indices_batch)
                if not min_selected_flag:
                    min_selected_flag = total_selected >= self.min_n_samples
                if self.break_at_pos:
                    if np.any(y[indices_select] == 1) and min_selected_flag:
                        break
                if total_selected >= self.n_samples:
                    break

            self.train_indices = indices_select

        self.labeling_count = len(self.train_indices)
        if self.pos_only:
            self.train_indices = pos_indices[self.indices]

        if self.groups is not None:
            add_groups = AddGroupsToData(self.groups)
            self.train_indices = add_groups.fit_transform(self.train_indices)
        self.test_indices = np.setdiff1d(np.arange(n_input), self.train_indices)
        return self

    def fit(self, X, y, *, model_outputs=None, groups=None):
        """
        Fit the model with the selected samples.
        Args:
            X: input data
            y: labels
            model_outputs: model outputs, of init_model for the input data X, if None, it will be computed. Default is None
            groups: groups for transfer active learning. Default is None. If not None, all samples in the selected
                    groups will be added to the training data.

        Returns:
            fitted model
        """
        if groups is not None:
            self.groups = groups

        if model_outputs is None:
            model_outputs = self.init_model.decision_function(X)

        self.select_samples(X, y, model_outputs=model_outputs)

        # self.X = X[self.train_indices]
        # self.y = y[self.train_indices]

        if (self.ratio is not None and (sum(self.y[self.train_indices]==-1)/sum(self.y[self.train_indices]==1) >
                self.ratio)):
            rus = RandomUnderSampler(sampling_strategy=1/self.ratio, random_state=self.random_state)
            self.train_indices, _ = rus.fit_resample(self.train_indices.reshape(-1,1), self.y[self.train_indices])
            self.train_indices = self.train_indices.flatten()

        self.X = X[self.train_indices, :]
        self.y = y[self.train_indices]

        # retrain
        self.model.fit(self.X, self.y)

        return self


class RandomSampleClassifier(BaseActiveLearnClassifier):
    def __init__(self, init_model, n_samples, strategy, batch_size=2048, al_parameters=None, model_type="SVC",
                 model_params=None, groups=None,
                 ratio=None, random_state=42, random_init=False):
        super().__init__(init_model, n_samples, strategy, al_parameters, model_type, model_params, groups, ratio,
                         random_state, random_init)
        self.batch_size = batch_size

    def select_samples(self, X, y=None, **kwargs):

        if self.strategy == "random":
            if self.ratio is None:
                self.ratio = 1
            num_pos = self.n_samples // (1 + self.ratio)
            num_neg = self.n_samples - num_pos
            indices = np.arange(len(y))
            rus = RandomUnderSampler(sampling_strategy={1: num_pos, -1: num_neg}, random_state=self.random_state)
            self.train_indices, _ = rus.fit_resample(indices.reshape(-1,1), y)
        elif self.strategy == "LOGI":   # leave one group in
            # select one group
            #TODO: change to crossvalidation
            np.random.seed(self.random_state)
            group = np.random.choice(np.unique(self.groups[~np.isnan(self.groups)]))
            pos_indices = np.where(self.groups == group)[0]
            assert np.all(y[pos_indices] == 1)
            num_pos = len(pos_indices)
            if self.ratio is None:
                num_neg = self.n_samples - num_pos
            else:
                num_neg = num_pos * self.ratio
            neg_indices = np.where(y==-1)[0]
            neg_indices = np.random.choice(neg_indices, num_neg, replace=False)
            self.train_indices = np.concatenate([pos_indices, neg_indices])
            # shuffle
            np.random.shuffle(self.train_indices)
        else:
            raise ValueError("Invalid strategy")

        self.labeling_count = len(self.train_indices)
        self.test_indices = np.setdiff1d(np.arange(len(y)), self.train_indices)
        return self

    def fit(self, X, y, *, groups=None):
        """
        Fit the model with the selected samples.
        Args:
            X:
            y:

        Returns:
            fitted model
        """
        if groups is not None:
            self.groups = groups

        self.select_samples(X, y)

        self.X = X[self.train_indices, :]
        self.y = y[self.train_indices]

        # retrain
        self.model.fit(self.X, self.y)

        return self


class AddGroupsToData(BaseEstimator, TransformerMixin):
    """
        For transfer active learning with groups, we need to add all of the (positive) group data to the training
        data.
    """
    def __init__(self, groups):
        self.groups = groups
        self.included_groups = None

    def fit(self, idx):
        self.included_groups = np.unique(self.groups[idx])
        return self

    def transform(self, idx):
        added_idx = [idx]
        for i, group in enumerate(self.included_groups):
            added_idx.append(np.where(self.groups == group)[0])
        idx = reduce(np.union1d, added_idx)

        return idx





def _compute_similarity(self, x, y, **kwargs):
    """
    Compute the similarity between the samples x and y.
    Args:
        x: matrix of N_x x D samples
        y: matrix of N_y x D samples, where N_y is usually 1
        **kwargs: additional arguments for the similarity measure. For cosine similarity of feature-maps,
        the following are
        required: m: number of basis functions or order of polynomial, feature_map: feature map to use,
        map_param: parameter for the feature map (or kernel function), Ld: interval
        for rbf kernel, gamma: kernel parameter


    Returns:
        similarity: similarity matrix of N_x x N_y

    """
    if self.similarity == 'rbf':
        return rbf_kernel(x, y, **kwargs)
    elif self.similarity == 'cos_feat_map':
        return cos_sim_map(x, y, **kwargs)


#
# class ActiveLearner:
#     def __init__(self, data, outputs, n_samples, strategy, *, parameters={}, batch_size=None, break_at_pos=False,
#                  labels=None,
#                  min_n_samples=50, pos_only=False):
#
#         self.data = data
#         self.outputs = outputs
#         self.n_samples = n_samples
#         self.strategy = strategy
#         self.batch_size = batch_size
#         self.algorithm = None
#         self.indices = None
#         self.min_n_samples = min_n_samples
#         self.pos_only = pos_only
#         self.parameters = parameters
#         if self.strategy == 'uncertainty':
#             self.parameters['l'] = 1
#             self.algorithm = partial(combined_strategy, **self.parameters)
#         elif self.strategy == 'combined':
#             self.algorithm = partial(combined_strategy, **self.parameters)
#         else:
#             raise ValueError("Invalid strategy")
#
#         self.labels = labels
#         self.break_at_pos = break_at_pos
#
#     def select_samples(self, **kwargs):
#         """
#         Select the most informative samples. If batch_size is set, select samples in batches.
#
#         Args:
#             **kwargs: Depends on the strategy used.
#             For combined strategy, the following are required:
#                 l: trade-off parameter between uncertainty and diversity (0.5 by default)
#                 sim_measure: similarity measure, default is cosine similarity: 'cos'
#                 feature_map: feature map to use, default is rbf kernel: 'rbf'
#
#             For diversity strategy, the following are required:
#                 x_feat: features of the X
#                 max_samples: maximum number of samples to select for the batch
#                 sim_measure: similarity measure, default is cosine similarity: 'cos'
#                 feature_map: feature map to use, default is rbf kernel: 'rbf'
#                 map_param: parameter for the feature map (or kernel function), default is 1.0
#                 m: number of basis functions or order of polynomial
#
#         Returns:
#             indices: indices of the most uncertain samples
#         """
#         if self.pos_only:
#             pos_indices = np.where(self.outputs > 0)[0]
#             self.data = self.data[pos_indices]
#             self.outputs = self.outputs[pos_indices]
#             self.labels = self.labels[pos_indices]
#
#         if self.batch_size is None:
#             self.indices = self.algorithm(self.data, self.outputs, self.n_samples, break_at_pos=self.break_at_pos,
#                                           labels=self.labels, min_n_samples=self.min_n_samples, **kwargs)
#
#         else:
#             total_samples = len(self.outputs)
#             # n_batches = total_samples // self.batch_size
#             # n_samples_per_batch = self.n_samples // n_batches
#             # n_samples_last_batch = self.n_samples - n_samples_per_batch * (n_batches - 1)
#             indices = np.arange(0, total_samples)
#             batch_indices = [indices[i:i + self.batch_size] for i in range(0, total_samples, self.batch_size)]
#             n_batches = len(batch_indices)
#             n_samples_per_batch = np.ceil(self.n_samples / n_batches)
#             # n_samples_last_batch = np.min(len(batch_indices[-1]), self.n_samples - n_samples_per_batch * (n_batches -
#             #                                                                                               1))
#             indices_select = np.array([])
#             min_selected_flag = False
#             total_selected = 0
#             for k, idx in enumerate(batch_indices):
#                 al_indices_batch = self.algorithm(self.data[idx, :], self.outputs[idx],
#                                                   n_samples_per_batch,
#                                                   break_at_pos=self.break_at_pos, labels=self.labels[idx],
#                                                   min_n_samples=self.min_n_samples * (1 - min_selected_flag),
#                                                   prev_batch=self.data[indices_select, :], **kwargs)
#
#                 indices_select = np.append(indices_select, idx[al_indices_batch])
#                 total_selected += len(al_indices_batch)
#                 if not min_selected_flag:
#                     min_selected_flag = total_selected >= self.min_n_samples
#                 if self.break_at_pos:
#                     if np.any(self.labels[indices_select] == 1) and min_selected_flag:
#                         break
#                 if total_selected >= self.n_samples:
#                     break
#
#             # for k in range(n_batches):
#             #     idx_start = k * self.batch_size
#             #     if k == n_batches - 1: # last batch
#             #         cur_indices = indices[idx_start:]
#             #         sel_indices = self.algorithm(self.X[idx_start:, :], self.outputs[idx_start:],
#             #                                      n_samples_last_batch,
#             #                                      break_at_pos=self.break_at_pos, labels=self.labels[idx_start:],
#             #                                      min_n_samples=self.min_n_samples*(1-min_selected_flag), **kwargs)
#             #
#             #     else:
#             #         idx_end = (k + 1) * self.batch_size
#             #         cur_indices = indices[idx_start:idx_end]
#             #         sel_indices = self.algorithm(self.X[idx_start:idx_end, :], self.outputs[idx_start:idx_end],
#             #                                      n_samples_per_batch,
#             #                                       break_at_pos=self.break_at_pos, labels=self.labels[idx_start:idx_end],
#             #                                      min_n_samples=self.min_n_samples*(1-min_selected_flag), **kwargs)
#             #
#             #     indices_select.append(cur_indices[sel_indices])
#             #     if not min_selected_flag:
#             #         min_selected_flag = sum(len(arr) for arr in indices_select) > self.min_n_samples
#             #     if self.break_at_pos:
#             #         # check where indices are positive
#             #         if np.any(self.labels[indices_select[-1]] == 1) and min_selected_flag:
#             #             break
#
#             self.indices = tl.concatenate(indices_select)
#
#         if self.pos_only:
#             self.indices = pos_indices[self.indices]
#
#         return self.indices


def cos_sim_map(x, y, m=10, *, feature_map='rbf', map_param=1., Ld=1.):
    # x (N_x x d), y (N_y x d)
    # compute the cosine similarity between samples
    # TODO: precompute the norms

    assert x.shape[1] == y.shape[1]
    D = x.shape[1]
    N_x = x.shape[0]
    N_y = y.shape[0]
    K = tl.ones((N_x, N_y))  # initialize "kernel" matrix Phi^T Phi

    norm_x = tl.ones((N_x, 1))
    norm_y = tl.ones((N_y, 1))
    ones_m = tl.ones((m, 1))
    for d in range(D):
        phi_x = features(x[:, d], m=m, feature_map=feature_map, map_param=map_param, Ld=Ld)
        phi_y = features(y[:, d], m=m, feature_map=feature_map, map_param=map_param, Ld=Ld)
        K *= (phi_x @ phi_y.T)

        norm_x *= (phi_x * phi_x) @ ones_m
        norm_y *= (phi_y * phi_y) @ ones_m

    # take sqrt to get norms
    norm_x = tl.sqrt(norm_x)
    norm_y = tl.sqrt(norm_y)
    # divide rows by norm_x
    K /= (norm_x @ norm_y.T)
    return K


def uncertainty_strategy(outputs, n_samples=-1, thresh=-1, *, break_at_pos=False, labels=None):
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
        indices = tl.argsort(tl.abs(outputs), axis=0)[:n_samples]
    else:
        indices = tl.where(tl.abs(outputs) < thresh)

    if break_at_pos:
        # check where indices are positive
        ind_pos = indices[labels[indices] == 1]
        if len(ind_pos) > 0:
            indices = indices[:ind_pos[0]]

    return indices


def combined_strategy(x_feat, outputs, max_samples, l=0.5, m=10, sim_measure='cos', feature_map='rbf', map_param=1.0, \
                      approx=False, max_min_sim=1., break_at_pos=False, labels=None, min_n_samples=50, prev_batch=None):
    """
    Perform combined strategy for active learning.
    Args:
        x_feat: features of the X
        outputs: model outputs
        max_samples: maximum number of samples to select for the batch
        l: trade-off parameter between uncertainty and diversity (0.5 by default)
        sim_measure: similarity measure, default is cosine similarity: 'cos'
        feature_map: feature map to use, default is rbf kernel: 'rbf'
        map_param: parameter for the feature map (or kernel function), default is 1.0
        approx: whether to use the approximate feature map instead of kernel function
        max_min_sim: minimum value the maximum diversity measure, break if all values are below this threshold

    Returns:
        indices: indices of the most uncertain samples
    """
    if l == 1:
        indices = uncertainty_strategy(outputs, n_samples=max_samples, break_at_pos=break_at_pos, labels=labels)
        return indices
    outputs = tl.abs(outputs)
    if prev_batch is not None and prev_batch.shape[0] > 0:
        len_batch = prev_batch.shape[0]
        max_samples = len_batch + max_samples
        x_feat = np.concatenate([prev_batch, x_feat], axis=0)
        indices = tl.zeros(max_samples, dtype=int)
        indices[:len_batch] = np.arange(len_batch)
        kstart = len_batch
        outputs = np.concatenate([np.ones(len_batch)*1e10, outputs])
        if break_at_pos:
            labels = np.concatenate([-1*np.ones(len_batch), labels])
    else:
        # initialize indices
        indices = tl.zeros(max_samples, dtype=int) # set high to avoid re-selection and to keep the indices
        # compute similarity measure to the first sample
        kstart = 0

    sim = tl.zeros((x_feat.shape[0], max_samples - 1))
    # select first sample as most uncertain (closest to boundary)
    if l != 0:
        indices[kstart] = tl.argmin(outputs)
    else:  # choose at random
        indices[kstart] = random.randint(0, x_feat.shape[0])

    outputs[indices[kstart]] = 1e10
    breaktime = False

    for k in range(kstart, max_samples - 1):
        # calculate the similarity measure for the k-th sample to add to similarity matrix (kernel matrix)
        if not approx:
            if feature_map == 'rbf':
                sim[:, k] = (rbf_kernel(x_feat, x_feat[indices[k], :].reshape(1, -1), gamma=map_param)).reshape(-1)
            else:
                raise NotImplementedError
        else:
            sim[:, k] = cos_sim_map(x_feat, x_feat[indices[k], :].reshape(1, -1), m=m, feature_map=feature_map,
                                    map_param=map_param).reshape(-1)

        # div[indices[k], k] = 0  # for max to work
        # take max over the columns o
        sim_max = tl.max(sim, axis=1)  # results in N_feats x 1
        if tl.min(np.delete(sim_max, indices[:k + 1])) > max_min_sim:
            indices = indices[:k + 1]
            break

        indices[k + 1] = tl.argmin(l * outputs + (1 - l) * sim_max)
        if break_at_pos:
            if tl.any(labels[indices[:k + 2]] == 1):
                breaktime = True
            if breaktime and k >= min_n_samples - 1:
                indices = indices[:k + 2]
                break

        outputs[indices[k + 1]] = 1e10  # set high to avoid re-selection

    if prev_batch is not None and prev_batch.shape[0] > 0:
        indices = indices[len_batch:] - len_batch

    if np.any(indices > x_feat.shape[0]):
        print("Error")
    return indices




# def diversity_strategy(x_feat, max_samples, sim_measure='cos', feature_map='rbf', map_param=1.0, m=10, \
#     approx=False, min_div_max=0., break_at_pos=False, labels=None, min_n_samples=50):
#     """
#     Perform combined strategy for active learning.
#     Args:
#         x_feat: features of the X
#         outputs: model outputs
#         max_samples: maximum number of samples to select for the batch
#         l: trade-off parameter between uncertainty and diversity (0.5 by default)
#         sim_measure: similarity measure, default is cosine similarity: 'cos'
#         feature_map: feature map to use, default is rbf kernel: 'rbf'
#         map_param: parameter for the feature map (or kernel function), default is 1.0
#         approx: whether to use the approximate feature map instead of kernel function
#         min_div_max: minimum value the maximum diversity measure, break if all values are below this threshold
#
#     Returns:
#         indices: indices of the most uncertain samples
#     """
#     # initialize indices
#     indices = np.zeros(max_samples, dtype=np.int32)
#     indices[0] = random.randint(0, x_feat.shape[0])
#     # compute diversity measure to the first sample
#     sim = np.zeros((x_feat.shape[0], max_samples-1))
#     breaktime = False
#
#     for k in range(0, max_samples-1):
#         if not approx:
#             if feature_map == 'rbf' and sim_measure == 'cos': # cosine similarity for rbf kernel is the same as rbf kernel
#                 # sim[:, k] = (rbf_kernel(x_feat, x_feat[indices[k], :].reshape(1, -1), gamma=map_param)).reshape(-1)
#                 sim[:,k] = rbf(x_feat, x_feat[indices[k], :], map_param)
#             else:
#                 raise NotImplementedError
#         else:
#             sim[:, k] = cos_sim_map(x_feat, x_feat[indices[k], :].reshape(1, -1), m=m, feature_map=feature_map,
#                                     map_param=map_param).reshape(-1)
#         # take max over the columns
#         sim_max = np.max(sim, axis=1)  # results in N_feats x 1
#         if np.all(sim_max[~indices[k]] < min_div_max):
#             indices = indices[:k]
#             break
#
#         indices[k+1] = np.argmin(sim_max)
#         if break_at_pos:
#             if tl.any(labels[indices[k]] == 1):
#                 breaktime = True
#             if breaktime and sum(indices > 0) > min_n_samples:
#                 indices = indices[:k + 2]
#                 break
#
#     return indices



def rbf(x, y, sigma):
    """
    Compute the RBF kernel function.
    Args:
        x: first input
        y: second input
        sigma: kernel parameter

    Returns:
        scalar: kernel function value
    """
    return np.exp(-0.5 * np.linalg.norm(x - y, axis=1, ord=2) ** 2 / sigma ** 2)
