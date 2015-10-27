import numpy as np

from .interfaces.preprocess import Preprocess


class NormalizeFeature(Preprocess):
    def __init__(self, dataset, feature_selector, mu=None, sigma=None):
        super().__init__(dataset)
        self.feature_selector = feature_selector
        self.mu = self._get_values().get_value()[feature_selector].mean() if mu is None else mu
        self.sigma = self._get_values().get_value()[feature_selector].std() if sigma is not None else sigma

    def reverse_preprocessing(self, dset):
        return dset * self.sigma + self.mu

    def _apply_preprocessing(self):
        data = self._get_values().get_value()
        data[self.feature_selector] = (data[self.feature_selector] - self.mu) / self.sigma
        self._get_values().set_value(data)

    def _get_values(self):
        return self.inputs


class NormalizeTargets(NormalizeFeature):
    def _get_values(self):
        return self.targets


class LagExamples(Preprocess):
    def __init__(self, dataset, lag):
        super().__init__(dataset)
        self.lag = lag

    def reverse_preprocessing(self, dset):
        raise NotImplementedError

    def _apply_preprocessing(self):
        np_dset = self.inputs.get_value()
        dset_length = np_dset.shape[0]
        nb_chunks = dset_length - self.lag

        chunks = np.expand_dims(np_dset[:self.lag].ravel(), 0)
        for k in range(1, nb_chunks):
            chunks = np.vstack((chunks, np.expand_dims(np_dset[k:self.lag+k].ravel(), 0)))
        self.inputs.set_value(chunks)

        if self.has_targets:
            targets = np.expand_dims(self.targets.get_value()[self.lag-1], 0)
            for k in range(1, nb_chunks):
                targets = np.vstack((targets, np.expand_dims(self.targets.get_value()[self.lag+k-1], 0)))
            self.targets.set_value(targets)
