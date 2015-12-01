import numpy as np

from .interfaces.preprocess import Preprocess


class NormalizeFeature(Preprocess):
    def __init__(self, dataset, feature_selector, mu=None, sigma=None, parent_preprocess=None):
        super().__init__(parent_preprocess)
        self.feature_selector = feature_selector
        self.mu = self._get_values(dataset)[self.feature_selector].mean() if mu is None else mu
        self.sigma = self._get_values(dataset)[self.feature_selector].std() if sigma is not None else sigma

    def _reverse_preprocessing(self, dset):
        data = self._get_values(dset).get_value()
        data[self.feature_selector] = data[self.feature_selector] * self.sigma + self.mu
        return self._dataset_copy(dset, data)

    def _apply_preprocessing(self, dset):
        data = self._get_values(dset).get_value()
        data[self.feature_selector] = (data[self.feature_selector] - self.mu) / self.sigma
        return self._dataset_copy(dset, data)

    def _get_values(self, dset):
        return self._get_array(dset).get_value()

    def _get_array(self, dset):
        return dset.inputs


class NormalizeTargets(NormalizeFeature):
    def _get_array(self, dset):
        return dset.targets


class LagExamples(Preprocess):
    def __init__(self, lag, parent_preprocess=None):
        super().__init__(parent_preprocess)
        self.lag = lag

    def _reverse_preprocessing(self, dset):
        raise NotImplementedError

    def _apply_preprocessing(self, dset):
        np_dset = dset.inputs.get_value()
        dset_length = np_dset.shape[0]
        nb_chunks = dset_length - self.lag

        chunks = np.expand_dims(np_dset[:self.lag].ravel(), 0)
        for k in range(1, nb_chunks):
            chunks = np.vstack((chunks, np.expand_dims(np_dset[k:self.lag+k].ravel(), 0)))

        if dset.has_targets:
            targets = np.expand_dims(dset.targets.get_value()[self.lag-1], 0)
            for k in range(1, nb_chunks):
                targets = np.vstack((targets, np.expand_dims(dset.targets.get_value()[self.lag+k-1], 0)))

        return self._dataset_copy(dset, chunks, targets)
