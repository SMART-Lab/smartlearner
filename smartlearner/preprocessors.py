import numpy as np

from .interfaces.preprocess import Preprocess


class NormalizeFeature(Preprocess):
    def __init__(self, dataset, feature_selector, mu=None, sigma=None, parent_preprocess=None):
        """ Transform given features to their Z-score

        Parameters
        ----------
        dataset: Dataset
        feature_selector: [int] or slice object
            This is passed to the dataset to specify which features this preprocessor is
            applied on.
        mu: float, optional
            This is the mean used to normalize the features. If it is not provided, the
            mean of all selected features will be used.
        sigma: float, optional
            This is the standard deviation used to normalize the features. If it is not <
            provided, the standard deviation of all selected features will be used.
        parent_preprocess: Preprocess
            The previously applied preprocessor. Used to chain preprocessors.

        """
        super().__init__(parent_preprocess)
        self.feature_selector = feature_selector
        self.mu = self._get_values(dataset)[self.feature_selector].mean() if mu is None else mu
        self.sigma = self._get_values(dataset)[self.feature_selector].std() if sigma is None else sigma

    def _reverse_preprocessing(self, dset):
        data = self._get_values(dset)
        data[self.feature_selector] = data[self.feature_selector] * self.sigma + self.mu
        return self._dataset_copy(dset, data)

    def _apply_preprocessing(self, dset):
        data = self._get_values(dset).copy()
        data[self.feature_selector] = (data[self.feature_selector] - self.mu) / self.sigma
        return self._dataset_copy(dset, data)

    def _get_values(self, dset):
        return self._get_array(dset).get_value()

    def _get_array(self, dset):
        return dset.inputs


class NormalizeTargets(NormalizeFeature):
    def _get_array(self, dset):
        return dset.targets
