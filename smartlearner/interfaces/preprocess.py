from abc import ABCMeta, abstractmethod

from .dataset import Dataset


class Preprocess(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, dataset):
        tgts = dataset.targets.get_value() if dataset.has_targets else None
        super().__init__(inputs=dataset.inputs.get_value(), targets=tgts,
                         name=dataset.name, keep_on_cpu=dataset.keep_on_cpu)
        self.symb_inputs = dataset.symb_inputs
        self.symb_targets = dataset.symb_targets

        self._is_preprocessed = False
        self._dataset = dataset

    def apply_preprocess(self):
        if not self._is_preprocessed:
            try:
                self._dataset.apply_preprocess()
            except AttributeError:
                pass
            self._apply_preprocessing()
            self._is_preprocessed = True
        return self

    @abstractmethod
    def reverse_preprocessing(self, dset):
        raise NotImplementedError("Subclass of 'Preprocess' must implement 'reverse_preprocess'.")

    @abstractmethod
    def _apply_preprocessing(self):
        raise NotImplementedError("Subclass of 'Preprocess' must implement '_apply_preprocess'.")
