from abc import ABCMeta, abstractmethod


class Preprocess:
    __metaclass__ = ABCMeta

    def __init__(self, parent_preprocess=None):
        self._parent_preprocess = parent_preprocess

    def __call__(self, dset):
        return self.apply_preprocess(dset)

    def apply_preprocess(self, dset):
        try:
            dset = self._parent_preprocess.apply_preprocess(dset)
        except AttributeError:
            pass
        return self._apply_preprocessing(dset)

    def reverse_preprocessing(self, dset):
        try:
            dset = self._reverse_preprocessing(dset)
        except NotImplementedError:
            raise NotImplementedError(type(self).__name__ + " preprocessing can't be reversed.")
        try:
            dset = self._parent_preprocess.reverse_preprocess(dset)
        except AttributeError:
            pass
        return dset

    @staticmethod
    def _dataset_copy(dataset, inputs=None, targets=None):
        inp = inputs if inputs is not None else dataset.inputs
        tar = targets if targets is not None else dataset.targets
        return dataset.create_linked_dataset(inp, tar, dataset.name, dataset.keep_on_cpu)

    @abstractmethod
    def _reverse_preprocessing(self, dset):
        raise NotImplementedError("Subclass of 'Preprocess' must implement '_reverse_preprocess'.")

    @abstractmethod
    def _apply_preprocessing(self, dset):
        raise NotImplementedError("Subclass of 'Preprocess' must implement '_apply_preprocess'.")
