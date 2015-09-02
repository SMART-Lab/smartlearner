import numpy as np
import theano
from theano import tensor as T

from .interfaces import View


class ItemGetter(View):
    def __init__(self, view, attribute):
        """ Retrieves `attribute` from a `view` which return an indexable object. """
        super(ItemGetter, self).__init__()
        self.view_obj = view
        self.attribute = attribute

    def update(self, status):
        infos = self.view_obj.view(status)
        return infos[self.attribute]


class ClassificationError(View):
    def __init__(self, predict_fct, dataset, batch_size=None):
        super().__init__()
        self._batch_size = batch_size

        input = dataset.symb_inputs
        target = dataset.symb_targets
        classification_errors = T.neq(predict_fct(input), target)

        if batch_size is None:
            batch_size = len(dataset)

        self.nb_batches = int(np.ceil(len(dataset) / batch_size))
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}

        self.compute_classification_error = theano.function([no_batch],
                                                            classification_errors,
                                                            givens=givens,
                                                            name="compute_classification_error" + dataset.name)

    def update(self, status):
        classif_errors = np.concatenate([self.compute_classification_error(i) for i in range(self.nb_batches)])
        return float(classif_errors.mean()), float(classif_errors.std(ddof=1) / np.sqrt(len(classif_errors)))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)


class RegressionError(View):
    def __init__(self, predict_fct, dataset, batch_size=None):
        super().__init__()
        self._batch_size = batch_size

        input = dataset.symb_inputs
        target = dataset.symb_targets

        regression_errors = T.sqr(predict_fct(input) - target).mean(axis=1)

        if batch_size is None:
            batch_size = len(dataset)

        self.nb_batches = int(np.ceil(len(dataset) / batch_size))
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}

        self.compute_regression_error = theano.function([no_batch],
                                                        regression_errors,
                                                        givens=givens,
                                                        name="compute_reconstruction_error_" + dataset.name)

    def update(self, status):
        regression_errors = np.concatenate([self.compute_regression_error(i) for i in range(self.nb_batches)])
        return float(regression_errors.mean()), float(regression_errors.std(ddof=1) / np.sqrt(len(regression_errors)))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)
