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
    def __init__(self, predict_fct, dataset):
        super(ClassificationError, self).__init__()

        batch_size = 1024  # Internal buffer
        self.nb_batches = int(np.ceil(len(dataset) / batch_size))

        input = dataset.symb_inputs
        target = dataset.symb_targets
        classification_errors = T.neq(predict_fct(input), target)

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}
        self.compute_classification_error = theano.function([no_batch],
                                                            classification_errors,
                                                            givens=givens,
                                                            name="compute_classification_error")

    def update(self, status):
        classif_errors = []
        for i in range(self.nb_batches):
            classif_errors.append(self.compute_classification_error(i))

        classif_errors = np.concatenate(classif_errors)
        return classif_errors.mean(), classif_errors.std(ddof=1) / np.sqrt(len(classif_errors))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)


class ReconstructionError(View):
    def __init__(self, predict_fct, dataset):
        super().__init__()

        batch_size = 1024  # Internal buffer
        self.nb_batches = int(np.ceil(len(dataset) / batch_size))

        input = dataset.symb_inputs
        target = dataset.symb_targets
        reconstruction_errors = T.sqr(predict_fct(input) - target).mean()

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}
        self.compute_reconstruction_error = theano.function([no_batch],
                                                            reconstruction_errors,
                                                            givens=givens,
                                                            name="compute_reconstruction_error_" + dataset.name)

    def update(self, status):
        reconstruction_errors = []
        for i in range(self.nb_batches):
            reconstruction_errors.append(self.compute_reconstruction_error(i))

        reconstruction_errors = np.array(reconstruction_errors)
        return reconstruction_errors.mean(), reconstruction_errors.std(ddof=1) / np.sqrt(len(reconstruction_errors))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)
