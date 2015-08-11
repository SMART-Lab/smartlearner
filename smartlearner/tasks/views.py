from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
from theano import tensor as T
import theano


class View(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.updates = OrderedDict()
        self.value = None
        self.last_update = -1

    def view(self, status):
        if self.last_update != status.current_update:
            self.value = self.update(status)
            self.last_update = status.current_update

        return self.value

    @abstractmethod
    def update(self, status):
        raise NotImplementedError("Subclass of 'View' must implement 'update(status)'.")

    def __str__(self):
        return "{0}".format(self.value)


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

        input = T.matrix('input')
        target = T.matrix('target')
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
