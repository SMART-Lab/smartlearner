from collections import OrderedDict

from theano import tensor as T

from abc import ABCMeta, abstractmethod


class Loss(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.consider_constant = []  # Part of the computational graph to be considered as a constant.
        self._gradients = None

    @abstractmethod
    def _get_updates(self):
        raise NotImplementedError("Subclass of 'Loss' must implement '_get_updates()'.")

    @abstractmethod
    def _compute_loss(self, model_output):
        raise NotImplementedError("Subclass of 'Loss' must implement '_compute_loss(model_output)'.")

    @property
    def gradients(self):
        if self._gradients is None:
            self._gradients = self._get_gradients()

        return self._gradients

    @property
    def updates(self):
        updates = OrderedDict()
        updates.update(self.model.updates)
        updates.update(self._get_updates())
        return updates

    def _get_gradients(self):
        model_output = self.model.get_model_output(self.dataset.symb_inputs)
        loss = self._compute_loss(model_output)

        gparams = T.grad(cost=loss,
                         wrt=self.model.parameters,
                         consider_constant=self.consider_constant)
        self._gradients = dict(zip(self.model.parameters, gparams))
        return self.gradients
