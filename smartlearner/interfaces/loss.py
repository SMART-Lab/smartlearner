import numpy as np
from collections import OrderedDict
from os.path import join as pjoin

from theano import tensor as T

from abc import ABCMeta, abstractmethod


class Loss(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.consider_constant = []  # Part of the computational graph to be considered as a constant.
        self._tasks = []
        self._gradient_modifiers = []
        self._gradients = None
        self._loss = None
        self._losses = None
        self._gradient_modifiers_updates = {}

    def append_gradient_modifier(self, gradient_modifier):
        self._gradient_modifiers.append(gradient_modifier)

    @abstractmethod
    def _get_updates(self):
        raise NotImplementedError("Subclass of 'Loss' must implement '_get_updates()'.")

    @abstractmethod
    def getstate(self):
        """ Returns the state of the loss. """
        raise NotImplementedError("Subclass of 'Loss' must implement 'getstate()'.")

    @abstractmethod
    def setstate(self, state):
        """ Restores the loss to a given state. """
        raise NotImplementedError("Subclass of 'Loss' must implement 'setstate(state)'.")

    def _compute_losses(self, model_output):
        class_name = self.__class__.__name__
        raise NotImplementedError("{0} does not implement '_compute_losses(model_output)'.".format(class_name))

    def _compute_loss(self, model_output):
        """ Computes the loss for the whole batch.

        Notes:
        ------
        Unless overriden, the default behavior is to return the mean of all individual losses.
        """
        try:
            return T.mean(self.losses)
        except NotImplementedError:
            raise NotImplementedError("Subclass of 'Loss' must either implement '_compute_loss(model_output)' or '_compute_losses(model_output)'.")

    @property
    def losses(self):
        """ Gets individual losses (one for each batch example). """
        if self._losses is None:
            model_output = self.model.get_output(self.dataset.symb_inputs)
            self._losses = self._compute_losses(model_output)

        return self._losses

    @property
    def loss(self):
        """ Gets loss for the whole batch. """
        if self._loss is None:
            model_output = self.model.get_output(self.dataset.symb_inputs)
            self._loss = self._compute_loss(model_output)

        return self._loss

    @property
    def gradients(self):
        if self._gradients is None:
            self._gradients = self._get_gradients()

        return self._gradients

    @property
    def tasks(self):
        return self.model.tasks + self._tasks

    @property
    def updates(self):
        updates = OrderedDict()
        updates.update(self.model.updates)
        updates.update(self._get_updates())
        updates.update(self._gradient_modifiers_updates)
        return updates

    def _get_gradients(self):
        gparams = T.grad(cost=self.loss,
                         wrt=self.model.parameters,
                         consider_constant=self.consider_constant)
        self.orig_gradients = OrderedDict(zip(self.model.parameters, gparams))

        # Apply gradients modifiers and gather updates from these modifiers.
        gradients = self.orig_gradients.copy()
        self._gradient_modifiers_updates = self._apply_modifiers(self._gradient_modifiers, gradients)
        return gradients

    def _apply_modifiers(self, list_modifiers, objects_to_modify):
        # TODO: this is a copy-paste of the method in the class Optimizer.
        updates = OrderedDict()
        for modifier in list_modifiers:
            modified_objects = modifier.apply(objects_to_modify)
            objects_to_modify.update(modified_objects)
            updates.update(modifier.updates)

        return updates

    def save(self, path):
        self.model.save(path)
        state = self.getstate()
        state["__name__"] = type(self).__name__
        np.savez(pjoin(path, 'loss.npz'), **state)

    def load(self, path):
        self.model.load(path)
        state = np.load(pjoin(path, 'loss.npz'))
        self.setstate(state)
