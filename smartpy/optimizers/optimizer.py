import theano.tensor as T
from abc import ABCMeta, abstractmethod


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, loss_fct, dataset):
        self.model = model
        self.loss = loss_fct
        self.dataset = dataset

        self._update_rules = []
        self._param_modifiers = []

        self.data = [dataset.inputs_shared, dataset.targets_shared]
        self.inputs = [T.matrix('input' + str(i)) for i in range(len(self.data))]
        self.loss = loss_fct(*self.inputs)

    def append_update_rule(self, update_rule):
        self._update_rules.append(update_rule)

    def append_param_modifier(self, param_modifier):
        self._param_modifiers.append(param_modifier)

    @abstractmethod
    def _build_learning_function(self, extra_updates={}):
        raise NotImplementedError("Subclass of 'Optimizer' must implement 'build_learning_function(extra_updates)'.")
