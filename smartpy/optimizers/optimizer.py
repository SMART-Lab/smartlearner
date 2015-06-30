import theano
import theano.tensor as T
from collections import OrderedDict

from abc import ABCMeta, abstractmethod


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, loss):
        self.loss = loss

        self._update_rules = []
        self._param_modifiers = []
        self.batch_size = 10

        #self.data = [dataset.inputs_shared, dataset.targets_shared]
        #self.inputs = [T.matrix('input' + str(i)) for i in range(len(self.data))]

    def append_update_rule(self, update_rule):
        self._update_rules.append(update_rule)

    def append_param_modifier(self, param_modifier):
        self._param_modifiers.append(param_modifier)

    @abstractmethod
    def _get_directions(self):
        raise NotImplementedError("Subclass of 'Optimizer' must implement '_get_directions()'.")

    def _build_learning_function(self, task_updates={}):
        self.directions, updates_from_get_directions = self._get_directions()

        # Apply update rules
        updates_from_update_rules = OrderedDict()
        for update_rule in self._update_rules:
            directions, updates_from_update_rule = update_rule.apply(self.directions)
            self.directions.update(directions)
            updates_from_update_rules.update(updates_from_update_rule)

        # Update parameters
        params_updates = OrderedDict()
        for param, gparam in self.directions.items():
            params_updates[param] = param + self.directions[param]

        # Modify parameters, if needed
        updates_from_param_modifiers = OrderedDict()
        for param_modifier in self._param_modifiers:
            modified_params_updates, updates_from_param_modifier = param_modifier.apply(params_updates)
            params_updates.update(modified_params_updates)
            updates_from_param_modifiers.update(updates_from_param_modifier)

        # Merge all different updates
        updates = OrderedDict()
        updates.update(params_updates)
        updates.update(updates_from_get_directions)
        updates.update(updates_from_update_rules)
        updates.update(updates_from_param_modifiers)
        updates.update(task_updates)

        no_batch = T.iscalar('no_batch')
        givens = {self.loss.dataset.symb_inputs: self.loss.dataset.inputs[no_batch * self.batch_size:(no_batch + 1) * self.batch_size],
                  self.loss.dataset.symb_targets: self.loss.dataset.targets[no_batch * self.batch_size:(no_batch + 1) * self.batch_size]}
            #{input: data[no_batch * self.batch_size:(no_batch + 1) * self.batch_size] for input, data in zip(self.inputs, self.data)}
        learn = theano.function([no_batch],
                                updates=updates,
                                givens=givens,
                                name="learn")

        return learn
