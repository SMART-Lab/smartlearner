import theano
import theano.tensor as T
from collections import OrderedDict

import numpy as np
from . import Optimizer


class SGD(Optimizer):
    def __init__(self, model, loss_fct, dataset, batch_size=1, update_rules=None, param_modifiers=None):
        super(SGD, self).__init__(model, loss_fct, dataset, update_rules=update_rules, param_modifiers=param_modifiers)
        self.batch_size = batch_size
        self.nb_updates_per_epoch = int(np.ceil(len(dataset) / self.batch_size))

    def build_learning_function(self, extra_updates={}):
        self.gradients, updates_from_get_gradients = self.model.get_gradients(self.loss)

        # Apply update rules
        updates_from_update_rules = OrderedDict()
        for update_rule in self.update_rules:
            gradients, updates_from_update_rule = update_rule.apply(self.gradients)
            self.gradients.update(gradients)
            updates_from_update_rules.update(updates_from_update_rule)

        # Update parameters
        params_updates = OrderedDict()
        for param, gparam in self.gradients.items():
            params_updates[param] = param - self.gradients[param]

        # Modify parameters, if needed
        updates_from_param_modifiers = OrderedDict()
        for param_modifier in self.param_modifiers:
            modified_params_updates, updates_from_param_modifier = param_modifier.apply(params_updates)
            params_updates.update(modified_params_updates)
            updates_from_param_modifiers.update(updates_from_param_modifier)

        # Merge all different updates
        updates = OrderedDict()
        updates.update(params_updates)
        updates.update(updates_from_get_gradients)
        updates.update(updates_from_update_rules)
        updates.update(updates_from_param_modifiers)
        updates.update(extra_updates)

        no_batch = T.iscalar('no_batch')
        givens = {input: data[no_batch * self.batch_size:(no_batch + 1) * self.batch_size] for input, data in zip(self.inputs, self.data)}
        learn = theano.function([no_batch],
                                updates=updates,
                                givens=givens,
                                name="learn")
        return learn
