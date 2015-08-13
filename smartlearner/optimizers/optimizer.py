from collections import OrderedDict

from abc import ABCMeta, abstractmethod


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, loss):
        self.loss = loss

        self._update_rules = []
        self._param_modifiers = []

        self.graph_updates = OrderedDict()
        self._directions = None

    def append_update_rule(self, update_rule):
        self._update_rules.append(update_rule)

    def append_param_modifier(self, param_modifier):
        self._param_modifiers.append(param_modifier)

    @property
    def directions(self):
        if self._directions is None:
            self._directions = self._get_directions()

        return self._directions

    @abstractmethod
    def _get_directions(self):
        raise NotImplementedError("Subclass of 'Optimizer' must implement '_get_directions()'.")

    def gather_updates(self):
        updates = OrderedDict()

        # Get graph updates of the loss.
        updates.update(self.loss.graph_updates)

        # Get graph updates of the optimizer.
        updates.update(self.graph_updates)

        # Apply directions modifiers
        updates.update(self._apply_modifier(self._update_rules, self.directions))

        # Update parameters
        params_updates = OrderedDict()
        for param, gparam in self.directions.items():
            params_updates[param] = param + self.directions[param]
        updates.update(params_updates)

        # Apply parameters modifiers
        updates.update(self._apply_modifier(self._param_modifiers, params_updates))

        return updates

    def _apply_modifier(self, list_modifiers, object_to_update):
        update_dict = OrderedDict()
        for modifier in list_modifiers:
            modified_object, updates_to_add = modifier.apply(object_to_update)
            object_to_update.update(modified_object)
            update_dict.update(updates_to_add)
        return update_dict
