from collections import OrderedDict

from abc import ABCMeta, abstractmethod


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, loss):
        self.loss = loss

        self._update_rules = []
        self._param_modifiers = []

    def append_update_rule(self, update_rule):
        self._update_rules.append(update_rule)

    def append_param_modifier(self, param_modifier):
        self._param_modifiers.append(param_modifier)

    @abstractmethod
    def _get_directions(self):
        raise NotImplementedError("Subclass of 'Optimizer' must implement '_get_directions()'.")

    def gather_updates(self):
        updates = OrderedDict()

        self.directions, updates_from_get_directions = self._get_directions()
        updates.update(updates_from_get_directions)

        updates.update(self._apply_updates(self._update_rules, self.directions))

        # Update parameters
        params_updates = OrderedDict()
        for param, gparam in self.directions.items():
            params_updates[param] = param + self.directions[param]
        updates.update(params_updates)

        updates.update(self._apply_updates(self._param_modifiers, params_updates))

        return updates

    def _apply_updates(self, list_updates, object_to_update):
        update_dict = OrderedDict()
        for update in list_updates:
            modified_object, updates_to_add = update.apply(object_to_update)
            object_to_update.update(modified_object)
            update_dict.update(updates_to_add)
        return update_dict