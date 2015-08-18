from collections import OrderedDict

from abc import ABCMeta, abstractmethod


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, loss):
        self.loss = loss

        self._update_rules = []
        self._param_modifiers = []
        self._directions = None

    def append_update_rule(self, update_rule):
        self._update_rules.append(update_rule)

    def append_param_modifier(self, param_modifier):
        self._param_modifiers.append(param_modifier)

    @abstractmethod
    def _get_directions(self):
        raise NotImplementedError("Subclass of 'Optimizer' must implement '_get_directions()'.")

    @abstractmethod
    def _get_updates(self):
        raise NotImplementedError("Subclass of 'Optimizer' must implement private property '_updates'.")

    @property
    def directions(self):
        if self._directions is None:
            self._directions = self._get_directions()

        return self._directions

    @property
    def updates(self):
        updates = OrderedDict()

        directions = self.directions
        updates.update(self.loss.updates)  # Gather updates from the loss.
        updates.update(self._get_updates())  # Gather updates from the optimizer.

        # Apply directions modifiers and gather updates from these modifiers.
        updates.update(self._apply_modifiers(self._update_rules, directions))

        # Update parameters
        params_updates = OrderedDict()
        for param, gparam in directions.items():
            params_updates[param] = param + gparam
        updates.update(params_updates)

        # Apply parameters modifiers and gather updates from these modifiers.
        updates.update(self._apply_modifiers(self._param_modifiers, params_updates))

        return updates

    def _apply_modifiers(self, list_modifiers, objects_to_modify):
        updates = OrderedDict()
        for modifier in list_modifiers:
            modified_objects = modifier.apply(objects_to_modify)
            objects_to_modify.update(modified_objects)
            updates.update(modifier.updates)

        return updates
