import numpy as np
from collections import OrderedDict
from os.path import join as pjoin
from .. import utils

from abc import ABCMeta, abstractmethod


class Optimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, loss):
        self.loss = loss
        self._tasks = []

        self._direction_modifiers = []
        self._param_modifiers = []
        self._directions = None

    def append_direction_modifier(self, direction_modifier):
        self._direction_modifiers.append(direction_modifier)

    def append_param_modifier(self, param_modifier):
        self._param_modifiers.append(param_modifier)

    @abstractmethod
    def _get_directions(self):
        raise NotImplementedError("Subclass of 'Optimizer' must implement '_get_directions()'.")

    @abstractmethod
    def _get_updates(self):
        raise NotImplementedError("Subclass of 'Optimizer' must implement private property '_updates'.")

    @abstractmethod
    def getstate(self):
        """ Returns the state of the optimizer. """
        raise NotImplementedError("Subclass of 'Optimizer' must implement 'getstate()'.")

    @abstractmethod
    def setstate(self, state):
        """ Restores the optimizer to a given state. """
        raise NotImplementedError("Subclass of 'Optimizer' must implement 'setstate(state)'.")

    @property
    def directions(self):
        if self._directions is None:
            self._directions = self._get_directions()

        return self._directions

    @property
    def tasks(self):
        tasks = []
        tasks.extend(self.loss.tasks)

        for direction_modifier in self._direction_modifiers:
            tasks.extend(direction_modifier.tasks)

        for param_modifier in self._param_modifiers:
            tasks.extend(param_modifier.tasks)

        tasks.extend(self._tasks)
        return tasks

    @property
    def updates(self):
        updates = OrderedDict()

        directions = self.directions
        updates.update(self.loss.updates)  # Gather updates from the loss.
        updates.update(self._get_updates())  # Gather updates from the optimizer.

        # Apply directions modifiers and gather updates from these modifiers.
        updates.update(self._apply_modifiers(self._direction_modifiers, directions))

        # Update parameters
        params_updates = OrderedDict()
        for param, direction in directions.items():
            params_updates[param] = param + direction
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

    def save(self, path):
        self.loss.save(path)
        state = self.getstate()
        state["__name__"] = type(self).__name__
        np.savez(pjoin(path, 'optimizer.npz'), **state)

        direction_modifiers_savedir = utils.create_folder(pjoin(path, 'direction_modifiers'))
        for i, direction_modifier in enumerate(self._direction_modifiers):
            direction_modifier_savedir = utils.create_folder(pjoin(direction_modifiers_savedir, 'direction_modifier_{}'.format(i)))
            direction_modifier.save(direction_modifier_savedir)

        param_modifiers_savedir = utils.create_folder(pjoin(path, 'param_modifiers'))
        for i, param_modifier in enumerate(self._param_modifiers):
            param_modifier_savedir = utils.create_folder(pjoin(param_modifiers_savedir, 'param_modifier_{}'.format(i)))
            param_modifier.save(param_modifier_savedir)

    def load(self, path):
        self.loss.load(path)
        state = np.load(pjoin(path, 'optimizer.npz'))
        self.setstate(state)

        direction_modifiers_loaddir = pjoin(path, 'direction_modifiers')
        for i, direction_modifier in enumerate(self._direction_modifiers):
            direction_modifier.load(pjoin(direction_modifiers_loaddir, 'direction_modifier_{}'.format(i)))

        param_modifiers_loaddir = pjoin(path, 'param_modifiers')
        for i, param_modifier in enumerate(self._param_modifiers):
            param_modifier.load(pjoin(param_modifiers_loaddir, 'param_modifier_{}'.format(i)))
