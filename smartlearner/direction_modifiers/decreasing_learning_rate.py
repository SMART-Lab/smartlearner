from collections import OrderedDict

import numpy as np
from os.path import join as pjoin

from ..utils import sharedX
from ..interfaces import DirectionModifier


class DecreasingLearningRate(DirectionModifier):
    def __init__(self, lr, dc=0.):
        """
        Implements a decreasing learning rate update rule.

        Parameters
        ----------
        lr: float
            Learning rate.
        dc: float in [0,1) (optional)
            Decreasing constant (decay). Default: 0.
        """
        if dc < 0. or dc >= 1:
            raise ValueError("`dc` ({}) must be between 0 (inclusive) and 1 (exclusive)!".format(dc))

        super(DecreasingLearningRate, self).__init__()
        self.base_lr = lr
        self.dc = dc
        self.parameters = {}
        self._updates = OrderedDict()

    def _get_updates(self):
        return self._updates

    def apply(self, directions):
        new_directions = OrderedDict()

        for i, (param, direction) in enumerate(directions.items()):
            param_name = param.name if param.name is not None else str(i)
            lr = sharedX(self.base_lr * np.ones_like(param.get_value()), name='lr_' + param_name)
            self.parameters[lr.name] = lr

            if self.dc != 0.:
                # Decrease the learning rate by a factor of `dc` after each update.
                self._updates[lr] = self.dc * lr

            new_directions[param] = lr * direction

        return new_directions

    def save(self, path):
        state = {"version": 1,
                 "dc": self.dc,
                 "base_lr": self.base_lr}

        for k, param in self.parameters.items():
            state[k] = param.get_value()

        np.savez(pjoin(path, type(self).__name__ + '.npz'), **state)

    def load(self, path):
        state = np.load(pjoin(path, type(self).__name__ + '.npz'))

        for k, param in self.parameters.items():
            param.set_value(state[k])
