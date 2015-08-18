from collections import OrderedDict

import numpy as np

from ..utils import sharedX
from . import UpdateRule


class DecreasingLearningRate(UpdateRule):
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
        self.lr = lr
        self.dc = dc
        self._updates = OrderedDict()

    def _get_updates(self):
        return self._updates

    def apply(self, directions):
        new_directions = OrderedDict()

        for param, gparam in directions.items():
            lr = sharedX(self.lr * np.ones_like(param.get_value()), name='lr_' + param.name)

            if self.dc != 0.:
                # Decrease the learning rate by a factor of `dc` after each update.
                self._updates[lr] = self.dc * lr

            new_directions[param] = lr * gparam

        return new_directions
