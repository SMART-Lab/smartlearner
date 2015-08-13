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
            learning rate
        dc: float
            decreasing constant (decay)
        """
        super(DecreasingLearningRate, self).__init__()
        assert dc <= 1.
        assert dc >= 0.
        self.lr = lr
        self.dc = dc

    def apply(self, directions):
        updates = OrderedDict()
        new_directions = OrderedDict()

        for param, gparam in directions.items():
            lr = sharedX(self.lr * np.ones_like(param.get_value()), name='lr_' + param.name)

            if self.dc != 0.:
                # Decrease the learning rate by a factor of `dc` after each update.
                updates[lr] = self.dc * lr

            new_directions[param] = lr * gparam

        return new_directions, updates
