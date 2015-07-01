import numpy as np
from . import Optimizer


class SGD(Optimizer):
    def __init__(self, loss):
        super(SGD, self).__init__(loss)

    def _get_directions(self):
        self.gradients, updates_from_get_gradients = self.loss.get_gradients()

        # Take the opposite of the gradient.
        for param, gparam in self.gradients.items():
            self.gradients[param] = -gparam

        return self.gradients, updates_from_get_gradients
