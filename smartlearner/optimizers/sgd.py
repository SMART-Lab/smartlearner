from collections import OrderedDict

from ..interfaces import Optimizer


class SGD(Optimizer):
    def __init__(self, loss):
        super().__init__(loss)

    def _get_updates(self):
        return {}  # There is no updates for SGD.

    def _get_directions(self):
        # Take the opposite of the gradients as directions.
        directions = OrderedDict()
        for param, gradient in self.loss.gradients.items():
            directions[param] = -gradient

        return directions

    def getstate(self):
        state = {"version": 1}
        return state

    def setstate(self, state):
        pass
