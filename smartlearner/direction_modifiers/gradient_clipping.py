from collections import OrderedDict

import theano.tensor as T

from ..interfaces import DirectionModifier


class GradientClipping(DirectionModifier):
    def __init__(self, max_norm=1.):
        super(GradientClipping).__init__()
        self.max_norm = max_norm
        self._updates = OrderedDict()

    def _get_updates(self):
        return self._updates

    def apply(self, directions):
        new_directions = OrderedDict()

        for param, gparam in directions.items():
            length = gparam.flatten().norm(2)
            gparam *= T.min(length, self.max_norm)/length
            new_directions[param] = gparam

        return new_directions
