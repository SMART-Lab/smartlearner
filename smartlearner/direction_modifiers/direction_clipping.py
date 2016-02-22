from collections import OrderedDict

import theano.tensor as T

from ..interfaces import DirectionModifier


class DirectionClipping(DirectionModifier):
    """ Norm clips a direction to a given threshold."

    This is an implementation of what is proposed in [Pascanu12]_.

    References
    ----------
    .. [Pascanu12] Pascanu, R., Mikolov, T., & Bengio, Y. "On the difficulty
                   of training recurrent neural networks", ICML,
                   http://doi.org/10.1109/72.279181, 2012
    """
    def __init__(self, threshold=1):
        super(DirectionClipping).__init__()
        self.threshold = threshold
        self._updates = OrderedDict()

    def _get_updates(self):
        return self._updates

    def apply(self, directions):
        new_directions = OrderedDict()

        self.grad_norm = T.sqrt(sum(map(lambda d: T.sqr(d).sum(), directions.values())))

        # Note that rescaling is one if grad_norm <= threshold.
        rescaling = self.threshold / T.maximum(self.grad_norm, self.threshold)

        for param, direction in directions.items():
            direction_clipped = direction * rescaling
            new_directions[param] = direction_clipped

        return new_directions
