from collections import OrderedDict

import theano
import theano.tensor as T
import numpy as np

from ..interfaces import Optimizer

class Adam(Optimizer):
    def __init__(self, loss, learning_rate=0.0001, b1=0.9, b2=0.999, epsilon=1e-8, lmbda=(1 - 1e-8)):
        super().__init__(loss)
        self.lr = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = epsilon
        self.lmbda = lmbda

        self._updates = OrderedDict()
        self.t = theano.shared(0)
        self.mts = []
        self.vts = []

        for param in loss.gradients:        # iterates on the parameters key
            self.mts.append(theano.shared(np.zeros_like(param.get_value())))
            self.vts.append(theano.shared(np.zeros_like(param.get_value())))

    def _get_updates(self):
        return self._updates

    def _get_directions(self):
        directions = OrderedDict()
        for (param, gradient), mtm1, vtm1 in zip(self.loss.gradients.items(), self.mts, self.vts):
            tp1 = self.t + 1
            self._updates[self.t] = tp1
            mt = self.b1 * mtm1 + (1 - self.b1) * gradient
            vt = self.b2 * vtm1 + (1 - self.b2) * gradient**2
            self._updates[mtm1] = mt
            self._updates[vtm1] = vt
            at = self.lr * T.sqrt(1 - self.b2**tp1) / (1 - self.b1**tp1)
            directions[param] = -at * mt / (T.sqrt(vt) + self.eps)

        return directions
