from os.path import join as pjoin
from collections import OrderedDict

import theano.tensor as T
import numpy as np

from ..interfaces import Optimizer
from ..utils import sharedX


class Adam(Optimizer):
    def __init__(self, loss, lr=0.0001, b1=0.9, b2=0.999, epsilon=1e-8):
        """ Implements the Adam optimizer.

        The implementation details are available in [King15]_.

        Parameters
        ----------
        loss : `Loss`
            The value (a theano computation graph) to optimize.
        lr : float, optional
            The length of the direction that is used for optimisation.
            Analogous to the learning rate for `SGD`.
        b1 : float, optional
            The moving average factor for the mean of past gradients.
        b2 : float, optional
            The moving average factor for the second moment of past gradients.
        epsilon : float, optional
            Used to prevent a division by zero in early iterations.

        Attributes
        ----------
        loss : `Loss`
        lr : float
        b1 : float
        b2 : float
        eps : float
        t : float
            The current timestep counter.
        mts : [theano.shared]
            The m_t vectors used in the algorithm computations.
        vts : [theano.shared]
            The v_t vectors used in the algorithm computations.


        .. [King15] Kingma D. and Lei Ba, J. "Adam: A Method for Stochastic Optimization."
                    International Conference on Learning Representations, 2015.

        """
        super().__init__(loss)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = epsilon

        self._updates = OrderedDict()
        self.t = sharedX(0, name='t')
        self.mts = []
        self.vts = []

        for param in loss.gradients:  # iterates on the parameters key
            self.mts.append(sharedX(np.zeros_like(param.get_value()), name='mt_{}'.format(param.name)))
            self.vts.append(sharedX(np.zeros_like(param.get_value()), name='vt_{}'.format(param.name)))

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

    def getstate(self):
        state = {"version": 1,
                 "__name__": type(self).__name__,
                 "t": self.t.get_value()}

        for param in self.mts + self.vts:
            state[param.name] = param.get_value()

        return state

    def setstate(self, state):
        if state["__name__"] != type(self).__name__:
            msg = "Trying to restore a '{}'' object with the state of '{}'."
            raise NameError(msg.format(type(self).__name__, state["__name__"]))

        self.t.set_value(state["t"])

        for param in self.mts + self.vts:
            param.set_value(state[param.name])
