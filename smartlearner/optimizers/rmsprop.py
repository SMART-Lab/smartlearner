from collections import OrderedDict

import theano
import theano.tensor as T

from ..interfaces import Optimizer
from ..utils import sharedX


class RMSProp(Optimizer):
    def __init__(self, loss, lr, decay=0.95, epsilon=1e-6):
        """ Implements the RMSProp optimizer.

        The implementation details are available in [Tieleman12]_.

        Parameters
        ----------
        loss : `Loss`
            The value (a theano computation graph) to optimize.
        lr : float
            Learning rate.
        decay : float, optional
            Decay rate (related to the window of the moving average).
        epsilon : float, optional
            Used to prevent a division by zero in early iterations.

        Refrences
        ---------
        .. [Tieleman12] Tieleman, T. and Hinton, G. "Lecture 6.5 - rmsprop"
                        Coursera: Neural Networks for Machine Learning, 2012
        """
        super().__init__(loss)
        if decay <= 0 or decay > 1:
            raise ValueError("RMSProp's decay must be in (0,1].")

        self.lr = lr
        self.dc = decay
        self.eps = epsilon

        self._updates = OrderedDict()
        self.mean_squared_grads = []

    def _get_updates(self):
        return self._updates

    def _get_directions(self):
        """ Produces descending directions. """
        directions = OrderedDict()

        for i, (param, gradient) in enumerate(self.loss.gradients.items()):
            # mean_squared_grad := \sum dc * g_{t-1}^2 + (1-dc) g_t^2
            param_name = param.name if param.name is not None else str(i)
            mean_squared_grad = sharedX(param.get_value() * 0., name='mean_squared_grad_' + param_name)
            self.mean_squared_grads.append(mean_squared_grad)

            # Compute moving average of the squared gradient
            new_mean_squared_grad = T.cast(self.dc * mean_squared_grad + (1 - self.dc) * T.sqr(gradient), dtype=theano.config.floatX)

            # Compute new direction
            root_mean_squared = T.sqrt(new_mean_squared_grad + self.eps)

            # Apply update
            self._updates[mean_squared_grad] = new_mean_squared_grad
            directions[param] = -(self.lr/root_mean_squared) * gradient

        return directions

    def getstate(self):
        state = {"version": 1,
                 "lr": self.lr,
                 "dc": self.dc,
                 "eps": self.eps}

        for param in self.mean_squared_grads:
            state[param.name] = param.get_value()

        return state

    def setstate(self, state):

        for param in self.mean_squared_grads:
            param.set_value(state[param.name])
