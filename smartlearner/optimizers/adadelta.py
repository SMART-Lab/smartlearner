from collections import OrderedDict

import theano
import theano.tensor as T

from ..interfaces import Optimizer
from ..utils import sharedX


class Adadelta(Optimizer):
    def __init__(self, loss, decay=0.95, epsilon=1e-6):
        """ Implements the Adadelta optimizer.

        The implementation details are available in [Zeiler12]_.

        Parameters
        ----------
        loss : `Loss`
            The value (a theano computation graph) to optimize.
        decay : float, optional
            Decay rate (related to the window of the moving average).
        epsilon : float, optional
            Used to prevent a division by zero in early iterations.

        Refrences
        ---------
        .. [Zeiler12] Matthew D. Zeiler "AdaDelta: An Adaptive Learning Rate Method",
                      http://arxiv.org/abs/1212.5701, 2012
        """
        super().__init__(loss)
        if decay <= 0 or decay > 1:
            raise ValueError("Adadelta's decay must be in (0,1].")

        self.dc = decay
        self.eps = epsilon

        self._updates = OrderedDict()
        self.mean_squared_grads = []
        self.mean_squared_dxs = []

    def _get_updates(self):
        return self._updates

    def _get_directions(self):
        """ Produces descending directions. """
        directions = OrderedDict()

        for i, (param, gradient) in enumerate(self.loss.gradients.items()):
            # mean_squared_grad := E[g^2]_{t-1}
            param_name = param.name if param.name is not None else str(i)
            mean_squared_grad = sharedX(param.get_value() * 0., name='mean_squared_grad_' + param_name)
            self.mean_squared_grads.append(mean_squared_grad)
            # mean_squared_dx := E[(\Delta x)^2]_{t-1}
            param_name = param.name if param.name is not None else str(i)
            mean_squared_dx = sharedX(param.get_value() * 0., name='mean_squared_dx_' + param_name)
            self.mean_squared_dxs.append(mean_squared_dx)

            # Accumulate gradient
            new_mean_squared_grad = self.dc * mean_squared_grad + (1 - self.dc) * T.sqr(gradient)

            # Compute update
            rms_dx_tm1 = T.sqrt(mean_squared_dx + self.eps)
            rms_grad_t = T.sqrt(new_mean_squared_grad + self.eps)
            delta_x_t = -(rms_dx_tm1/rms_grad_t) * gradient

            # Accumulate updates
            new_mean_squared_dx = self.dc * mean_squared_dx + (1 - self.dc) * T.sqr(delta_x_t)

            # Apply update
            self._updates[mean_squared_grad] = new_mean_squared_grad
            self._updates[mean_squared_dx] = new_mean_squared_dx
            directions[param] = delta_x_t

        return directions

    def getstate(self):
        state = {"version": 1,
                 "dc": self.dc,
                 "eps": self.eps}

        for param in self.mean_squared_grads + self.mean_squared_dxs:
            state[param.name] = param.get_value()

        return state

    def setstate(self, state):

        for param in self.mean_squared_grads + self.mean_squared_dxs:
            param.set_value(state[param.name])
