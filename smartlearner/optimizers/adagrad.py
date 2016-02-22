from collections import OrderedDict
import theano.tensor as T

from ..interfaces import Optimizer
from ..utils import sharedX


class AdaGrad(Optimizer):
    """ Implements the AdaGrad optimizer [Duchi11]_.

    .. [Duchi11] Duchi, J., Hazan, E., & Singer, Y., "Adaptive Subgradient
                 Methods for Online Learning and Stochastic Optimization",
                 Journal of Machine Learning Research, vol. 12, pp. 2121-2159,
                 2011.
    """
    def __init__(self, loss, lr, eps=1e-6):
        """
        Parameters
        ----------
        loss: `smartlearner.interfaces.loss.Loss` object
            Loss function from which to obtain the gradients.
        lr: float
            Initial learning rate.
        eps: float (optional)
            Epsilon needed to avoid division by zero.
        """
        super().__init__(loss)
        self.lr = lr
        self.eps = eps
        self.parameters = {}
        self._updates = OrderedDict()

    def _get_updates(self):
        return self._updates

    def _get_directions(self):
        """ Produces descending directions. """
        directions = OrderedDict()

        for i, (param, gradient) in enumerate(self.loss.gradients.items()):
            # sum_squared_grad := \sum g_t^2
            param_name = param.name if param.name is not None else str(i)
            sum_squared_grad = sharedX(param.get_value() * 0., name='sum_squared_grad_' + param_name)
            self.parameters[sum_squared_grad.name] = sum_squared_grad

            # Accumulate gradient
            new_sum_squared_grad = sum_squared_grad + T.sqr(gradient)

            # Compute update
            root_sum_squared = T.sqrt(new_sum_squared_grad + self.eps)

            # Apply update
            self._updates[sum_squared_grad] = new_sum_squared_grad
            directions[param] = -(self.lr/root_sum_squared) * gradient

        return directions

    def getstate(self):
        state = {"version": 1}

        for k, param in self.parameters.items():
            state[k] = param.get_value()

        return state

    def setstate(self, state):
        for k, param in self.parameters.items():
            param.set_value(state[k])
