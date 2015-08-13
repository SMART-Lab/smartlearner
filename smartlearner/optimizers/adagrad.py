from collections import OrderedDict
import theano.tensor as T

from . import SGD
from ..utils import sharedX


class ADAGRAD(SGD):
    """
    Implements the ADAGRAD optimizer.

    Reference
    ---------
    Duchi, J., Hazan, E., & Singer, Y. (2010).
    Adaptive subgradient methods for online learning and stochastic optimization.
    Journal of Machine Learning
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

    def _get_directions(self):
        """ Produces descending directions. """
        directions = OrderedDict()
        updates = OrderedDict()

        # Get the opposite of the gradients.
        gradients, updates_from_get_gradients = super()._get_directions()
        updates.update(updates_from_get_gradients)

        for param, gparam in gradients.items():
            # sum_squared_grad := \sum g_t^2
            sum_squared_grad = sharedX(param.get_value() * 0., name='sum_squared_grad_' + param.name)
            self.parameters[sum_squared_grad.name] = sum_squared_grad

            # Accumulate gradient
            new_sum_squared_grad = sum_squared_grad + T.sqr(gparam)

            # Compute update
            root_sum_squared = T.sqrt(new_sum_squared_grad + self.eps)

            # Apply update
            updates[sum_squared_grad] = new_sum_squared_grad
            directions[param] = (self.lr/root_sum_squared) * gparam

        return directions, updates
