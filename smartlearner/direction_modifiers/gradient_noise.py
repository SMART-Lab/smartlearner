from collections import OrderedDict
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from smartlearner.interfaces.direction_modifier import DirectionModifier
from smartlearner.utils import sharedX


class GradientNoise(DirectionModifier):
    """
    Adds annealed gaussian noise to the gradients at every time step, by decaying the variance at each time step

    g_t <- g_t + N(0, sigma_t^2)
    sigma_t^2 = eta / (1 + t)^gamma

    with eta selected from {0.01, 0.3, 1.0) and gamma = 0.55

    See: "Adding gradient noise improves learning for very deep networks", http://arxiv.org/pdf/1511.06807v1.pdf
    """

    def __init__(self, eta=0.01, seed=1234):
        super(GradientNoise, self).__init__()
        self._updates = OrderedDict()
        self._srng = RandomStreams(seed)

        # theano's normal distribution function takes the std (sigma_t) instead of the variance (sigma_t^2);
        # sqrt is therefore applied beforehand to the parameters ( sigma_t = sqrt(eta) / (1/t)^(gamme/2) )
        self._eta = eta ** 0.5
        self._gamma = 0.55 / 2

    def _get_updates(self):
        return self._updates

    def apply(self, directions):
        new_directions = OrderedDict()

        # Define t, increasing at each timestep
        # TODO: Find a way to have access to t instead of managing it here
        t = sharedX(0, 'gradient_noise_t')
        self.updates[t] = t + 1

        # Define sigma_t, decaying at each timestep
        std = sharedX(self._eta / 1 ** self._gamma, name='gradient_noise_std')
        self.updates[std] = self._eta / T.pow(1 + t, self._gamma)

        # Sample noise for each parameter
        for param, gparam in directions.items():
            gradient_noise = self._srng.normal(param.get_value().shape, 0, std)

            new_directions[param] = gparam + gradient_noise

        return new_directions
