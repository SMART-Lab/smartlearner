import numpy as np

from collections import OrderedDict
from theano import tensor as T
from os.path import join as pjoin
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
        self._seed = seed
        self._srng = RandomStreams(self._seed)

        # theano's normal distribution function takes the std (sigma_t) instead of the variance (sigma_t^2);
        # sqrt is therefore applied beforehand to the parameters ( sigma_t = sqrt(eta) / (1/t)^(gamme/2) )
        self._eta = eta ** 0.5
        self._gamma = 0.55 / 2

        self.t = sharedX(1, name='gradient_noise_t')
        self.std = sharedX(self._eta / 1 ** self._gamma, name='gradient_noise_std')

    def _get_updates(self):
        return self._updates

    def apply(self, directions):
        new_directions = OrderedDict()

        # Define t, increasing at each timestep
        # TODO: Find a way to have access to t instead of managing it here
        self._updates[self.t] = self.t + 1

        # Define sigma_t, decaying at each timestep
        self._updates[self.std] = self._eta / T.pow(1 + self.t, self._gamma)

        # Sample noise for each parameter
        for param, gparam in directions.items():
            gradient_noise = self._srng.normal(param.get_value().shape, 0, self.std)

            new_directions[param] = gparam + gradient_noise

        return new_directions

    def save(self, path):
        state = {"version": 1,
                 "self._eta": self._eta,
                 "self._gamma": self._gamma,
                 "self._seed": self._seed,
                 "t": self.t.get_value(),
                 "std": self.std.get_value(),
                 "_srng_rstate": self._srng.rstate,
                 "_srng_state_updates": [state_update[0].get_value() for state_update in self._srng.state_updates]}

        np.savez(pjoin(path, type(self).__name__ + '.npz'), **state)

    def load(self, path):
        state = np.load(pjoin(path, type(self).__name__ + '.npz'))

        self.t.set_value(state["t"])
        self.std.set_value(state["std"])
        self._srng.rstate[:] = state['_srng_rstate']

        for state_update, saved_state in zip(self._srng.state_updates, state["_srng_state_updates"]):
            state_update[0].set_value(saved_state)
