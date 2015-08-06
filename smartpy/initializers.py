import numpy as np
import theano

floatX = theano.config.floatX

class WeightInitializer:
    def __init__(self, random_seed=None):
        self.rng = np.random.mtrand.RandomState(random_seed)

    def __call__(self, shared_var):
        dim = shared_var.get_value().shape
        shared_var.set_value(self._generate_array(dim))
        return shared_var

    def _generate_array(self, dim):
        raise NotImplementedError("Use an implementation of this abstract class.")

    @staticmethod
    def _init_range(dim):
        return np.sqrt(6. / sum(dim))


class UniformInitializer(WeightInitializer):
    def _generate_array(self, dim):
        init_range = self._init_range(dim)
        return np.asarray(self.rng.uniform(low=-init_range, high=init_range, size=dim), dtype=floatX)


class ZerosInitializer(WeightInitializer):
    def _generate_array(self, dim):
        return np.zeros(dim, dtype=floatX)


class DiagonalInitializer(WeightInitializer):
    def _generate_array(self, dim):
        return np.eye(dim, dtype=floatX)


class OrthogonalInitializer(WeightInitializer):
    def _generate_array(self, dim):
        max_dim = max(dim)
        uniform_initer = UniformInitializer()
        uniform_initer.rng = self.rng
        return np.asarray(np.linalg.svd(uniform_initer((max_dim, max_dim)))[2][:dim[0], :dim[1]], dtype=floatX)


class GaussienInitializer(WeightInitializer):
    def _generate_array(self, dim):
        return np.asarray(self.rng.normal(loc=0, scale=self._init_range(dim), size=dim), dtype=floatX)
