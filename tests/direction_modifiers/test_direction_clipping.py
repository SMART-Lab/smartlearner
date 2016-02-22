import numpy as np
import theano
import theano.tensor as T
import unittest

from numpy.testing import assert_array_equal
from nose.tools import assert_true

from smartlearner import views, stopping_criteria, Trainer, tasks
from smartlearner.direction_modifiers import DirectionClipping
from smartlearner.optimizers import SGD
from smartlearner.testing import DummyLoss, DummyBatchScheduler, DummyModel
from smartlearner.utils import sharedX


floatX = theano.config.floatX


class DummyLossWithGradient(DummyLoss):
    def __init__(self, cost, param):
        super().__init__()
        self.cost = cost
        self.model = DummyModel()
        self.model._parameters = [param]

    @property
    def loss(self):
        return self.cost


class TestDirectionClipping(unittest.TestCase):

    def _build_experiment(self, threshold=1):
        # Create an Nd gaussian function to optimize. This function is not
        # well-conditioned and there exists no perfect gradient step to converge in
        # only one iteration.
        N = 4
        center = 5*np.ones((1, N)).astype(floatX)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), (param-center).T))
        loss = DummyLossWithGradient(cost, param)
        gradient_clipping = DirectionClipping(threshold=threshold)
        loss.append_gradient_modifier(gradient_clipping)

        optimizer = SGD(loss)
        trainer = Trainer(optimizer, DummyBatchScheduler())

        # Monitor the learning rate.
        logger = tasks.Logger(views.MonitorVariable(list(optimizer.directions.values())[0]),
                              views.MonitorVariable(list(loss.gradients.values())[0]),
                              views.MonitorVariable(list(loss.orig_gradients.values())[0]),
                              views.MonitorVariable(gradient_clipping.grad_norm))
        trainer.append_task(logger)

        return trainer, logger, gradient_clipping

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_epoch = 10
        self.trainer, self.logger, self.gradient_clipping = self._build_experiment()
        self.trainer.append_task(stopping_criteria.MaxEpochStopping(self.max_epoch))
        self.trainer.train()

    def test_behaviour(self):
        max_epoch = 10
        for threshold in [0.5, 1, 1.5]:
            trainer, logger, gradient_clipping = self._build_experiment(threshold)
            trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))
            trainer.train()

            directions = np.array(logger.get_variable_history(0)).squeeze()
            assert_true(np.all(np.sqrt(np.sum(directions**2, axis=1)) <= threshold + 1e-6))

            gradients_clipped = np.array(logger.get_variable_history(1)).squeeze()
            assert_array_equal(-directions, gradients_clipped)

            gradients = np.array(logger.get_variable_history(2)).squeeze()
            norms = np.array(logger.get_variable_history(3)).squeeze()[:, None]
            assert_array_equal(gradients_clipped,
                               gradients*threshold/np.maximum(threshold, norms))
