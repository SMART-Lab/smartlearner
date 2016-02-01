import numpy as np
import theano
import theano.tensor as T
import unittest
import tempfile

from numpy.testing import assert_array_equal

from smartlearner import views, stopping_criteria, Trainer, tasks
from smartlearner.direction_modifiers import DecreasingLearningRate
from smartlearner.optimizers import SGD
from smartlearner.testing import DummyLoss, DummyBatchScheduler
from smartlearner.utils import sharedX


floatX = theano.config.floatX


class DummyLossWithGradient(DummyLoss):
    def __init__(self, cost, param):
        super().__init__()
        self.cost = cost
        self.param = param

    def _get_gradients(self):
        gparam = T.grad(cost=self.cost, wrt=self.param)
        return {self.param: gparam}


class TestDecreasingLearningRate(unittest.TestCase):
    def _build_experiment(self):
        # Create an Nd gaussian function to optimize. This function is not
        # well-conditioned and there exists no perfect gradient step to converge in
        # only one iteration.
        N = 4
        center = 5*np.ones((1, N)).astype(floatX)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), (param-center).T))
        loss = DummyLossWithGradient(cost, param)

        optimizer = SGD(loss)
        direction_modifier = DecreasingLearningRate(lr=self.lr, dc=self.dc)
        optimizer.append_direction_modifier(direction_modifier)
        trainer = Trainer(optimizer, DummyBatchScheduler())

        # Monitor the learning rate.
        logger = tasks.Logger(views.MonitorVariable(list(direction_modifier.parameters.values())[0]))
        trainer.append_task(logger)

        return trainer, logger, direction_modifier

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lr = 1
        self.dc = 0.5
        self.max_epoch = 10
        self.trainer, self.logger, self.direction_modifier = self._build_experiment()
        self.trainer.append_task(stopping_criteria.MaxEpochStopping(self.max_epoch))
        self.trainer.train()

    def test_behaviour(self):
        learning_rate_per_update = np.array(self.logger.get_variable_history(0))[:, :, 0].flatten()
        expected_learning_rate_per_update = [self.lr * self.dc**i for i in range(self.max_epoch)]
        assert_array_equal(learning_rate_per_update, expected_learning_rate_per_update)

    def test_save_load(self):
        # Save training and resume it.
        with tempfile.TemporaryDirectory() as experiment_dir:
            # Save current training state of the experiment.
            self.trainer.save(experiment_dir)

            # Load previous training state of the experiment.
            trainer, logger, direction_modifier = self._build_experiment()
            trainer.load(experiment_dir)

            # Check the state of the direction modifier.
            for key in direction_modifier.parameters:
                assert_array_equal(direction_modifier.parameters[key].get_value(),
                                   self.direction_modifier.parameters[key].get_value())

    def test_resume(self):
        trainer1, logger1, direction_modifier1 = self._build_experiment()
        trainer1.append_task(stopping_criteria.MaxEpochStopping(5))
        trainer1.train()

        # Save training and resume it.
        with tempfile.TemporaryDirectory() as experiment_dir:
            # Save current training state of the experiment.
            trainer1.save(experiment_dir)

            # Load previous training state of the experiment.
            trainer2, logger2, direction_modifier2 = self._build_experiment()
            trainer2.append_task(stopping_criteria.MaxEpochStopping(10))
            trainer2.load(experiment_dir)
            trainer2.train()

        # Check that concatenating `logger1` with `logger2` is the same as `self.logger`.
        learning_rate_per_update_part1 = np.array(logger1.get_variable_history(0))[:, :, 0].flatten()
        learning_rate_per_update_part2 = np.array(logger2.get_variable_history(0))[:, :, 0].flatten()
        expected_learning_rate_per_update = np.array(self.logger.get_variable_history(0))[:, :, 0].flatten()
        assert_array_equal(np.r_[learning_rate_per_update_part1, learning_rate_per_update_part2],
                           expected_learning_rate_per_update)
