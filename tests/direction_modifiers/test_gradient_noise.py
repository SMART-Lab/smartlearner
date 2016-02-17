import numpy as np
import theano
import theano.tensor as T
import unittest
import tempfile

from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true

from smartlearner import views, stopping_criteria, Trainer, tasks
from smartlearner.direction_modifiers import GradientNoise
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

    def getstate(self):
        return {"param": self.param.get_value()}

    def setstate(self, state):
        self.param.set_value(state["param"])


class TestGradientNoise(unittest.TestCase):

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
        direction_modifier = GradientNoise()
        optimizer.append_direction_modifier(direction_modifier)
        trainer = Trainer(optimizer, DummyBatchScheduler())

        # Monitor the learning rate.
        logger = tasks.Logger(views.MonitorVariable(direction_modifier.t),
                              views.MonitorVariable(direction_modifier.std),
                              views.MonitorVariable(list(optimizer.directions.values())[0]),
                              views.MonitorVariable(list(loss.gradients.values())[0]))
        trainer.append_task(logger)

        return trainer, logger, direction_modifier

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_epoch = 10
        self.trainer, self.logger, self.direction_modifier = self._build_experiment()
        self.trainer.append_task(stopping_criteria.MaxEpochStopping(self.max_epoch))
        self.trainer.train()

    def test_behaviour(self):
        t_per_update = np.array(self.logger.get_variable_history(0)).flatten()
        expected_t_per_update = np.arange(1, self.max_epoch+1)
        assert_array_equal(t_per_update, expected_t_per_update)

        # Directions should not be the same as gradients at first.
        for i in range(self.max_epoch):
            assert_true(not np.allclose(abs(self.logger[i][2]), abs(self.logger[i][3])))

        std_per_update = np.array(self.logger.get_variable_history(1)).flatten()
        # std is expected to decay at each update.
        assert_true(np.all(np.diff(std_per_update) < 0))

    def test_save_load(self):
        # Save training and resume it.
        with tempfile.TemporaryDirectory() as experiment_dir:
            # Save current training state of the experiment.
            self.trainer.save(experiment_dir)

            # Load previous training state of the experiment.
            trainer, logger, direction_modifier = self._build_experiment()
            trainer.load(experiment_dir)

            assert_equal(direction_modifier._eta, self.direction_modifier._eta)
            assert_equal(direction_modifier._gamma, self.direction_modifier._gamma)
            assert_equal(direction_modifier.t.get_value(), self.direction_modifier.t.get_value())
            assert_equal(direction_modifier.std.get_value(), self.direction_modifier.std.get_value())
            assert_array_equal(direction_modifier._srng.rstate, self.direction_modifier._srng.rstate)

            for state, expected_state in zip(direction_modifier._srng.state_updates,
                                             self.direction_modifier._srng.state_updates):
                assert_array_equal(state[0].get_value(), expected_state[0].get_value())

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
        learning_rate_per_update_part1 = np.array(logger1.get_variable_history(0)).flatten()
        learning_rate_per_update_part2 = np.array(logger2.get_variable_history(0)).flatten()
        expected_learning_rate_per_update = np.array(self.logger.get_variable_history(0)).flatten()
        assert_array_equal(np.r_[learning_rate_per_update_part1, learning_rate_per_update_part2],
                           expected_learning_rate_per_update)

        # Check that concatenating `logger1` with `logger2` is the same as `self.logger`.
        learning_rate_per_update_part1 = np.array(logger1.get_variable_history(1)).flatten()
        learning_rate_per_update_part2 = np.array(logger2.get_variable_history(1)).flatten()
        expected_learning_rate_per_update = np.array(self.logger.get_variable_history(1)).flatten()
        assert_array_equal(np.r_[learning_rate_per_update_part1, learning_rate_per_update_part2],
                           expected_learning_rate_per_update)

        # Check that concatenating `logger1` with `logger2` is the same as `self.logger`.
        learning_rate_per_update_part1 = np.array(logger1.get_variable_history(2)).flatten()
        learning_rate_per_update_part2 = np.array(logger2.get_variable_history(2)).flatten()
        expected_learning_rate_per_update = np.array(self.logger.get_variable_history(2)).flatten()
        assert_array_almost_equal(np.r_[learning_rate_per_update_part1, learning_rate_per_update_part2],
                                  expected_learning_rate_per_update)
