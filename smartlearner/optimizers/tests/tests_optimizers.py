import numpy as np

import theano.tensor as T

from smartlearner.tasks import stopping_criteria
from smartlearner.utils import sharedX

from smartlearner import Trainer
from smartlearner.optimizers import SGD, ADAGRAD
from smartlearner.batch_scheduler import BatchScheduler

from smartlearner.tasks import tasks
from smartlearner.interfaces.loss import Loss

from numpy.testing import assert_almost_equal


class DummyLoss(Loss):
    def __init__(self, cost, param):
        super().__init__(model=None, dataset=None)
        self.cost = cost
        self.param = param

    @property
    def gradients(self):
        gparam = T.grad(cost=self.cost, wrt=self.param)
        gradients = {self.param: gparam}
        return gradients


class DummyBatchScheduler(BatchScheduler):
    def __init__(self, nb_updates):
        self.nb_updates = nb_updates

    @property
    def givens(self):
        return {}

    def __iter__(self):
        return iter(range(self.nb_updates))


def test_sgd():
    max_epoch = 30

    # Create simple Nd gaussian functions to optimize.
    for N in range(1, 5):
        center = T.arange(N)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.3 * T.dot(T.dot((param-center), T.eye(N)), ((param-center).T)))
        loss = DummyLoss(cost, param)

        batch_scheduler = DummyBatchScheduler(nb_updates=1)
        trainer = Trainer(SGD(loss), batch_scheduler)
        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))
        #trainer.append_task(tasks.PrintVariable("Loss param: {}", param))
        trainer.append_task(tasks.PrintVariable("Loss gradient: {}", loss.gradients[param]))

        # Monitor the gradient of `loss` w.r.t. to `param`.
        gparam = tasks.MonitorVariable(loss.gradients[param])
        trainer.append_task(gparam)
        trainer.train()

        # After 30 epochs, param should be around the center and gradients near 0.
        assert_almost_equal(param.get_value(), np.arange(N)[None, :])
        assert_almost_equal(gparam.value, 0.)


def test_adagrad():
    max_epoch = 30

    # Create simple Nd gaussian functions to optimize.
    for N in range(1, 5):
        center = T.arange(N)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.3 * T.dot(T.dot((param-center), T.eye(N)), ((param-center).T)))
        loss = DummyLoss(cost, param)

        batch_scheduler = DummyBatchScheduler(nb_updates=1)
        trainer = Trainer(ADAGRAD(loss, lr=0.1, eps=1e-2), batch_scheduler)
        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch*N))
        #trainer.append_task(tasks.PrintVariable("Loss param: {}", param))
        #trainer.append_task(tasks.PrintVariable("Loss gradient: {}", loss.gradients[param]))

        # Monitor the gradient of `loss` w.r.t. to `param`.
        gparam = tasks.MonitorVariable(loss.gradients[param])
        trainer.append_task(gparam)
        trainer.train()

        # After 30 epochs, param should be around the center and gradients near 0.
        assert_almost_equal(param.get_value(), np.arange(N)[None, :])
        assert_almost_equal(gparam.value, 0.)
