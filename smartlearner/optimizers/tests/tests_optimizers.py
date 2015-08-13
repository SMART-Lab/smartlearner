import numpy as np

import theano.tensor as T

from smartlearner.tasks import stopping_criteria
from smartlearner.utils import sharedX

from smartlearner import Trainer, Dataset
from smartlearner.optimizers import SGD, ADAGRAD
from smartlearner.batch_scheduler import BatchScheduler

from smartlearner.tasks.views import View
from smartlearner.tasks import tasks
from smartlearner.interfaces.loss import Loss

import unittest
from nose.tools import assert_false
from numpy.testing import assert_equal


class DummyLoss(Loss):
    def __init__(self):
        super().__init__(model=None, dataset=None)
        self.param = sharedX(np.array(0))

    @property
    def gradients(self):
        cost = (self.param-1)**2
        gparam = T.grad(cost=cost, wrt=self.param)
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
    max_epoch = 5

    loss = DummyLoss()
    optimizer = SGD(loss)
    #optimizer.append_update_rule(ConstantLearningRate(1))

    batch_scheduler = DummyBatchScheduler(nb_updates=1)
    trainer = Trainer(optimizer, batch_scheduler)
    trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))
    trainer.append_task(tasks.PrintVariable("Loss gradient: {}", loss.gradients[loss.param]))
    trainer.append_task(tasks.PrintVariable("Loss param: {}", loss.param))
    trainer.train()

    assert_equal(trainer.status.current_epoch, max_epoch)


# def test_adagrad():
#     MAX_EPOCH = 100  # Add a max epoch just in case we got an infinite loop.

#     class DummyCost(View):
#         def __init__(self, initial_cost, costs):
#             super().__init__()
#             self.initial_cost = initial_cost
#             self.costs = costs
#             self.cpt = 0

#         def update(self, status):
#             if status.current_update == 0:
#                 return self.initial_cost

#             cost = self.costs[self.cpt]
#             self.cpt += 1
#             return cost

#     # 20 identical costs but should stop after 9 unchanged epochs.
#     constant_cost = DummyCost(1, np.ones(20))
#     lookahead = 9

#     def callback(task, status):
#         # This callback function should not be called.
#         raise NameError("Callback should be fired up at epoch #1.")

#     early_stopping = stopping_criteria.EarlyStopping(constant_cost, lookahead, callback=callback)

#     trainer = Trainer(DummyOptimizer(), DummyBatchScheduler(nb_updates=5))
#     trainer.append_task(early_stopping)
#     trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
#     trainer.train()

#     assert_equal(trainer.status.current_epoch, lookahead)
#     assert_equal(early_stopping.best_epoch, 0)
#     assert_equal(early_stopping.best_cost, 1.)
#     assert_equal(constant_cost.cpt, lookahead)
#     assert_false(trainer.status.current_epoch >= MAX_EPOCH)

#     # `lookahead` identical costs than `lookahead` lower identical costs.
#     lookahead = 9
#     costs = np.r_[np.ones(lookahead-1), np.zeros(lookahead+1)]
#     simple_cost = DummyCost(1, costs)

#     def callback(task, status):
#         # This callback function should be called once after `lookahead` epoch.
#         if status.current_epoch != lookahead:
#             msg = "Callback should be fired up at epoch #{} not #{}.".format(lookahead, status.current_epoch)
#             raise NameError(msg)

#     early_stopping = stopping_criteria.EarlyStopping(simple_cost, lookahead, callback=callback)

#     trainer = Trainer(DummyOptimizer(), DummyBatchScheduler(nb_updates=5))
#     trainer.append_task(early_stopping)
#     trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
#     trainer.train()

#     assert_equal(trainer.status.current_epoch, 2*lookahead)
#     assert_equal(early_stopping.best_epoch, lookahead)
#     assert_equal(early_stopping.best_cost, 0.)
#     assert_false(trainer.status.current_epoch >= MAX_EPOCH)

#     # 20 increasing costs but should stop after 9 unchanged epochs.
#     lookahead = 9
#     costs = range(20)
#     increasing_cost = DummyCost(0, costs)

#     def callback(task, status):
#         # This callback function should not be called.
#         raise NameError("Callback should be fired up at epoch #1.")

#     early_stopping = stopping_criteria.EarlyStopping(increasing_cost, lookahead, callback=callback)

#     trainer = Trainer(DummyOptimizer(), DummyBatchScheduler(nb_updates=5))
#     trainer.append_task(early_stopping)
#     trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
#     trainer.train()

#     assert_equal(trainer.status.current_epoch, lookahead)
#     assert_equal(early_stopping.best_epoch, 0)
#     assert_equal(early_stopping.best_cost, 0.)
#     assert_false(trainer.status.current_epoch >= MAX_EPOCH)
