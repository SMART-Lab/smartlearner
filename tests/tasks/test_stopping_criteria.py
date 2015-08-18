import numpy as np

from smartlearner import Trainer
from smartlearner.interfaces import View

from smartlearner import tasks
from smartlearner import stopping_criteria
from smartlearner.utils import sharedX

from smartlearner.testing import DummyOptimizer, DummyBatchScheduler

from numpy.testing import assert_equal, assert_array_equal


def test_max_epoch_stopping():
    max_epoch = 7
    trainer = Trainer(DummyOptimizer(), DummyBatchScheduler())
    trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))
    trainer.train()

    assert_equal(trainer.status.current_epoch, max_epoch)


def test_early_stopping():
    MAX_EPOCH = 100  # Add a max epoch just in case we got an infinite loop.

    class DummyCost(View):
        def __init__(self, initial_cost, costs):
            super().__init__()
            self.initial_cost = initial_cost
            self.costs = costs
            self.cpt = 0

        def update(self, status):
            if status.current_update == 0:
                return self.initial_cost

            cost = self.costs[self.cpt]
            self.cpt += 1
            return cost

    # 20 identical costs but should stop after 9 unchanged epochs.
    constant_cost = DummyCost(1, np.ones(20))
    lookahead = 9

    def callback(task, status):
        # This callback function should not be called.
        raise NameError("This callback function should not be called.")

    early_stopping = stopping_criteria.EarlyStopping(constant_cost, lookahead, callback=callback)

    trainer = Trainer(DummyOptimizer(), DummyBatchScheduler())
    trainer.append_task(early_stopping)
    trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
    trainer.train()

    assert_equal(trainer.status.current_epoch, lookahead)
    assert_equal(early_stopping.best_epoch, 0)
    assert_equal(early_stopping.best_cost, 1.)
    assert_equal(constant_cost.cpt, lookahead)

    # `lookahead` identical costs followed by `lookahead` lower identical costs.
    lookahead = 9
    costs = np.r_[np.ones(lookahead-1), np.zeros(lookahead+1)]
    simple_cost = DummyCost(1, costs)

    def callback(task, status):
        # This callback function should be called once after `lookahead` epoch.
        if status.current_epoch != lookahead:
            msg = "Callback should be fired up at epoch #{} not #{}.".format(lookahead, status.current_epoch)
            raise NameError(msg)

    early_stopping = stopping_criteria.EarlyStopping(simple_cost, lookahead, callback=callback)

    trainer = Trainer(DummyOptimizer(), DummyBatchScheduler())
    trainer.append_task(early_stopping)
    trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
    trainer.train()

    assert_equal(trainer.status.current_epoch, 2*lookahead)
    assert_equal(early_stopping.best_epoch, lookahead)
    assert_equal(early_stopping.best_cost, 0.)

    # 20 increasing costs but should stop after 9 increasing epochs.
    lookahead = 9
    costs = range(20)
    increasing_cost = DummyCost(0, costs)

    def callback(task, status):
        # This callback function should not be called.
        raise NameError("This callback function should not be called.")

    early_stopping = stopping_criteria.EarlyStopping(increasing_cost, lookahead, callback=callback)

    trainer = Trainer(DummyOptimizer(), DummyBatchScheduler())
    trainer.append_task(early_stopping)
    trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
    trainer.train()

    assert_equal(trainer.status.current_epoch, lookahead)
    assert_equal(early_stopping.best_epoch, 0)
    assert_equal(early_stopping.best_cost, 0.)

    # Test `min_nb_epochs`
    lookahead = 9
    min_nb_epochs = 5
    costs = range(20)
    increasing_cost = DummyCost(0, costs)
    early_stopping = stopping_criteria.EarlyStopping(increasing_cost, lookahead, min_nb_epochs=min_nb_epochs)

    trainer = Trainer(DummyOptimizer(), DummyBatchScheduler())
    trainer.append_task(early_stopping)
    trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
    trainer.train()

    assert_equal(trainer.status.current_epoch, lookahead+min_nb_epochs)

    # Test that at the end the model is the best one.
    # `lookahead` decreasing costs followed by `lookahead+1` constant identical costs.
    lookahead = 9
    costs = np.r_[-np.arange(lookahead), np.zeros(lookahead+1)]
    simple_cost = DummyCost(1, costs)

    trainer = Trainer(DummyOptimizer(), DummyBatchScheduler())
    model = trainer._optimizer.loss.model
    # Add some parameters to the model.
    model.parameters.extend([sharedX(np.zeros(4)), sharedX(np.zeros((3, 5)))])

    # Callback that will change model parameters after each epoch.
    def callback(task, status):
        for param in model.parameters:
            param.set_value(param.get_value() + 1)

    trainer.append_task(tasks.Callback(callback))

    early_stopping = stopping_criteria.EarlyStopping(simple_cost, lookahead)
    trainer.append_task(early_stopping)
    trainer.append_task(stopping_criteria.MaxEpochStopping(MAX_EPOCH))  # To be safe
    trainer.train()

    for param in model.parameters:
        assert_array_equal(param.get_value(), lookahead*np.ones_like(param.get_value()))
