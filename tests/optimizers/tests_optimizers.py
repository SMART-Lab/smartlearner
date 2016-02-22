import numpy as np
import theano
import theano.tensor as T
from numpy.testing import assert_array_almost_equal

from smartlearner import views, stopping_criteria, Trainer, tasks
from smartlearner.optimizers import SGD, AdaGrad, Adam, RMSProp, Adadelta
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


def test_sgd():
    # Create simple Nd gaussian functions to optimize. These functions are
    # (perfectly) well-conditioned so it should take only one gradient step
    # to converge using 1/L, where L is the largest eigenvalue of the hessian.
    max_epoch = 2
    for N in range(1, 5):
        center = np.arange(1, N+1)[None, :].astype(floatX)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.5*T.dot(T.dot((param-center), T.eye(N)), (param-center).T))
        loss = DummyLossWithGradient(cost, param)

        trainer = Trainer(SGD(loss), DummyBatchScheduler())

        # Monitor the gradient of `loss` w.r.t. to `param`.
        tracker = tasks.Tracker(loss.gradients[param])
        trainer.append_task(tracker)

        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))
        trainer.train()

        # Since the problem is well-conditionned and we use an optimal gradient step 1/L,
        # two epochs should be enough for `param` to be around `center` and the gradients near 0.
        assert_array_almost_equal(param.get_value(), center)
        assert_array_almost_equal(tracker[0], 0.)

    # Create an Nd gaussian function to optimize. This function is not
    # well-conditioned and there exists no perfect gradient step to converge in
    # only one iteration.
    # cost = T.sum(N*0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), ((param-center).T)))
    max_epoch = 80
    N = 4
    center = 5*np.ones((1, N)).astype(floatX)
    param = sharedX(np.zeros((1, N)))
    cost = T.sum(0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), (param-center).T))
    loss = DummyLossWithGradient(cost, param)

    trainer = Trainer(SGD(loss), DummyBatchScheduler())
    trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))

    # Monitor the gradient of `loss` w.r.t. to `param`.
    tracker = tasks.Tracker(loss.gradients[param])
    trainer.append_task(tracker)
    trainer.train()

    # Since the problem is well-conditionned and we use an optimal gradient step 1/L,
    # two epochs should be enough for `param` to be around `center` and the gradients near 0.
    assert_array_almost_equal(param.get_value(), center, decimal=6)
    assert_array_almost_equal(tracker[0], 0.)


def test_adagrad():
    max_epoch = 15

    # Create an Nd gaussian functions to optimize. These functions are not
    # well-conditioned and there exists no perfect gradient step to converge in
    # only one iteration.
    for N in range(1, 5):
        center = 5*np.ones((1, N)).astype(floatX)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), ((param-center).T)))
        loss = DummyLossWithGradient(cost, param)

        # Even with a really high gradient step, AdaGrad can still converge.
        # Actually, it is faster than using the optimal gradient step with SGD.
        optimizer = AdaGrad(loss, lr=100, eps=1e-1)
        trainer = Trainer(optimizer, DummyBatchScheduler())
        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))

        # Monitor the gradient of `loss` w.r.t. to `param`.
        tracker = tasks.Tracker(loss.gradients[param])
        trainer.append_task(tracker)
        trainer.train()

        # After 15 epochs, param should be around the center and gradients near 0.
        assert_array_almost_equal(param.get_value(), center)
        assert_array_almost_equal(tracker[0], 0.)


def test_adam():
    max_epoch = 300

    # Create an Nd gaussian functions to optimize. These functions are not
    # well-conditioned and there exists no perfect gradient step to converge in
    # only one iteration.
    for N in range(1, 5):
        center = 5*np.ones((1, N)).astype(floatX)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), ((param-center).T)))
        loss = DummyLossWithGradient(cost, param)

        # Even with a really high gradient step, Adam can still converge.
        optimizer = Adam(loss, lr=1)
        trainer = Trainer(optimizer, DummyBatchScheduler())
        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))

        # Monitor the gradient of `loss` w.r.t. to `param`.
        tracker = tasks.Tracker(loss.gradients[param])
        trainer.append_task(tracker)
        trainer.train()

        # After 300 epochs, param should be around the center and gradients near 0.
        assert_array_almost_equal(param.get_value(), center)
        assert_array_almost_equal(tracker[0], 0.)


def test_rmsprop():
    max_epoch = 10

    # Create an Nd gaussian functions to optimize. These functions are not
    # well-conditioned and there exists no perfect gradient step to converge in
    # only one iteration.
    for N in range(1, 5):
        center = 5*np.ones((1, N)).astype(floatX)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), ((param-center).T)))
        loss = DummyLossWithGradient(cost, param)

        # Even with a really high gradient step, RMSProp can still converge.
        optimizer = RMSProp(loss, lr=1)
        trainer = Trainer(optimizer, DummyBatchScheduler())
        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))

        # Monitor the gradient of `loss` w.r.t. to `param`.
        tracker = tasks.Tracker(loss.gradients[param])
        trainer.append_task(tracker)
        trainer.train()

        # After 10 epochs, param should be around the center and gradients near 0.
        assert_array_almost_equal(param.get_value(), center)
        assert_array_almost_equal(tracker[0], 0.)


def test_adadelta():
    max_epoch = 1500

    # Create an Nd gaussian functions to optimize. These functions are not
    # well-conditioned and there exists no perfect gradient step to converge in
    # only one iteration.
    for N in range(1, 5):
        center = 5*np.ones((1, N)).astype(floatX)
        param = sharedX(np.zeros((1, N)))
        cost = T.sum(0.5*T.dot(T.dot((param-center), np.diag(1./np.arange(1, N+1))), ((param-center).T)))
        loss = DummyLossWithGradient(cost, param)

        # Adadelta requires no learning rate.
        optimizer = Adadelta(loss)
        trainer = Trainer(optimizer, DummyBatchScheduler())
        trainer.append_task(stopping_criteria.MaxEpochStopping(max_epoch))

        # Monitor the gradient of `loss` w.r.t. to `param`.
        tracker = tasks.Tracker(loss.gradients[param])
        trainer.append_task(tracker)
        trainer.train()

        # After 1500 epochs, param should be around the center and gradients near 0.
        assert_array_almost_equal(param.get_value(), center)
        assert_array_almost_equal(tracker[0], 0.)
