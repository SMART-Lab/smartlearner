import os

import numpy as np
import theano
import theano.tensor as T
import tempfile
from os.path import join as pjoin

from numpy.testing import assert_equal, assert_array_equal

from smartlearner import Trainer, Dataset, Model
from smartlearner import tasks
from smartlearner import views
from smartlearner import stopping_criteria
import smartlearner.initializers as initer
from smartlearner.utils import sharedX
from smartlearner import utils
from smartlearner.optimizers import SGD
from smartlearner.direction_modifiers import ConstantLearningRate
from smartlearner.batch_schedulers import FullBatchScheduler, MiniBatchScheduler
from smartlearner.losses.classification_losses import NegativeLogLikelihood as NLL
from smartlearner.losses.classification_losses import ClassificationError

DATASETS_ENV = 'DATASETS'


def load_mnist():
    dataset_name = "mnist"
    datasets_repo = os.environ.get(DATASETS_ENV, os.path.join(os.environ["HOME"], '.smartdatasets'))
    if not os.path.isdir(datasets_repo):
        os.mkdir(datasets_repo)

    repo = os.path.join(datasets_repo, dataset_name)
    dataset_npy = os.path.join(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(repo):
            os.mkdir(repo)

        import urllib.request
        urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_train.txt', os.path.join(repo, 'mnist_train.txt'))
        urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_valid.txt', os.path.join(repo, 'mnist_valid.txt'))
        urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_test.txt', os.path.join(repo, 'mnist_test.txt'))

        train_file, valid_file, test_file = [os.path.join(repo, 'mnist_' + ds + '.txt') for ds in ['train', 'valid', 'test']]

        def parse_file(filename):
            return np.array([np.fromstring(l, dtype=np.float32, sep=" ") for l in open(filename)])

        trainset, validset, testset = parse_file(train_file), parse_file(valid_file), parse_file(test_file)
        trainset_inputs, trainset_targets = trainset[:, :-1], trainset[:, [-1]]
        validset_inputs, validset_targets = validset[:, :-1], validset[:, [-1]]
        testset_inputs, testset_targets = testset[:, :-1], testset[:, [-1]]

        np.savez(dataset_npy,
                 trainset_inputs=trainset_inputs, trainset_targets=trainset_targets,
                 validset_inputs=validset_inputs, validset_targets=validset_targets,
                 testset_inputs=testset_inputs, testset_targets=testset_targets)

    data = np.load(dataset_npy)
    trainset = Dataset(data['trainset_inputs'].astype(theano.config.floatX), data['trainset_targets'].astype(theano.config.floatX))
    validset = Dataset(data['validset_inputs'].astype(theano.config.floatX), data['validset_targets'].astype(theano.config.floatX))
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), data['testset_targets'].astype(theano.config.floatX))

    return trainset, validset, testset


class Perceptron(Model):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = sharedX(value=np.zeros((input_size, output_size)), name='W', borrow=True)
        self.b = sharedX(value=np.zeros(output_size), name='b', borrow=True)

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        weights_initializer(self.W)

    @property
    def updates(self):
        return {}  # No updates.

    @property
    def parameters(self):
        return [self.W, self.b]

    def get_output(self, X):
        preactivation = T.dot(X, self.W) + self.b
        probs = T.nnet.softmax(preactivation)
        return probs

    def use(self, X):
        probs = self.get_output(X)
        return T.argmax(probs, axis=1, keepdims=True)

    def save(self, path):
        savedir = utils.create_folder(pjoin(path, "model"))
        hyperparameters = {'input_size': self.input_size,
                           'output_size': self.output_size}
        utils.save_dict_to_json_file(pjoin(savedir, "meta.json"), {"name": self.__class__.__name__})
        utils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), hyperparameters)

        params = {param.name: param.get_value() for param in self.parameters}
        np.savez(pjoin(savedir, "params.npz"), **params)

    def load(self, path):
        loaddir = pjoin(path, "model")
        parameters = np.load(pjoin(loaddir, "params.npz"))
        for param in self.parameters:
            param.set_value(parameters[param.name])

    @classmethod
    def create(cls, path):
        loaddir = pjoin(path, "model")
        meta = utils.load_dict_from_json_file(pjoin(loaddir, "meta.json"))
        assert meta['name'] == cls.__name__

        hyperparams = utils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        model = cls(**hyperparams)
        model.load(path)
        return model


def test_simple_perceptron():
    #Loading dataset
    trainset, validset, testset = load_mnist()

    #Creating model
    nb_classes = 10
    model = Perceptron(trainset.input_size, nb_classes)
    model.initialize()  # By default, uniform initialization.

    #Building optimizer
    loss = NLL(model, trainset)
    optimizer = SGD(loss=loss)
    optimizer.append_direction_modifier(ConstantLearningRate(0.1))

    # Use mini batches of 100 examples.
    batch_scheduler = MiniBatchScheduler(trainset, 100)

    # Build trainer and add some tasks.
    trainer = Trainer(optimizer, batch_scheduler)

    # Print time for one epoch
    trainer.append_task(tasks.PrintEpochDuration())
    trainer.append_task(tasks.PrintTrainingDuration())

    # Log training error
    loss_monitor = views.MonitorVariable(loss.loss)
    avg_loss = tasks.AveragePerEpoch(loss_monitor)
    accum = tasks.Accumulator(loss_monitor)
    logger = tasks.Logger(loss_monitor, avg_loss)
    trainer.append_task(logger, avg_loss, accum)

    # Print NLL mean/stderror.
    nll = views.LossView(loss=NLL(model, validset), batch_scheduler=FullBatchScheduler(validset))
    trainer.append_task(tasks.Print("Validset - NLL          : {0:.1%} ± {1:.1%}",
                                    nll.mean, nll.stderror))

    # Print mean/stderror of classification errors.
    classif_error = views.LossView(loss=ClassificationError(model, validset),
                                   batch_scheduler=FullBatchScheduler(validset))
    trainer.append_task(tasks.Print("Validset - Classif error: {0:.1%} ± {1:.1%}",
                                    classif_error.mean, classif_error.stderror))

    # Train for 10 epochs (stopping criteria should be added at the end).
    trainer.append_task(stopping_criteria.MaxEpochStopping(10))
    trainer.train()


def test_resume_experiment():
    #Loading dataset
    trainset, validset, testset = load_mnist()

    #Creating model
    nb_classes = 10
    model1 = Perceptron(trainset.input_size, nb_classes)
    model1.initialize(initer.UniformInitializer(random_seed=1234))

    #Building optimizer
    loss1 = NLL(model1, trainset)
    optimizer1 = SGD(loss=loss1)
    optimizer1.append_direction_modifier(ConstantLearningRate(0.1))

    # Use mini batches of 100 examples.
    batch_scheduler1 = MiniBatchScheduler(trainset, 100)

    # Build trainer and add some tasks.
    trainer1 = Trainer(optimizer1, batch_scheduler1)

    # Print time for one epoch
    trainer1.append_task(tasks.PrintEpochDuration())
    trainer1.append_task(tasks.PrintTrainingDuration())

    # Log training error
    loss_monitor1 = views.MonitorVariable(loss1.loss)
    avg_loss1 = tasks.AveragePerEpoch(loss_monitor1)

    # Print NLL mean/stderror.
    nll1 = views.LossView(loss=NLL(model1, validset), batch_scheduler=FullBatchScheduler(validset))
    logger1 = tasks.Logger(loss_monitor1, avg_loss1, nll1.mean)
    trainer1.append_task(logger1, avg_loss1)

    # Train for 10 epochs (stopping criteria should be added at the end).
    trainer1.append_task(stopping_criteria.MaxEpochStopping(10))

    trainer1.train()

    # Redo training but stop at epoch #5, save and resume.

    #Creating model
    model2a = Perceptron(trainset.input_size, nb_classes)
    model2a.initialize(initer.UniformInitializer(random_seed=1234))

    #Building optimizer
    loss2a = NLL(model2a, trainset)
    optimizer2a = SGD(loss=loss2a)
    optimizer2a.append_direction_modifier(ConstantLearningRate(0.1))

    # Use mini batches of 100 examples.
    batch_scheduler2a = MiniBatchScheduler(trainset, 100)

    # Build trainer and add some tasks.
    trainer2a = Trainer(optimizer2a, batch_scheduler2a)

    # Print time for one epoch
    trainer2a.append_task(tasks.PrintEpochDuration())
    trainer2a.append_task(tasks.PrintTrainingDuration())

    # Log training error
    loss_monitor2a = views.MonitorVariable(loss2a.loss)
    avg_loss2a = tasks.AveragePerEpoch(loss_monitor2a)

    # Print NLL mean/stderror.
    nll2a = views.LossView(loss=NLL(model2a, validset), batch_scheduler=FullBatchScheduler(validset))
    logger2a = tasks.Logger(loss_monitor2a, avg_loss2a, nll2a.mean)
    trainer2a.append_task(logger2a, avg_loss2a)

    # Train for 5 epochs (stopping criteria should be added at the end).
    trainer2a.append_task(stopping_criteria.MaxEpochStopping(5))

    # Will stop training after 5 epochs
    trainer2a.train()

    # Simulate resuming an experiment.

    #Creating model
    model2b = Perceptron(trainset.input_size, nb_classes)
    model2b.initialize(initer.UniformInitializer(random_seed=1234))

    #Building optimizer
    loss2b = NLL(model2b, trainset)
    optimizer2b = SGD(loss=loss2b)
    optimizer2b.append_direction_modifier(ConstantLearningRate(0.1))

    # Use mini batches of 100 examples.
    batch_scheduler2b = MiniBatchScheduler(trainset, 100)

    # Build trainer and add some tasks.
    trainer2b = Trainer(optimizer2b, batch_scheduler2b)

    # Print time for one epoch
    trainer2b.append_task(tasks.PrintEpochDuration())
    trainer2b.append_task(tasks.PrintTrainingDuration())

    # Log training error
    loss_monitor2b = views.MonitorVariable(loss2b.loss)
    avg_loss2b = tasks.AveragePerEpoch(loss_monitor2b)

    # Print NLL mean/stderror.
    nll2b = views.LossView(loss=NLL(model2b, validset), batch_scheduler=FullBatchScheduler(validset))
    logger2b = tasks.Logger(loss_monitor2b, avg_loss2b, nll2b.mean)
    trainer2b.append_task(logger2b, avg_loss2b)

    # Train for 10 epochs (stopping criteria should be added at the end).
    trainer2b.append_task(stopping_criteria.MaxEpochStopping(10))

    # Save model halfway during training and resume it.
    with tempfile.TemporaryDirectory() as experiment_dir:
        # Save current state of the model (i.e. after 5 epochs).
        trainer2a.save(experiment_dir)
        # Load previous state from which training will resume.
        trainer2b.load(experiment_dir)

    # Check that the `model` state after loading matches the one saved.
    assert_array_equal(model2b.W.get_value(), model2a.W.get_value())
    assert_array_equal(model2b.b.get_value(), model2a.b.get_value())

    # Check that the `status` state after loading matches the one saved.
    assert_equal(trainer2b.status.current_epoch, trainer2a.status.current_epoch)
    assert_equal(trainer2b.status.current_update, trainer2a.status.current_update)
    assert_equal(trainer2b.status.current_update_in_epoch, trainer2a.status.current_update_in_epoch)
    assert_equal(trainer2b.status.training_time, trainer2a.status.training_time)
    assert_equal(trainer2b.status.done, trainer2a.status.done)
    assert_equal(trainer2b.status.extra, trainer2a.status.extra)

    # Check that the `batch_scheduler` state after loading matches the one saved.
    assert_equal(trainer2b._batch_scheduler.batch_size, trainer2a._batch_scheduler.batch_size)
    assert_equal(trainer2b._batch_scheduler.shared_batch_count.get_value(),
                 trainer2a._batch_scheduler.shared_batch_count.get_value())

    # Resume training
    trainer2b.train()

    # Check that the `model` state after completing the training matches the one trained in one shot.
    assert_array_equal(model2b.W.get_value(), model1.W.get_value())
    assert_array_equal(model2b.b.get_value(), model1.b.get_value())

    # Check that the `status` state after completing the training matches the one trained in one shot.
    assert_equal(trainer2b.status.current_epoch, trainer1.status.current_epoch)
    assert_equal(trainer2b.status.current_update, trainer1.status.current_update)
    assert_equal(trainer2b.status.current_update_in_epoch, trainer1.status.current_update_in_epoch)
    assert_equal(trainer2b.status.done, trainer1.status.done)

    # Check that the `batch_scheduler` state after completing the training matches the one trained in one shot.
    assert_equal(trainer2b._batch_scheduler.shared_batch_count.get_value(),
                 trainer1._batch_scheduler.shared_batch_count.get_value())

    # Check that concatenating `logger2a` with `logger2b` is the same as `logger1`.
    for i in range(len(logger1[0])):
        assert_equal(logger2a._history[i] + logger2b._history[i], logger1._history[i])
