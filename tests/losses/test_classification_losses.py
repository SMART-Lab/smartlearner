import numpy as np
import theano.tensor as T

from numpy.testing import assert_equal
from smartlearner.testing import DummyDataset, DummyModel

from smartlearner.interfaces import Loss
from smartlearner.losses.classification_losses import CategoricalCrossEntropy
from smartlearner.losses.classification_losses import NegativeLogLikelihood
from smartlearner.losses.classification_losses import ClassificationError


def test_categorical_cross_entropy():
    batch_size = 13
    output_dim = 32

    rng = np.random.RandomState(42)
    targets = rng.randint(output_dim, size=(batch_size, 1))

    dataset = DummyDataset()
    dataset.symb_targets = targets

    loss = CategoricalCrossEntropy(DummyModel(), dataset)
    model_output = rng.rand(batch_size, output_dim).astype(np.float32)

    # Test the shape of the output of _compute_losses.
    losses = loss._compute_losses(model_output).eval()
    assert_equal(losses.shape, (batch_size, ))


def test_negative_log_likelihood():
    batch_size = 13
    output_dim = 32

    rng = np.random.RandomState(42)
    targets = rng.randint(output_dim, size=(batch_size, 1))

    dataset = DummyDataset()
    dataset.symb_targets = targets

    loss = NegativeLogLikelihood(DummyModel(), dataset)
    model_output = rng.rand(batch_size, output_dim).astype(np.float32)

    # Test the shape of the output of _compute_losses.
    losses = loss._compute_losses(model_output).eval()
    assert_equal(losses.shape, (batch_size, ))


def test_classification_error():
    batch_size = 13
    output_dim = 32

    rng = np.random.RandomState(42)
    targets = rng.randint(output_dim, size=(batch_size, 1))

    dataset = DummyDataset()
    dataset.symb_targets = targets

    loss = ClassificationError(DummyModel(), dataset)
    model_output = rng.rand(batch_size, output_dim).astype(np.float32)

    # Test the shape of the output of _compute_losses.
    losses = loss._compute_losses(model_output).eval()
    assert_equal(losses.shape, (batch_size, ))