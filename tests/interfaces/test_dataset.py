import numpy as np
import theano
import theano.tensor as T

from nose.tools import assert_true
from numpy.testing import assert_equal, assert_array_equal

from smartlearner.interfaces.dataset import Dataset

floatX = theano.config.floatX
ALL_DTYPES = np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']


def test_dataset_used_in_theano_function():
    rng = np.random.RandomState(1234)

    nb_examples = 10

    inputs = (rng.randn(nb_examples, 5) * 100).astype(floatX)
    targets = (rng.randn(nb_examples, 1) > 0.5).astype(floatX)
    dataset = Dataset(inputs, targets)

    input_sqr_norm = T.sum(dataset.symb_inputs**2)
    result = input_sqr_norm - dataset.symb_targets
    f = theano.function([dataset.symb_inputs, dataset.symb_targets], result)

    assert_array_equal(f(inputs, targets), np.sum(inputs**2)-targets)


def test_dataset_without_targets():
    rng = np.random.RandomState(1234)

    nb_examples = 10
    nb_features = 3
    sequences_length = 4
    nb_channels = 2
    image_shape = (5, 5)

    # Test creating dataset with different example shapes:
    # scalar feature, vector features, sequence of vector features, multiple channels images features.
    for example_shape in [(), (nb_features,), (sequences_length, nb_features), (nb_channels,)+image_shape]:
        inputs_shape = (nb_examples,) + example_shape

        for dtype in ALL_DTYPES:
            inputs = (rng.randn(*inputs_shape) * 100).astype(dtype)
            dataset = Dataset(inputs)

            # Data should be converted into `floatX`.
            assert_equal(dataset.inputs.dtype, floatX)
            assert_equal(dataset.symb_inputs.dtype, floatX)
            assert_equal(dataset.symb_inputs.ndim, inputs.ndim)
            assert_equal(dataset.input_shape, example_shape)
            assert_array_equal(dataset.inputs.get_value(), inputs.astype(floatX))

            # Everything related to target should be None
            assert_true(dataset.targets is None)
            assert_true(dataset.symb_targets is None)
            assert_true(dataset.target_shape is None)
            assert_true(dataset.target_size is None)

    # Create dataset from nested Pyton lists.
    inputs = [[1, 2, 3]] * nb_examples
    dataset = Dataset(inputs)
    # Data should be converted into `floatX`.
    assert_equal(dataset.inputs.dtype, floatX)
    assert_equal(dataset.symb_inputs.dtype, floatX)
    assert_equal(dataset.symb_inputs.ndim, 2)
    assert_equal(dataset.input_shape, (3,))
    assert_array_equal(dataset.inputs.get_value(), np.array(inputs, dtype=floatX))


def test_dataset_with_targets():
    rng = np.random.RandomState(1234)

    nb_examples = 10
    nb_features = 3
    sequences_length = 4
    nb_channels = 2
    image_shape = (5, 5)

    # Test creating dataset with different example shapes and target shapes:
    # scalar feature, vector features, sequence of vector features, multiple channels images features.
    for target_shape in [(), (nb_features,), (sequences_length, nb_features), (nb_channels,)+image_shape]:
        for example_shape in [(), (nb_features,), (sequences_length, nb_features), (nb_channels,)+image_shape]:
            inputs_shape = (nb_examples,) + example_shape
            targets_shape = (nb_examples,) + target_shape

            for example_dtype in ALL_DTYPES:
                for target_dtype in ALL_DTYPES:
                    inputs = (rng.randn(*inputs_shape) * 100).astype(example_dtype)
                    targets = (rng.randn(*targets_shape) * 100).astype(target_dtype)
                    dataset = Dataset(inputs, targets)

                    # Data should be converted into `floatX`.
                    assert_equal(dataset.inputs.dtype, floatX)
                    assert_equal(dataset.symb_inputs.dtype, floatX)
                    assert_equal(dataset.symb_inputs.ndim, inputs.ndim)
                    assert_equal(dataset.input_shape, example_shape)
                    assert_array_equal(dataset.inputs.get_value(), inputs.astype(floatX))

                    assert_equal(dataset.targets.dtype, floatX)
                    assert_equal(dataset.symb_targets.dtype, floatX)
                    assert_equal(dataset.symb_targets.ndim, targets.ndim)
                    assert_equal(dataset.target_shape, target_shape)
                    assert_array_equal(dataset.targets.get_value(), targets.astype(floatX))

    # Create dataset from nested Pyton lists.
    inputs = [[1, 2, 3]] * nb_examples
    targets = [[1, 2, 3]] * nb_examples
    dataset = Dataset(inputs, targets)
    # Data should be converted into `floatX`.
    assert_equal(dataset.inputs.dtype, floatX)
    assert_equal(dataset.symb_inputs.dtype, floatX)
    assert_equal(dataset.symb_inputs.ndim, 2)
    assert_equal(dataset.input_shape, (3,))
    assert_array_equal(dataset.inputs.get_value(), np.array(inputs, dtype=floatX))

    assert_equal(dataset.targets.dtype, floatX)
    assert_equal(dataset.symb_targets.dtype, floatX)
    assert_equal(dataset.symb_targets.ndim, 2)
    assert_equal(dataset.target_shape, (3,))
    assert_array_equal(dataset.targets.get_value(), np.array(targets, dtype=floatX))


def test_dataset_with_test_value():
    rng = np.random.RandomState(1234)

    nb_examples = 10

    theano.config.compute_test_value = 'warn'
    try:
        inputs = (rng.randn(nb_examples, 5) * 100).astype(floatX)
        targets = (rng.randn(nb_examples, 1) > 0.5).astype(floatX)
        dataset = Dataset(inputs, targets)

        input_sqr_norm = T.sum(dataset.symb_inputs**2)
        result = input_sqr_norm - dataset.symb_targets
        assert_array_equal(result.tag.test_value, np.sum(inputs**2)-targets)
    finally:
        theano.config.compute_test_value = 'off'
