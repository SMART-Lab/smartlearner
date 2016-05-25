import os
import json

import theano
import theano.tensor as T
import numpy as np


def sharedX(value, name=None, borrow=True, keep_on_cpu=False):
    """ Transform value into a shared variable of type floatX """
    if keep_on_cpu:
        return T._shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)

    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}

        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        return np.array(dct['__ndarray__'], dtype=theano.config.floatX)

    return dct


def save_dict_to_json_file(path, dictionary):
    """ Saves a dict in a json formatted file. """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': '), cls=NumpyEncoder))


def load_dict_from_json_file(path):
    """ Loads a dict from a json formatted file. """
    with open(path, "r") as json_file:
        return json.loads(json_file.read(), object_hook=json_numpy_obj_hook)


def create_folder(path):
    """ Creates a leaf directory and all intermediate ones (thread safe). """
    try:
        os.makedirs(path)
    except:
        pass

    return path


def split_dataset(dataset, proportions):
    """ Split a dataset into many smaller datasets.

    Parameters
    ----------
    dataset : Dataset
    proportions : [numbers]
        A list of numbers giving the proportions of the dataset to be included in each
        sub-datasets. The proportions are computed according to the sum all the numbers.

    Returns
    -------
    [Datasets]
        A list of datasets which respect the given `proportions`.

    """
    from .interfaces.dataset import Dataset
    indices = np.cumsum(np.ceil(np.array(proportions) / np.sum(proportions) * len(dataset)))
    indices[-1] = len(dataset)
    dsets = []

    for s, f in zip([0] + list(indices), indices):
        s, f = int(s), int(f)

        covars = dataset.inputs.get_value()[s:f]
        targets = dataset.targets.get_value()[s:f] if dataset.has_targets else None

        dset = Dataset(covars, targets, dataset.name + '_' + str(s) + 'to' + str(f))
        dset.symb_inputs = dataset.symb_inputs
        dset.symb_targets = dataset.symb_targets

        dsets.append(dset)

    return dsets


def kfold(dataset, k):
    return split_dataset(dataset, np.ones((k,)))


def growing_sequential_kfold(dataset, k):
    """ Builds `k` growing validations triplets.

    Parameters
    ----------
    dataset : Dataset
    k : int
        The number of triplets required as validations sets.

    Returns
    -------
    [[Dataset]]
        Returns a list of list of three datasets (training set, validation set, test set) where the length of the
        training set is growing for each further triplets.
    """
    if k < 2:
        raise ValueError("Parameter k has to be greater than 1.")
    folds = []

    for i in range(1, k + 1):
        proportions = [i, 1, 1, k - i]  # trainset, validset, testset, leftover
        folds.append(split_dataset(dataset, proportions)[:3])

    return folds
