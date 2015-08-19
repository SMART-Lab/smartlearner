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


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def load_dict_from_json_file(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())


def split_dataset(dataset, proportions):
    from .interfaces.dataset import Dataset
    indices = np.cumsum(np.ceil(np.array(proportions) / np.sum(proportions) * len(dataset)))
    indices[-1] = len(dataset)
    dsets = []

    for (s, f) in zip([0] + list(indices), indices):
        s, f = int(s), int(f)

        covars = dataset.inputs.get_value()[s:f]
        targets = dataset.targets.get_value()[s:f] if dataset.targets is not None else None

        dset = Dataset(covars, targets, dataset.name + '_' + str(s) + 'to' + str(f))
        dset.symb_inputs = dataset.symb_inputs
        dset.symb_targets = dataset.symb_targets

        dsets.append(dset)

    return dsets


def kfold(dataset, k):
    return split_dataset(dataset, np.ones((k,)))


def sequential_kfold(dataset, k):
    if k < 2:
        raise ValueError("Parameter k has to be greater than 1.")
    folds = []

    for i in range(1, k + 1):
        if k - i:
            proportions = [i, 1, 1]
        else:
            proportions = [i, 1, 1, k - i]  # trainset, validset, testset, leftover

        folds.append(split_dataset(dataset, proportions)[:3])

    return folds
