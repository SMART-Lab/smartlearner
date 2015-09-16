import numpy as np

from .interfaces import Dataset
from .interfaces import Model
from .interfaces import Loss
from .interfaces import Optimizer
from .interfaces import BatchScheduler


class DummyDataset(Dataset):
    def __init__(self):
        super(DummyDataset, self).__init__(np.array([]))


class DummyModel(Model):
    def __init__(self):
        super(DummyModel, self).__init__()
        self._parameters = []

    @property
    def parameters(self):
        return self._parameters

    @property
    def updates(self):
        return {}

    def get_output(self, inputs):
        return inputs

    def save(self, path):
        pass

    def load(self, path):
        pass


class DummyLoss(Loss):
    def __init__(self):
        super(DummyLoss, self).__init__(DummyModel(), DummyDataset())

    def _compute_batch_losses(self, model_output):
        return model_output

    def _get_updates(self):
        return {}


class DummyOptimizer(Optimizer):
    def __init__(self):
        super(DummyOptimizer, self).__init__(loss=DummyLoss())

    def _get_updates(self):
        return {}

    def _get_directions(self):
        return {}


class DummyBatchScheduler(BatchScheduler):
    def __init__(self):
        super(DummyBatchScheduler, self).__init__(DummyDataset())

    @property
    def givens(self):
        return {}

    @property
    def updates(self):
        return {}

    def __iter__(self):
        return iter(range(1))
