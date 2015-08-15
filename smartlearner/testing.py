import numpy as np

from smartlearner.interfaces.dataset import Dataset
from smartlearner.interfaces.model import Model
from smartlearner.interfaces.loss import Loss
from smartlearner.optimizers.optimizer import Optimizer
from smartlearner.batch_scheduler import BatchScheduler


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

    def get_model_output(self, inputs):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class DummyLoss(Loss):
    def __init__(self):
        super(DummyLoss, self).__init__(DummyModel(), DummyDataset())

    def _loss_function(self, model_output):
        pass


class DummyOptimizer(Optimizer):
    def __init__(self):
        super(DummyOptimizer, self).__init__(loss=DummyLoss())

    def _get_directions(self):
        return {}, {}


class DummyBatchScheduler(BatchScheduler):
    def __init__(self):
        super(DummyBatchScheduler, self).__init__(DummyDataset())

    @property
    def givens(self):
        return {}

    def __iter__(self):
        return iter(range(1))
