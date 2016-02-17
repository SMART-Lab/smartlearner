import theano
import numpy as np
from os.path import join as pjoin

from smartlearner.interfaces import BatchScheduler


class MiniBatchScheduler(BatchScheduler):
    def __init__(self, dataset, batch_size):
        super(MiniBatchScheduler, self).__init__(dataset)
        self._shared_batch_size = theano.shared(np.array(0, dtype='i4'))
        self.batch_size = batch_size
        self.shared_batch_count = theano.shared(np.array(0, dtype='i4'))

        # Keep only `batch_size` examples as test values.
        self.dataset.symb_inputs.tag.test_value = self.dataset.inputs.get_value()[:batch_size]
        if self.dataset.has_targets:
            self.dataset.symb_targets.tag.test_value = self.dataset.targets.get_value()[:batch_size]

        self.shared_batch_count.tag.test_value = self.shared_batch_count.get_value()

    @property
    def batch_size(self):
        return self._shared_batch_size.get_value()

    @batch_size.setter
    def batch_size(self, value):
        self._shared_batch_size.set_value(np.array(value, dtype='i4'))
        self.nb_updates_per_epoch = int(np.ceil(len(self.dataset)/self.batch_size))

    @property
    def updates(self):
        return {}  # No updates

    @property
    def givens(self):
        start = self.shared_batch_count * self._shared_batch_size
        end = (self.shared_batch_count + 1) * self._shared_batch_size

        if self.dataset.targets is not None:
            return {self.dataset.symb_inputs: self.dataset.inputs[start:end],
                    self.dataset.symb_targets: self.dataset.targets[start:end]}
        else:
            return {self.dataset.symb_inputs: self.dataset.inputs[start:end]}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1

    def save(self, path="./"):
        state = {"version": 1,
                 "batch_size": self.batch_size,
                 "shared_batch_count": self.shared_batch_count.get_value()}

        np.savez(pjoin(path, 'mini_batch_scheduler.npz'), **state)

    def load(self, path="./"):
        state = np.load(pjoin(path, 'mini_batch_scheduler.npz'))
        self.batch_size = state["batch_size"]
        self.shared_batch_count.set_value(state["shared_batch_count"])


class FullBatchScheduler(MiniBatchScheduler):
    def __init__(self, dataset):
        super(FullBatchScheduler, self).__init__(dataset, batch_size=len(dataset))
