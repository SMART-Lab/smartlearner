from abc import ABCMeta, abstractmethod

import numpy as np
import theano


class BatchScheduler(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def givens(self):
        raise NotImplementedError("Subclass of 'BatchScheduler' must implement 'givens'.")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("Subclass of 'BatchScheduler' must implement '__iter__()'.")


class MiniBatchScheduler(BatchScheduler):
    def __init__(self, dataset, batch_size):
        super(MiniBatchScheduler, self).__init__(dataset)
        self.batch_size = batch_size
        self.nb_updates_per_epoch = int(np.ceil(len(self.dataset)/self.batch_size))
        self.shared_batch_count = theano.shared(np.array(0, dtype='i4'))

    @property
    def givens(self):
        start = self.shared_batch_count * self.batch_size
        end = (self.shared_batch_count + 1) * self.batch_size

        if self.dataset.targets is not None:
            return {self.dataset.symb_inputs: self.dataset.inputs[start:end],
                    self.dataset.symb_targets: self.dataset.targets[start:end]}
        else:
            return {self.dataset.symb_inputs: self.dataset.inputs[start:end]}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1


class FullBatchScheduler(MiniBatchScheduler):
    def __init__(self, dataset):
        super(FullBatchScheduler, self).__init__(dataset, batch_size=len(self.dataset))