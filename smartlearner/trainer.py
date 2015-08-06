from collections import OrderedDict

import theano
from time import time
from .status import Status


class Trainer(object):
    def __init__(self, optimizer, batch_scheduler, stopping_criterion, status=None):
        self.status = status if status is not None else Status()
        self.status.trainer = self
        self._optimizer = optimizer
        self._batch_scheduler = batch_scheduler
        self._updates = OrderedDict()
        self._stopping_criteria = [stopping_criterion]
        self._tasks = []

    def build_theano_graph(self):
        updates = self._optimizer.gather_updates()
        updates.update(self._updates)
        self.learn = theano.function([],
                       updates=updates,
                       givens=self._batch_scheduler.givens,
                       name="learn")

    def train(self):
        self.build_theano_graph()
        #theano.printing.pydotprint(learn, '{0}_learn_{1}'.format(self.optimizer.model.__class__.__name__, theano.config.device), with_ids=True)

        # Only initialize tasks if not resuming
        if self.status.current_epoch == 0:
            self._init_tasks()

        # Learning
        while not any([stopping_criterion.check(self.status) for stopping_criterion in self._stopping_criteria]):
            self.status.current_epoch += 1

            self._pre_epoch_tasks()
            starttime = time()

            for batch_count in self._batch_scheduler:
                self.status.current_update_in_epoch = batch_count
                self.status.current_update += 1
                self._pre_update_tasks()
                self.learn()
                self._post_update_tasks()

            self.status.training_time += time() - starttime
            self._post_epoch_tasks()

        self._finished_tasks()
        self.status.done = True

    def append_stopping_criterion(self, criterion):
        self._stopping_criteria.append(criterion)

    def append_task(self, task):
        self._updates.update(task.updates)
        self._tasks.append(task)

    def _init_tasks(self):
        for task in self._tasks:
            task.init(self.status)

    def _pre_epoch_tasks(self):
        for task in self._tasks:
            task.pre_epoch(self.status)

    def _pre_update_tasks(self):
        for task in self._tasks:
            task.pre_update(self.status)

    def _post_update_tasks(self):
        for task in self._tasks:
            task.post_update(self.status)

    def _post_epoch_tasks(self):
        for task in self._tasks:
            task.post_epoch(self.status)

    def _finished_tasks(self):
        for task in self._tasks:
            task.finished(self.status)
