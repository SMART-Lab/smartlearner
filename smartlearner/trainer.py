from collections import OrderedDict

import theano
from time import time
from .status import Status
from smartlearner.tasks.stopping_criteria import TrainingExit


class Trainer(object):
    def __init__(self, optimizer, batch_scheduler, status=None):
        self.status = status if status is not None else Status(self)
        self._optimizer = optimizer
        self._batch_scheduler = batch_scheduler
        self._updates = OrderedDict()
        self._tasks = []

    def train(self):
        self._pre_learning()
        self._learning()
        self._post_learning()

    def append_task(self, task):
        self._updates.update(task.updates)
        self._tasks.append(task)

    def _build_theano_graph(self):
        updates = self._optimizer.gather_updates()
        updates.update(self._updates)
        self._learn = theano.function([],
                       updates=updates,
                       givens=self._batch_scheduler.givens,
                       name="learn")

    def _pre_learning(self):
        self._build_theano_graph()
        #theano.printing.pydotprint(learn, '{0}_learn_{1}'.format(self.optimizer.model.__class__.__name__, theano.config.device), with_ids=True)

        # Only initialize tasks if not resuming
        if self.status.current_update == 0:
            self._init_tasks()

    def _learning(self):
        # Learning
        try:
            while True:  # Runs until a TrainingExit exception is raised (usually inside a Task)
                self.status.increment_epoch()

                self._pre_epoch_tasks()

                for _ in self._batch_scheduler:
                    self.status.increment_update()
                    self._pre_update_tasks()
                    self._learn()
                    self._post_update_tasks()

                self._post_epoch_tasks()
        except TrainingExit:
            pass

    def _post_learning(self):
        self._finished_tasks()
        self.status.done = True

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
