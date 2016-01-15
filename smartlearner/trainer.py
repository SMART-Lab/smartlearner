from collections import OrderedDict

from os.path import join as pjoin
from . import utils

import theano
from .status import Status
from .stopping_criteria import TrainingExit


class Trainer(object):
    def __init__(self, optimizer, batch_scheduler, status=None):
        self.status = status if status is not None else Status(self)
        self._optimizer = optimizer
        self._batch_scheduler = batch_scheduler

        # Gather updates from the optimizer and the batch scheduler.
        self._graph_updates = OrderedDict()
        self._graph_updates.update(self._optimizer.updates)
        self._graph_updates.update(self._batch_scheduler.updates)

        # Gather tasks from the optimizer and the batch scheduler.
        self._tasks = []
        self._tasks.extend(self._optimizer.tasks)
        self._tasks.extend(self._batch_scheduler.tasks)

        self._learn = None

    def train(self):
        self._pre_learning()
        self._learning()
        self._post_learning()

    def append_task(self, *tasks):
        self._tasks.extend(tasks)

    def build_theano_graph(self):
        # Get updates from tasks.
        for task in self._tasks:
            self._graph_updates.update(task.updates)

        self._learn = theano.function([],
                                      updates=self._graph_updates,
                                      givens=self._batch_scheduler.givens,
                                      name="learn")
        #theano.printing.pydotprint(self._learn, '{0}_learn_{1}'.format(self._optimizer.loss.model.__class__.__name__, theano.config.device), with_ids=True)

    def save(self, path):
        savedir = utils.create_folder(pjoin(path, "training"))
        self.status.save(savedir)
        self._optimizer.save(savedir)
        self._batch_scheduler.save(savedir)

        tasks_dir = utils.create_folder(pjoin(savedir, 'tasks'))
        for task in self._tasks:
            task.save(tasks_dir)

    def load(self, path):
        loaddir = pjoin(path, "training")
        self.status.load(loaddir)
        self._optimizer.load(loaddir)
        self._batch_scheduler.load(loaddir)

        tasks_dir = pjoin(loaddir, 'tasks')
        for task in self._tasks:
            task.load(tasks_dir)

    def _pre_learning(self):
        if self._learn is None:
            self.build_theano_graph()

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
