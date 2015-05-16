from collections import OrderedDict

import theano
from time import time
from .status import Status


class Trainer(object):
    def __init__(self, optimizer, status=None):
        super(Trainer, self).__init__()
        self.status = status if status is not None else Status()
        self.status.trainer = self
        self.optimizer = optimizer
        self.updates = OrderedDict()
        self.stopping_criteria = []
        self.tasks = []

    def train(self):
        learn = self.optimizer.build_learning_function(extra_updates=self.updates)
        #theano.printing.pydotprint(learn, '{0}_learn_{1}'.format(self.optimizer.model.__class__.__name__, theano.config.device), with_ids=True)

        # Only initialize tasks if not resuming
        if self.status.current_epoch == 0:
            self._init_tasks()

        # Learning
        while not any([stopping_criterion.check(self.status) for stopping_criterion in self.stopping_criteria]):
            self.status.current_epoch += 1

            self._pre_epoch_tasks()
            starttime = time()

            for no_update in xrange(1, self.optimizer.nb_updates_per_epoch+1):
                self.status.relative_update = no_update
                self.status.current_update += 1
                self._pre_update_tasks()
                learn(no_update-1)
                self._post_update_tasks()

            self.status.training_time += time() - starttime
            self._post_epoch_tasks()

        self._finished_tasks()
        self.status.done = True

    def append_stopping_criterion(self, criterion):
        self.stopping_criteria.append(criterion)

    def append_task(self, task):
        self.updates.update(task.updates)
        self.tasks.append(task)

    def save(self, savedir="./"):
        self.status.save(savedir)
        self.optimizer.save(savedir)
        self.model.save(savedir)

    def load(self, loaddir="./"):
        self.status.load(loaddir)
        self.optimizer.load(loaddir)
        self.model.load(loaddir)

    def _init_tasks(self):
        for task in self.tasks:
            task.init(self.status)

    def _pre_epoch_tasks(self):
        for task in self.tasks:
            task.pre_epoch(self.status)

    def _pre_update_tasks(self):
        for task in self.tasks:
            task.pre_update(self.status)

    def _post_update_tasks(self):
        for task in self.tasks:
            task.post_update(self.status)

    def _post_epoch_tasks(self):
        for task in self.tasks:
            task.post_epoch(self.status)

    def _finished_tasks(self):
        for task in self.tasks:
            task.finished(self.status)
