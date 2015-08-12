import numpy as np

from smartlearner.interfaces.task import Task


class TrainingExit(Exception):
    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "Training exited with \n" + repr(self.status)


class MaxEpochStopping(Task):
    def __init__(self, nb_epochs_max):
        super().__init__()
        self.nb_epochs_max = nb_epochs_max

    def post_epoch(self, status):
        if status.current_epoch >= self.nb_epochs_max:
            raise TrainingExit(status)


class EarlyStopping(Task):
    def __init__(self, cost, lookahead, callback=None, eps=0.):
        super(EarlyStopping, self).__init__()

        self.cost = cost
        self.best_epoch = 0
        self.best_cost = np.inf

        self.lookahead = lookahead
        self.callback = callback
        self.eps = eps

    def init(self, status):
        self.best_cost = self.cost.view(status)

    def post_epoch(self, status):
        cost = self.cost.view(status)
        if cost + self.eps < self.best_cost:
            self.best_epoch = status.current_epoch
            self.best_cost = float(cost)

            if self.callback is not None:
                self.callback(self, status)

        if status.current_epoch - self.best_epoch >= self.lookahead:
            raise TrainingExit(status)
