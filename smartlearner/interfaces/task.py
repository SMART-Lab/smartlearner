from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class Task(object):
    def __init__(self):
        self.updates = OrderedDict()

    def init(self, status):
        pass

    def pre_epoch(self, status):
        pass

    def pre_update(self, status):
        pass

    def post_update(self, status):
        pass

    def post_epoch(self, status):
        pass

    def finished(self, status):
        pass


class RecurrentTask(Task):
    __metaclass__ = ABCMeta

    def __init__(self, each_k_epoch=0, each_k_update=0):
        super().__init__()
        if each_k_epoch == 0 and each_k_update == 0:
            each_k_epoch = 1
        self.each_k_epoch = each_k_epoch
        self.each_k_update = each_k_update

    @abstractmethod
    def execute(self, status):
        raise NotImplementedError("Subclass of 'RecurrentTask' must implement 'execute(status)'.")

    def post_update(self, status):
        if self.each_k_update != 0 and status.current_update % self.each_k_update == 0:
            self.execute(status)

    def post_epoch(self, status):
        if self.each_k_epoch != 0 and status.current_epoch % self.each_k_epoch == 0:
            self.execute(status)
