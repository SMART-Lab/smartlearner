from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class View(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.updates = OrderedDict()
        self.value = None
        self.last_update = -1
        self.last_epoch = -1

    def view(self, status):
        if self.last_update != status.current_update or self.last_epoch != status.current_epoch:
            self.value = self.update(status)
            self.last_update = status.current_update
            self.last_epoch = status.current_epoch

        return self.value

    @abstractmethod
    def update(self, status):
        raise NotImplementedError("Subclass of 'View' must implement 'update(status)'.")

    def __str__(self):
        return "{0}".format(self.value)
