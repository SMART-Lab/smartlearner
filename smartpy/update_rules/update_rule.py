from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class UpdateRule(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.params = OrderedDict()

    @abstractmethod
    def apply(self, gradients):
        raise NotImplementedError("Subclass of 'UpdateRule' must implement 'apply(gradients)'.")
