from abc import ABCMeta, abstractmethod


class UpdateRule(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, directions):
        raise NotImplementedError("Subclass of 'UpdateRule' must implement 'apply(directions)'.")
