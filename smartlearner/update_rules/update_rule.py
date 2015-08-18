from abc import ABCMeta, abstractmethod


class UpdateRule(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_updates(self):
        raise NotImplementedError("Subclass of 'UpdateRule' must implement '_get_updates()'.")

    @abstractmethod
    def apply(self, directions):
        raise NotImplementedError("Subclass of 'UpdateRule' must implement 'apply(directions)'.")

    @property
    def updates(self):
        return self._get_updates()
