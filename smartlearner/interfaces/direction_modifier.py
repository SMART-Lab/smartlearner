from abc import ABCMeta, abstractmethod


class DirectionModifier(object):
    __metaclass__ = ABCMeta

    @property
    def tasks(self):
        return []

    @property
    def updates(self):
        return self._get_updates()

    @abstractmethod
    def _get_updates(self):
        raise NotImplementedError("Subclass of 'DirectionModifier' must implement '_get_updates()'.")

    @abstractmethod
    def apply(self, directions):
        raise NotImplementedError("Subclass of 'DirectionModifier' must implement 'apply(directions)'.")

    def save(self, path):
        pass

    def load(self, path):
        pass


class ParamModifier(object):
    __metaclass__ = ABCMeta

    @property
    def tasks(self):
        return []

    @property
    def updates(self):
        return self._get_updates()

    @abstractmethod
    def _get_updates(self):
        raise NotImplementedError("Subclass of 'ParamModifier' must implement '_get_updates()'.")

    @abstractmethod
    def apply(self, params):
        raise NotImplementedError("Subclass of 'ParamModifier' must implement 'apply(params)'.")

    def save(self, path):
        pass

    def load(self, path):
        pass
