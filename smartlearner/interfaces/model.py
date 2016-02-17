from abc import ABCMeta, abstractmethod, abstractproperty


class Model(object):
    __metaclass__ = ABCMeta

    @property
    def tasks(self):
        return []

    @abstractmethod
    def get_output(self, inputs):
        raise NotImplementedError("Subclass of 'Model' must define a model output (a theano graph)")

    @abstractproperty
    def updates(self):
        raise NotImplementedError("Subclass of 'Model' must implement property 'updates'.")

    @abstractproperty
    def parameters(self):
        raise NotImplementedError("Subclass of 'Model' must implement property 'parameters'.")

    @abstractmethod
    def save(self, path):
        """ Saves model information to disk. """
        raise NotImplementedError("Subclass of 'Model' must implement 'save(path)'.")

    @abstractmethod
    def load(self, path):
        """ Loads model information from disk. """
        raise NotImplementedError("Subclass of 'Model' must implement 'load(path)'.")
