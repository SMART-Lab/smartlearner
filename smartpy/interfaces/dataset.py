import theano


class Dataset(object):
    def __init__(self, inputs, targets, name="dataset"):
        self.name = name
        self.inputs = inputs
        self.targets = targets

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value
        self._inputs_shared = theano.shared(self.inputs, name=self.name + "_inputs", borrow=True)

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value
        self._targets_shared = theano.shared(self.targets, name=self.name + "_targets", borrow=True)

    @property
    def inputs_shared(self):
        return self._inputs_shared

    @property
    def targets_shared(self):
        return self._targets_shared

    @property
    def input_size(self):
        return len(self._inputs[0])

    @property
    def target_size(self):
        return len(self._targets[0])

    def __len__(self):
        return len(self._inputs)
