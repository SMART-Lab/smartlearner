import theano


class Dataset(object):
    def __init__(self, inputs, targets=None, name="dataset"):
        self.name = name
        self.inputs = inputs
        self.targets = targets
        self.symb_inputs = theano.tensor.matrix(name=self.name+'_inputs')
        self.symb_targets = theano.tensor.matrix(name=self.name+'_targets')

    @property
    def inputs(self):
        return self._inputs_shared

    @inputs.setter
    def inputs(self, value):
        self._inputs_shared = theano.shared(value, name=self.name + "_inputs", borrow=True)

    @property
    def targets(self):
        return self._targets_shared

    @targets.setter
    def targets(self, value):
        if value is not None:
            self._targets_shared = theano.shared(value, name=self.name + "_targets", borrow=True)
        else:
            self._targets_shared = None

    @property
    def input_size(self):
        return self.inputs.get_value().shape[-1]

    @property
    def target_size(self):
        if self.targets is None:
            return 0
        else:
            return self.targets.get_value().shape[-1]

    def __len__(self):
        return len(self.inputs.get_value())
