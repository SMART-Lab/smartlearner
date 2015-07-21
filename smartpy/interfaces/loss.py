from collections import OrderedDict

from theano import tensor as T


class Loss(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.target = dataset.symb_targets
        self.consider_constant = []  # Part of the computational graph to be considered as a constant.

    def get_graph_output(self):
        return self._loss_function(self.model.get_model_output(self.dataset.symb_inputs))

    def get_gradients(self):
        gparams = T.grad(cost=self.get_graph_output(),
                         wrt=self.model.parameters,
                         consider_constant=self.consider_constant)
        gradients = dict(zip(self.model.parameters, gparams))
        return gradients, OrderedDict()

    def _loss_function(self, model_output):
        raise NotImplementedError("Subclass of 'Loss' must implement '_loss_function(model_output)'.")
