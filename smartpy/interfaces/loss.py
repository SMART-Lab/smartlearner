from collections import OrderedDict

from theano import tensor as T


class Loss(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.target = dataset.symb_targets

    def get_graph_output(self):
        return self._loss_function(self.model.get_model_output(self.dataset.symb_inputs))

    def get_gradients(self):
        gparams = T.grad(self.get_graph_output(), self.model.parameters)
        gradients = dict(zip(self.model.parameters, gparams))
        return gradients, OrderedDict()

    def _loss_function(self, model_output):
        raise NotImplementedError("Subclass of 'Loss' must implement '_loss_function(model_output)'.")
