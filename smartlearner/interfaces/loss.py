from collections import OrderedDict
from theano import tensor as T


class Loss(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.consider_constant = []  # Part of the computational graph to be considered as a constant.
        self.graph_updates = OrderedDict()
        self._gradients = None

    @property
    def targets(self):
        return self.dataset.symb_targets

    def get_graph_output(self):
        output, updates = self.model.get_model_output(self.dataset.symb_inputs)
        return self._loss_function(output), updates

    @property
    def gradients(self):
        if self._gradients is None:
            cost, updates_from_model_output = self.get_graph_output()
            gparams = T.grad(cost=cost,
                             wrt=self.model.parameters,
                             consider_constant=self.consider_constant)
            self._gradients = dict(zip(self.model.parameters, gparams))
            self.graph_updates.update(updates_from_model_output)

        return self._gradients

    # def get_gradients(self):
    #     cost, updates = self.get_graph_output()
    #     gparams = T.grad(cost=cost,
    #                      wrt=self.model.parameters,
    #                      consider_constant=self.consider_constant)
    #     gradients = dict(zip(self.model.parameters, gparams))
    #     return gradients, updates

    def _loss_function(self, model_output):
        raise NotImplementedError("Subclass of 'Loss' must implement '_loss_function(model_output)'.")
