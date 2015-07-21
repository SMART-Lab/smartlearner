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
        raise NotImplementedError("The loss function needs to be defined by subclassing the Loss class.")


class NegativeLogLikelihood(Loss):
    def _loss_function(self, model_output):
        nll = -T.log(model_output)
        indices = T.cast(self.target[:, 0], dtype="int32")  # Targets are floats.
        selected_nll = nll[T.arange(self.target.shape[0]), indices]
        return T.mean(selected_nll)


class L2Distance(Loss):
    def _loss_function(self, model_output):
        return T.mean((model_output - self.target)**2)


class L1Distance(Loss):
    def _loss_function(self, model_output):
        return T.mean(abs(model_output - self.target))