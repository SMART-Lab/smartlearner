from ..interfaces.loss import Loss

import theano.tensor as T


class NegativeLogLikelihood(Loss):
    def _loss_function(self, model_output):
        nll = -T.log(model_output)
        indices = T.cast(self.dataset.symb_targets[:, 0], dtype="int32")  # Targets are floats.
        selected_nll = nll[T.arange(self.dataset.symb_targets.shape[0]), indices]
        return T.mean(selected_nll)


class CategoricalCrossEntropy(Loss):
    def _loss_function(self, model_output):
        return T.mean(T.nnet.categorical_crossentropy(model_output, self.dataset.symb_targets))
