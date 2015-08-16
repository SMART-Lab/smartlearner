from ..interfaces.loss import Loss

import theano.tensor as T


class BinaryCrossEntropy(Loss):
    def _loss_function(self, model_output):
        return T.mean(T.nnet.binary_crossentropy(model_output, self.dataset.symb_targets))
