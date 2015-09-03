import theano.tensor as T

from ..interfaces import Loss


class BinaryCrossEntropy(Loss):
    def _get_updates(self):
        return {}  # There is no updates for BinaryCrossEntropy.

    def _compute_batch_losses(self, model_output):
        return T.nnet.binary_crossentropy(model_output, self.dataset.symb_targets)
