import theano.tensor as T

from ..interfaces import Loss


class BinaryCrossEntropy(Loss):
    def _get_updates(self):
        return {}  # There is no updates for BinaryCrossEntropy.

    def getstate(self):
        state = {"version": 1,
                 "__name__": type(self).__name__}

        return state

    def setstate(self, state):
        pass

    def _compute_losses(self, model_output):
        return T.sum(T.nnet.binary_crossentropy(model_output, self.dataset.symb_targets), axis=1)
