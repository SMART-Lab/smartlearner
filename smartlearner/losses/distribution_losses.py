import theano.tensor as T

from ..interfaces import Loss


class BinaryCrossEntropy(Loss):
    """
    Compute the crossentropy of binary random variables.

    Model output and target are each expectations of binary random
    variables; target may be exactly 0 or 1 but the model output must
    lie strictly between 0 and 1.

    References
    ----------
    https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
    http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.binary_crossentropy
    """

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
