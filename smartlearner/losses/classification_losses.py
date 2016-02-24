import theano.tensor as T

from ..interfaces import Loss


class NegativeLogLikelihood(Loss):
    def _get_updates(self):
        return {}  # There is no updates for NegativeLogLikelihood.

    def getstate(self):
        state = {"version": 1,
                 "__name__": type(self).__name__}

        return state

    def setstate(self, state):
        pass

    def _compute_losses(self, model_output):
        nll = -T.log(model_output)
        indices = T.cast(self.dataset.symb_targets[:, 0], dtype="int32")  # Targets are floats.
        selected_nll = nll[T.arange(self.dataset.symb_targets.shape[0]), indices]
        return selected_nll


class CategoricalCrossEntropy(Loss):
    def _get_updates(self):
        return {}  # There is no updates for CategoricalCrossEntropy.

    def getstate(self):
        state = {"version": 1,
                 "__name__": type(self).__name__}

        return state

    def setstate(self, state):
        pass

    def _compute_losses(self, model_output):
        return T.nnet.categorical_crossentropy(model_output, self.dataset.symb_targets)


class ClassificationError(Loss):
    """
    Notes:
    ------
    Each target should be the ID of the class.
    """
    def _get_updates(self):
        return {}  # There is no updates for ClassificationError.

    def getstate(self):
        state = {"version": 1,
                 "__name__": type(self).__name__}

        return state

    def setstate(self, state):
        pass

    def _compute_losses(self, model_output):
        predictions = T.argmax(model_output, axis=1, keepdims=True)
        return T.neq(predictions, self.dataset.symb_targets)[:, 0]
