import theano.tensor as T

from ..interfaces import Loss


class L2Distance(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_loss(self, model_output):
        return T.mean((model_output - self.dataset.symb_targets)**2)


class L1Distance(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L1Distance.

    def _compute_loss(self, model_output):
        return T.mean(abs(model_output - self.dataset.symb_targets))
