import thenao.tensor as T

from ..interfaces import Loss


class L2Distance(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_batch_losses(self, model_output):
        return T.mean((model_output - self.dataset.symb_targets)**2, axis=1)


class L1Distance(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L1Distance.

    def _compute_batch_losses(self, model_output):
        return T.mean(abs(model_output - self.dataset.symb_targets), axis=1)
