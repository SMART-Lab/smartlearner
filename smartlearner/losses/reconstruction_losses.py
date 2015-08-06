from ..interfaces.loss import Loss

import theano.tensor as T


class L2Distance(Loss):
    def _loss_function(self, model_output):
        return T.mean((model_output - self.target)**2)


class L1Distance(Loss):
    def _loss_function(self, model_output):
        return T.mean(abs(model_output - self.target))
