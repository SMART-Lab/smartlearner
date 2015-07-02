from smartpy.interfaces.loss import Loss

import theano.tensor as T


class BinaryCrossEntropy(Loss):
    def _loss_function(self, model_output):
        return T.mean(T.nnet.binary_crossentropy(model_output, self.target))
        #return T.mean(T.sum(T.nnet.softplus(-self.target.T * model_output.T + (1 - self.target.T) * model_output.T), axis=0))
