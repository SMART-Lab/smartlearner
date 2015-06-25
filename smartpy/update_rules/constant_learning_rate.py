from . import DecreasingLearningRate


class ConstantLearningRate(DecreasingLearningRate):
    def __init__(self, lr):
        """
        Implements a constant learning rate update rule.

        Parameters
        ----------
        lr: float
            learning rate
        """
        super(ConstantLearningRate, self).__init__(lr=lr, dc=0.)
