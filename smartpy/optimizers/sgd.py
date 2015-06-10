import numpy as np
from . import Optimizer


class SGD(Optimizer):
    def __init__(self, model, loss_fct, dataset, batch_size=1):
        super(SGD, self).__init__(model, loss_fct, dataset)
        self.batch_size = batch_size
        self.nb_updates_per_epoch = int(np.ceil(len(dataset) / self.batch_size))

    def _get_directions(self):
        self.gradients, updates_from_get_gradients = self.model.get_gradients(self.loss)

        # Take the opposite of the gradient.
        for param, gparam in self.gradients.items():
            self.gradients[param] = -gparam

        return self.gradients, updates_from_get_gradients
