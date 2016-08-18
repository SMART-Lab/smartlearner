import numpy as np
import theano
from theano import tensor as T

from collections import OrderedDict

from .interfaces import View


class ItemGetter(View):
    def __init__(self, view, attribute):
        """ Retrieves `attribute` from a `view` which return an indexable object. """
        super().__init__()
        self.view_obj = view
        self.attribute = attribute

    def update(self, status):
        infos = self.view_obj.view(status)
        return infos[self.attribute]

    def __getitem__(self, idx):
        return ItemGetter(self, attribute=idx)


class LossView(View):
    def __init__(self, loss, batch_scheduler):
        super().__init__()

        self.batch_scheduler = batch_scheduler

        losses = loss.losses

        # Gather updates from the optimizer and the batch scheduler.
        graph_updates = OrderedDict()
        graph_updates.update(loss.updates)
        graph_updates.update(batch_scheduler.updates)

        self.compute_loss = theano.function([],
                                            losses,
                                            updates=graph_updates,
                                            givens=batch_scheduler.givens,
                                            name="compute_loss")

    def update(self, status):
        losses = []
        for i in self.batch_scheduler:
            losses.append(self.compute_loss())

        losses = np.concatenate(losses)
        return (losses,
                float(losses.mean()), float(losses.std(ddof=1) / np.sqrt(len(losses))),
                float(losses.sum()))

    @property
    def losses(self):
        return ItemGetter(self, attribute=0)

    @property
    def mean(self):
        return ItemGetter(self, attribute=1)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=2)

    @property
    def sum(self):
        return ItemGetter(self, attribute=3)


class MonitorVariable(View):
    def __init__(self, var):
        super().__init__()
        self.var = self.track_variable(var)

    def update(self, status):
        return self.var.get_value(borrow=False)

    def track_variable(self, var, name=None):
        name = name if name is not None else var.name
        name = name if name is not None else var.auto_name
        var_shared = theano.shared(np.array(0, dtype=var.dtype, ndmin=var.ndim), name=name)
        self.updates[var_shared] = var
        return var_shared


class CallbackView(View):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def update(self, status):
        return self.callback(self, status)


class ClassificationError(View):
    def __init__(self, predict_fct, dataset, batch_size=None):
        super().__init__()
        self._batch_size = batch_size

        input = dataset.symb_inputs
        target = dataset.symb_targets
        classification_errors = T.neq(predict_fct(input), target)

        if batch_size is None:
            batch_size = len(dataset)

        self.nb_batches = int(np.ceil(len(dataset) / batch_size))
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}

        self.compute_classification_error = theano.function([no_batch],
                                                            classification_errors,
                                                            givens=givens,
                                                            name="compute_classification_error" + dataset.name)

    def update(self, status):
        classif_errors = np.concatenate([self.compute_classification_error(i) for i in range(self.nb_batches)])
        return float(classif_errors.mean()), float(classif_errors.std(ddof=1) / np.sqrt(len(classif_errors)))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)


class RegressionError(View):
    def __init__(self, predict_fct, dataset, batch_size=None):
        super().__init__()
        self._batch_size = batch_size

        input = dataset.symb_inputs
        target = dataset.symb_targets

        regression_errors = T.sqr(predict_fct(input) - target).mean(axis=1)

        if batch_size is None:
            batch_size = len(dataset)

        self.nb_batches = int(np.ceil(len(dataset) / batch_size))
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}

        self.compute_regression_error = theano.function([no_batch],
                                                        regression_errors,
                                                        givens=givens,
                                                        name="compute_reconstruction_error_" + dataset.name)

    def update(self, status):
        regression_errors = np.concatenate([self.compute_regression_error(i) for i in range(self.nb_batches)])
        return float(regression_errors.mean()), float(regression_errors.std(ddof=1) / np.sqrt(len(regression_errors)))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)
