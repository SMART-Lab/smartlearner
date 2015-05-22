# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from collections import OrderedDict
from time import time

import theano
import theano.tensor as T
from abc import ABCMeta, abstractmethod


class StoppingCriterion(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def check(self, status):
        raise NotImplementedError("Subclass of 'StoppingCriterion' must implement 'check(status)'.")


class Task(object):
    def __init__(self):
        self.updates = OrderedDict()

    def track_variable(self, var, name=""):
        var_shared = theano.shared(np.array(0, dtype=var.dtype, ndmin=var.ndim), name=name)
        self.updates[var_shared] = var
        return var_shared

    def init(self, status):
        pass

    def pre_epoch(self, status):
        pass

    def pre_update(self, status):
        pass

    def post_update(self, status):
        pass

    def post_epoch(self, status):
        pass

    def finished(self, status):
        pass


class View(object):
    # TODO: Should 'View' subclass 'Task'?
    __metaclass__ = ABCMeta

    def __init__(self):
        self.updates = OrderedDict()
        self.value = None
        self.last_update = -1

    def view(self, status):
        if self.last_update != status.current_update:
            self.value = self.update(status)
            self.last_update = status.current_update

        return self.value

    @abstractmethod
    def update(self, status):
        raise NotImplementedError("Subclass of 'View' must implement 'update(status)'.")

    def __str__(self):
        return "{0}".format(self.value)


class RecurrentTask(Task):
    __metaclass__ = ABCMeta

    def __init__(self, each_epoch=1, each_update=0):
        super(RecurrentTask, self).__init__()
        self.each_epoch = each_epoch
        self.each_update = each_update

    @abstractmethod
    def execute(self, status):
        raise NotImplementedError("Subclass of 'RecurrentTask' must implement 'execute(status)'.")

    def post_update(self, status):
        if self.each_update != 0 and status.current_update % self.each_update == 0:
            self.execute(status)

    def post_epoch(self, status):
        if self.each_epoch != 0 and status.current_epoch % self.each_epoch == 0:
            self.execute(status)


class ItemGetter(View):
    def __init__(self, view, attribute):
        """ Retrieves `attribute` from a `view` which return an indexable object. """
        super(ItemGetter, self).__init__()
        self.view_obj = view
        self.attribute = attribute

    def update(self, status):
        infos = self.view_obj.view(status)
        return infos[self.attribute]


class PrintVariable(RecurrentTask):
    def __init__(self, msg, *variables, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(PrintVariable, self).__init__(**recurrent_options)
        self.msg = msg
        self.variables = [self.track_variable(v) for v in variables]

    def execute(self, status):
        print(self.msg.format(*[v.get_value() for v in self.variables]))


class PrintEpochDuration(RecurrentTask):
    def __init__(self, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(PrintEpochDuration, self).__init__(**recurrent_options)

    def execute(self, status):
        print("Epoch {0} done in {1:.03f} sec.".format(status.current_epoch, time() - self.epoch_start_time))

    def pre_epoch(self, status):
        self.epoch_start_time = time()


class PrintTrainingDuration(Task):
    def finished(self, status):
        print("Training done in {:.03f} sec.".format(status.training_time))


class Breakpoint(RecurrentTask):
    def __init__(self, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Breakpoint, self).__init__(**recurrent_options)

    def execute(self, status):
        from ipdb import set_trace as dbg
        dbg()


class ClassificationError(View):
    def __init__(self, predict_fct, dataset):
        super(ClassificationError, self).__init__()

        batch_size = 1024  # Internal buffer
        self.nb_batches = int(np.ceil(len(dataset) / batch_size))

        input = T.matrix('input')
        target = T.matrix('target')
        classification_errors = T.neq(predict_fct(input), target)

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs_shared[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets_shared[no_batch * batch_size:(no_batch + 1) * batch_size]}
        self.compute_classification_error = theano.function([no_batch],
                                                            classification_errors,
                                                            givens=givens,
                                                            name="compute_classification_error")

    def update(self, status):
        classif_errors = []
        for i in range(self.nb_batches):
            classif_errors.append(self.compute_classification_error(i))

        classif_errors = np.concatenate(classif_errors)
        return classif_errors.mean(), classif_errors.std(ddof=1) / np.sqrt(len(classif_errors))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)


class Print(RecurrentTask):
    def __init__(self, msg, *views, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Print, self).__init__(**recurrent_options)
        self.msg = msg
        self.views = views

        # Add updates of the views.
        for view in self.views:
            self.updates.update(view.updates)

    def execute(self, status):
        values = [view.view(status) for view in self.views]
        print(self.msg.format(*values))


class MaxEpochStopping(StoppingCriterion):
    def __init__(self, nb_epochs_max):
        self.nb_epochs_max = nb_epochs_max

    def check(self, status):
        return status.current_epoch >= self.nb_epochs_max
