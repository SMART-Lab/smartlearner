from time import time

from .interfaces import Task, RecurrentTask, View
from .views import MonitorVariable


class PrintEpochDuration(RecurrentTask):
    def __init__(self, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(PrintEpochDuration, self).__init__(**recurrent_options)

    def execute(self, status):
        print("Epoch {0} done in {1:.03f} sec.".format(status.current_epoch, time() - self.epoch_start_time))

    def pre_epoch(self, status):
        self.epoch_start_time = time()


class PrintTrainingDuration(Task):
    def init(self, status):
        self.start_time = time()

    def finished(self, status):
        print("Training done in {:.03f} sec.".format(time() - self.start_time))


class Breakpoint(RecurrentTask):
    def __init__(self, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Breakpoint, self).__init__(**recurrent_options)

    def execute(self, status):
        try:
            from ipdb import set_trace as dbg
        except ImportError:
            from pdb import set_trace as dbg
        dbg()


class Print(RecurrentTask):
    def __init__(self, msg, *views, print_epoch=True, print_update=False, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Print, self).__init__(**recurrent_options)
        self.msg = msg
        self.views = views
        self._p_epoch = print_epoch
        self._p_update = print_update

        # Add updates of the views.
        for view in self.views:
            self.updates.update(view.updates)

    def execute(self, status):
        values = [view.view(status) for view in self.views]
        if self._p_epoch:
            print(status.current_epoch, end=':')
        if self._p_update:
            print(status.current_update_in_epoch, end='  ')
        print(self.msg.format(*values))


class SaveToFile(Task):
    def __init__(self, path, msg, *views):
        super().__init__()
        self.path = path
        self.msg = msg
        self.views = views
        for view in views:
            self.updates.update(view.updates)

    def finished(self, status):
        values = [view.view(status) for view in self.views]
        with open(self.path, 'a') as f:
            f.write(self.msg.format(*values))


class Callback(RecurrentTask):
    def __init__(self, callback, **recurrent_options):
        # TODO: docstring should include **recurrent_options.
        super(Callback, self).__init__(**recurrent_options)
        self.callback = callback

    def execute(self, status):
        self.callback(self, status)


class Logger(RecurrentTask):
    def __init__(self, *views, **freqs):
        super().__init__(**freqs)
        self._views = views
        self._history = self._create_history()

        # Add updates of the views.
        for view in self._views:
            self.updates.update(view.updates)

    def __getitem__(self, item):
        return [v[item] for v in self._history]

    def __iter__(self):
        return (v for v in zip(*self._history))

    def __str__(self):
        return '\n'.join(map(str, self))

    def get_variable_history(self, var):
        if isinstance(var, int):
            idx = var
        else:
            idx = self._views.index(var)

        return self._get_variable_history(idx)

    def execute(self, status):
        self._log([v.view(status) for v in self._views])

    def clear(self):
        self._history = self._create_history()

    def _log(self, values_to_log):
        for v, h in zip(values_to_log, self._history):
            h.append(v)

    def _create_history(self):
        return [[] for _ in self._views]

    def _get_variable_history(self, index):
        return self._history[index]


class Accumulator(Logger):
    def _log(self, values_to_log):
        self._history = [v+h for v, h in zip(values_to_log, self._history)]

    def _create_history(self):
        return len(self._views) * [0]

    def __iter__(self):
        return iter(self._history)

    def __getitem__(self, item):
        return self._history[item]


class Tracker(Accumulator):
    def __init__(self, *vars, **freqs):
        super().__init__(*[MonitorVariable(v) for v in vars], **freqs)

    def _log(self, values_to_log):
        self._history = list(values_to_log)


class AveragePerEpoch(View, Accumulator):
    def __init__(self, *views):
        View.__init__(self)
        Accumulator.__init__(self, *views, each_k_update=1)

    def update(self, status):
        n = status.current_update_in_epoch
        return [v/n for v in self]

    def pre_epoch(self, status):
        self.clear()
