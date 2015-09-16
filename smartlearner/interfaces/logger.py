from .task import RecurrentTask

class Logger(RecurrentTask):
    def __init__(self, *vars, **freqs):
        super().__init__(**freqs)
        self._symb_variables = vars
        self._shared_variables = [self.track_variable(v) for v in vars]
        self._names = [x.name for x in self._shared_variables]
        self._history = self._create_history()

    def __getitem__(self, item):
        return [v[item] for v in self._history]

    def __iter__(self):
        return (v for v in zip(*self._history))

    def get_variable_history(self, var):
        if isinstance(var, str):
            idx = self._names.index(var)
        elif isinstance(var, int):
            idx = var
        else:
            idx = self._symb_variables.index(var)

        return self._get_variable_history(idx)

    def execute(self, status):
        self._log([v.get_value(borrow=False) for v in self._shared_variables])

    def _log(self, values_to_log):
        for v, h in zip(values_to_log, self._history):
            h.append(v)

    def _create_history(self):
        return len(self._shared_variables) * [[]]

    def _get_variable_history(self, index):
        return self._history[index]
