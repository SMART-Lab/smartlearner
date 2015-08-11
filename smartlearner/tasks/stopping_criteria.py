from smartlearner.interfaces.task import Task


class TrainingExit(Exception):
    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "Training exited with \n" + repr(self.status)


class MaxEpochStopping(Task):
    def __init__(self, nb_epochs_max):
        super().__init__()
        self.nb_epochs_max = nb_epochs_max

    def post_epoch(self, status):
        if status.current_epoch >= self.nb_epochs_max:
            raise TrainingExit(status)
