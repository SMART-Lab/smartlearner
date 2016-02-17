from os.path import join as pjoin

from .utils import save_dict_to_json_file, load_dict_from_json_file


class Status(object):
    def __init__(self, trainer=None, starting_epoch=0, starting_update=0):
        self.current_epoch = starting_epoch
        self.current_update = starting_update
        self.current_update_in_epoch = 1

        self.trainer = trainer
        self.training_time = 0
        self.done = False

        self.extra = {}

    def increment_update(self):
        self.current_update += 1
        self.current_update_in_epoch += 1

    def increment_epoch(self):
        self.current_epoch += 1
        self.current_update_in_epoch = 0

    def __repr__(self):
        return ('Status object with state :\n' +\
                '  current_epoch = {!r}\n' +\
                '  current_update = {!r}\n' +\
                '  current_update_in_epoch  = {!r}\n' +\
                '  trainer = {!r}\n' +\
                '  done = {!r}\n' +\
                '  extra = {!r}\n').format(self.current_epoch, self.current_update, self.current_update_in_epoch,
                                           self.trainer, self.training_time, self.done, self.extra)

    def save(self, savedir="./"):
        state = {"version": 1,
                 "current_epoch": self.current_epoch,
                 "current_update": self.current_update,
                 "current_update_in_epoch": self.current_update_in_epoch,
                 "training_time": self.training_time,
                 "done": self.done,
                 "extra": self.extra,
                 }
        save_dict_to_json_file(pjoin(savedir, 'status.json'), state)

    def load(self, loaddir="./"):
        state = load_dict_from_json_file(pjoin(loaddir, 'status.json'))
        self.current_epoch = state["current_epoch"]
        self.current_update = state["current_update"]
        self.current_update_in_epoch = state["current_update_in_epoch"]
        self.training_time = state["training_time"]
        self.done = state["done"]
        self.extra = state["extra"]
