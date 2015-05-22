
class Status(object):
    def __init__(self, starting_epoch=0, starting_update=0):
        self.current_epoch = starting_epoch
        self.current_update = starting_update
        self.current_update_in_epoch = 1
        self.training_time = 0
        self.done = False
        self.extra = {}
