import theano.tensor as T


class Optimizer(object):
    def __init__(self, model, loss_fct, dataset, update_rules=None, param_modifiers=None):
        self.model = model
        self.loss = loss_fct
        self.dataset = dataset

        self.update_rules = update_rules if update_rules is not None else []
        self.param_modifiers = param_modifiers if param_modifiers is not None else []

        self.data = [dataset.inputs_shared, dataset.targets_shared]
        self.inputs = [T.matrix('input' + str(i)) for i in range(len(self.data))]
        self.loss = loss_fct(*self.inputs)

    def append_update_rule(self, update_rule):
        self.update_rules.append(update_rule)

    def save(self, savedir="./"):
        for update_rule in self.update_rules:
            update_rule.save(savedir, update_rule.__class__.__name__)

    def load(self, loaddir="./"):
        for update_rule in self.update_rules:
            update_rule.load(loaddir, update_rule.__class__.__name__)
