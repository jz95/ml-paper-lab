from .plugin import Plugin
import logging
from collections import defaultdict


class Logger(Plugin):
    def __init__(self, interval=None):
        super(Logger, self).__init__([('after_step', 100), ('after_epoch', 1)])

    def register(self, trainer):
        self.trainer = trainer

    def after_step(self, *args, **kwargs):

        msg = f"[step-{self.trainer.step}]: "
        for k, values in self.trainer.stats_stepwise.items():
            i, val = values[-1]
            if i == self.trainer.step:
                msg += f"{k}:{val:.3f}\t"

        logging.warning(msg)

    def after_epoch(self,):
        pass



