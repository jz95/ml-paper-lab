from ..trainer import Trainer
from typing import Optional, Dict

class Plugin(object):
    def __init__(self,
                 trigger_intervals: Optional[Dict[str, int]] = None,
                 first_trigger_time: Optional[Dict[str, int]] = None
                 ):
        if trigger_intervals is None:
            trigger_intervals = {}

        self.trigger_intervals: Dict[str, int] = trigger_intervals
        if self.trigger_intervals:
            if first_trigger_time is None:
                self.first_trigger_time: Dict[str, int] = {k: 0 for k in trigger_intervals}
            else:
                self.first_trigger_time = first_trigger_time
        else:
            self.first_trigger_time = {}

        self.trainer: Optional[Trainer] = None

    def register(self, trainer: Trainer):
        raise NotImplementedError
