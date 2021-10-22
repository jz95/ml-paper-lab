from ..trainer import Trainer
from typing import Optional, List, Tuple

class Plugin(object):
    def __init__(self, interval: Optional[List[Tuple[str, int]]] = None):
        if interval is None:
            interval = []

        self.trigger_interval = interval
        self.trainer: Optional[Trainer] = None

    def register(self, trainer: Trainer):
        raise NotImplementedError
