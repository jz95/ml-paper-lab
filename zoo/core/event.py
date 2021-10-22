class Event(object):
    def __init__(self, plugin, type_: str, interval: int, trigger_time: int = 0):
        self.plugin = plugin
        self.type = type_
        self.interval = interval
        self.trigger_time = trigger_time

    def trigger(self, *args, **kwargs):
        getattr(self.plugin, self.type)(*args, **kwargs)

    def next(self):
        return Event(plugin=self.plugin,
                     type_=self.type,
                     interval=self.interval,
                     trigger_time=self.trigger_time + self.interval)

    def __lt__(self, other):
        return self.trigger_time < other.trigger_time