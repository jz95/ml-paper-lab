class Trigger(object):
    def __init__(self, plugin, type_: str, interval: int, trigger_time: int = 0):
        assert type_ in ('before_step', 'after_step', 'before_epoch', 'after_epoch')
        self.plugin = plugin
        self.type = type_
        self.interval = interval
        self.trigger_time = trigger_time

    def __repr__(self):
        return f"Trigger(type:{self.type}, " \
               f"plugin: {self.plugin.__class__.__name__}, " \
               f"trigger_time:{self.trigger_time}, " \
               f"interval:{self.interval})"

    def __call__(self, *args, **kwargs):
        getattr(self.plugin, self.type)(*args, **kwargs)

    def next(self):
        return Trigger(plugin=self.plugin,
                       type_=self.type,
                       interval=self.interval,
                       # next trigger time
                       trigger_time=self.trigger_time + self.interval)

    def __lt__(self, other):
        assert isinstance(other, Trigger)
        assert self.type == other.type, f'trigger is incomparable between {self.type} and {other.type}'
        return self.trigger_time < other.trigger_time
