from .plugin import Plugin

class Monitor(Plugin):
    def __init__(self,
                 running_average=True,
                 epoch_average=True,
                 smoothing=0.7,
                 precision=4):
        super(Monitor, self).__init__({'after_step': 1})
        self.smoothing = smoothing
        self.running_average = running_average
        self.epoch_average = epoch_average

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields

        if self.epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields

        if self.epoch_average:
            stats['epoch_stats'] = (0, 0)


    def after_step(self, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        # 通过_get_value 方法拿到每个插件的值,放入到stats中
        stats['last'] = self._get_value(*args)

        if self.epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                                   stats['last'] * (1 - self.smoothing)


    def after_epoch(self, idx):
        # 每个epoch 进行的操作
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.epoch_average:
            # 如果需要计算每轮epoch 的精度等,需要 总数/轮数
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
        stats['epoch_stats'] = (0, 0)


class LossMonitor(Monitor):
    stat_name = 'loss'

    def _get_value(self, loss):
        return loss.item()