import heapq
import torch
from .trigger import Trigger
from typing import List, Dict, Tuple
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from collections import defaultdict

"""
CodeBase: https://zhuanlan.zhihu.com/p/414843341
"""

class Trainer(object):
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: _Loss,
                 optimizer: Optimizer,
                 data_loader: DataLoader,
                 use_cuda=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.use_cuda = use_cuda

        self.step = 0
        self.epoch = 0

        self.stats_stepwise: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.stats_epochwise: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

        self.trigger_queues: Dict[str, List[Trigger]] = {
            'before_step': [],
            'after_step': [],

            'before_epoch': [],
            'after_epoch': []
        }

    def register_plugin(self, plugin):
        plugin.register(self)

        if plugin.trigger_intervals:
            for type_, interval in plugin.trigger_intervals.items():
                t = Trigger(plugin, type_, interval)
                self.trigger_queues[type_].append(t)

    def call_plugin(self, type_, time, *args, **kwargs):
        queue = self.trigger_queues[type_]
        if len(queue) == 0:
            return

        while queue[0].trigger_time <= time:
            trigger = queue[0]
            trigger(*args, **kwargs)
            heapq.heappushpop(queue, trigger.next())

    def run(self, num_epoch=10):
        for q in self.trigger_queues.values():
            heapq.heapify(q)

        for i in range(num_epoch):
            self.call_plugin('before_epoch', i)
            self.train()
            self.call_plugin('after_epoch', i)
            self.epoch += 1

    def train(self):
        for x, y in self.data_loader:
            self.call_plugin('before_step', self.step, x, y)

            def wrap(input_data):
                if torch.is_tensor(input_data) and self.use_cuda:
                    input_data = input_data.cuda()
                return input_data

            x, y = wrap(x), wrap(y)

            plugin_data = [None, None]

            def closure():
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = out.data
                    item = (self.step, loss.data.item())
                    self.stats_stepwise['train_loss'].append(item)
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)

            self.call_plugin('after_step', self.step, x, y)
            self.step += 1
