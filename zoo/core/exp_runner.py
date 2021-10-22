from zoo.utils import isnotebook
import torch

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ExpRunner(object):
    def __init__(self, repeat_num, seed=0):
        self.seed = seed
        self.repeat_num = repeat_num
        self.exp_results = []

    def _run(self):
        raise NotImplementedError

    def run(self):
        torch.random.manual_seed(self.seed)
        for _ in tqdm(range(self.repeat_num)):
            data = self._run()
            self.exp_results.append(data)
