import torch
import logging


from typing import Dict
from paperlab.utils import isnotebook
from paperlab.core.utils import MultiProcessManager
from collections import Sequence


if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

    
class ExpRunner(object):
    """
    repeat experiments under the given exp config and random seeds
    """
    def __init__(self,
                 exp_func,
                 exp_config: Dict,
                 repeat_num=5,
                 seeds=None,
                 ):
        """
        :param exp_func: the function runs the experiment logic and returns the result
        :param exp_config: the arguments accepted by `exp_func` , stored in key-value format
        :param repeat_num:
        :param seeds:
        """
        if seeds is None:
            self.seeds = list(range(repeat_num))
        elif isinstance(seeds, Sequence):
            self.seeds = list(map(int, seeds))
            assert len(self.seeds) == repeat_num, f'seeds num: {len(self.seeds)} neq repeat_num: {repeat_num}'
        else:
            raise TypeError(f"unsupported seeds type: {type(seeds)}")

        if len(self.seeds) != len(set(self.seeds)):
            logging.warning(f"you have duplicated random seeds in {self.seeds}")

        self.exp_func = exp_func
        self.exp_config: Dict = exp_config
        self.repeat_num = repeat_num
    
        self.exp_results = []

    def clear(self):
        """
        clear the cached experiment records
        :return:
        """
        self.exp_results.clear()

    def _setup(self):
        self.clear()

        print("experiment starts ...")
        print(f"repeat running {self.repeat_num} times, random seeds are {self.seeds}")
        print(f"config:")
        for k, v in self.exp_config.items():
            print(f"\t {k} = {v}")

    def run(self):
        """
        run the repeated experiment
        :return:
        """
        global func_wrapper

        def func_wrapper(seed):
            torch.random.manual_seed(seed)
            return self.exp_func(**self.exp_config)

        self._setup()

        for i, seed in tqdm(zip(range(self.repeat_num), self.seeds)):
            data = func_wrapper(seed)
            self.exp_results.append(data)
    
    def run_mp(self, num_process=4):
        """
        multi-process version of run
        """
#         global func_wrapper
        
        def func_wrapper(seed):
            torch.random.manual_seed(seed)
            return self.exp_func(**self.exp_config)

        self._setup()
        
        manager = MultiProcessManager(num_process)
        for run_ret in tqdm(manager.map(func_wrapper, self.seeds), total=self.repeat_num):
            self.exp_results.append(run_ret)
        
#         with mp.Pool(num_process) as pool:
#             for run_ret in tqdm(pool.imap(func_wrapper, self.seeds), total=self.repeat_num):
#                 self.exp_results.append(run_ret)
