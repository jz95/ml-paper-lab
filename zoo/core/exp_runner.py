import torch
from zoo.utils import isnotebook
from collections import Sequence



if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
    
class ExpRunner(object):
    """
    Repeat Exp Runner 
    """
    def __init__(self, repeat_num=5, seeds=None, process_num=4):
        if seeds is None:
            self.seeds = range(repeat_num)
        elif isinstance(seeds, Sequence):
            self.seeds = list(map(int, seeds))
            assert len(self.seeds) == repeat_num, f'seeds num: {len(self.seeds)} neq repeat_num: {repeat_num}'
        else:
            raise TypeError(f"unsupport seeds type: {type(seeds)}")
        
        self.process_num = process_num
        self.repeat_num = repeat_num
    
        self.exp_results = []
    
    def run(self, func, **kwargs):
        global func_wrapper
        def func_wrapper(seed):
            torch.random.manual_seed(seed)
            return func(**kwargs)
        
        for i, seed in tqdm(zip(range(self.repeat_num), self.seeds)):
            data = func_wrapper(seed)
            self.exp_results.append(data)
    
    def run_mp(self, func, **kwargs):
        """ 
        multi-process version
        """
        import torch.multiprocessing as mp
        from copy import deepcopy
        
        global func_wrapper
        
        def func_wrapper(seed):
            torch.random.manual_seed(seed)
            return func(**kwargs)
        
        with mp.Pool(self.process_num) as pool:
            data = []
            for run_ret in tqdm(pool.imap(func_wrapper, self.seeds), total=self.repeat_num):
                self.exp_results.append(run_ret)

        
