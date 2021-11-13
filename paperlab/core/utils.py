import torch
import torch.multiprocessing as mp
import threading
import time

from typing import List
from torch.utils.data import DataLoader
from .base import BaseModel
from collections.abc import Mapping, Sequence


class MultiProcessManager(object):
    """
    The multi process Manager, 
    Jupyternotebook notebook environment dosent support `spawn` context under torch.multiprocessing.pool
    the torch.multiprocessing.pool doesnt support 
    """
    def __init__(self, num_parallel):
        self.process: List[mp.Process] = []
        self.num_parallel = num_parallel

    def map(self, target, args: List):
        num_task, num_task_done, next_task_id = len(args), 0, 0
        lock = threading.Lock()
        ret = [None for _ in range(num_task)]
        
        def _watch_dog():
            nonlocal num_task_done
            while num_task_done < num_task:
                time.sleep(0.1)
                i = 0
                while i < len(self.process):
                    p: mp.Process = self.process[i]
                    if not p.is_alive():
                        self.process.pop(i)
                        with lock:
                            num_task_done += 1
                        break
                    i += 1

        watch_dog = threading.Thread(target=_watch_dog, name='watch_dog')
        watch_dog.start()
        
        q = mp.Queue()
        def target_wrapper(task_id, args):
            ret = target(args)            
            q.put_nowait((task_id, ret))

        while True:

            if not q.empty():
                task_id, task_out = q.get_nowait()            
                ret[task_id] = task_out
            
            with lock:
                if num_task_done >= num_task:
                    return ret

                if num_task_done < num_task and next_task_id < num_task and len(self.process) < self.num_parallel:
                    arg = args[next_task_id]
                    p = mp.Process(target=target_wrapper, args=(next_task_id, arg, ))
                    p.start()
                    self.process.append(p)
                    next_task_id += 1
            
            time.sleep(0.1)

        return ret


def wrap_data(data):
    if torch.is_tensor(data):
        return data.cuda()
    elif isinstance(data, Mapping):
        return {key: wrap_data(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        data_type = type(data)
        return data_type(*(wrap_data(elem) for elem in data))
    elif isinstance(data, Sequence):
        return [wrap_data(elem) for elem in data]
    
    raise TypeError(f"unsupportted type of data {type(data)}")


def evaluate_loss(model: BaseModel, dataloader: DataLoader):
    loss_ = 0.
    with torch.no_grad():
        for data in dataloader:
            if torch.cuda.is_available():
                data = wrap_data(data)
            loss_ += model.compute_loss(data, reduction='sum').item()

    return loss_ / len(dataloader.dataset)
