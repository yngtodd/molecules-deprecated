import torch
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value

    Parameters:
    ----------
    name : str
        Name of the object to be tracked.

    rank : int
        Rank of the values computed (if distributed).
    """
    def __init__(self, name, rank=None):
        self.name = name
        self.rank = rank
        self.reset()

    def __str__(self):
        return f'Average Meter for {self.name} on rank {self.rank}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = np.empty(0)
        self.avgs = np.empty(0)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals = np.append(self.vals, self.val)
        self.avgs = np.append(self.avgs, self.avg)

    def save(self, path):
        np.save(path + '/' + self.name + '_avgs' + str(self.rank), self.avgs)
        np.save(path + '/' + self.name + '_vals' + str(self.rank), self.vals)
