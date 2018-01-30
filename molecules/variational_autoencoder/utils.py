"""Utility functions"""

def to_numpy(x):
    """
    Convert Pytorch tensor to numpy array.

    Parameters:
    ----------
    x : Pytorch tensor. 

    Returns:
    -------
    Numpy array

    References:
    ---------
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20
    """
    return x.data.cpu().numpy()


class AverageMeter(object):
    """
    Computes and stores the average and current loss value.
    
    References:
    ----------
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
