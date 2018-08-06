"""Utility functions"""

def to_numpy(tensor):
    """
    Convert Pytorch tensor to numpy array.

    Parameters:
    ----------
    tensor : Pytorch tensor.

    Returns:
    -------
    Numpy array

    References:
    ---------
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20
    """
    return x.data.cpu().numpy()


def to_img(tensor, heigh, width):
    """
    Convert Pytorch tensor to image.

    Parameters:
    ----------
    tensor : Pytorch tensor

    height : int
        Height of the image.
    
    width : int
        Width of the image.

    Returns:
    -------
    imgs : Pytorch tensor
        Tensor of shape (batch_size, 1, height, width)
    """
    tensor = tensor.clamp(0, 1)
    img = tensor.view(tensor.size(0), 1, height, width)
    return img 


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


def print_progress(epoch, batch_idx, data, dataloader, loss):
    """
    Print training progress.
    """
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(dataloader.dataset),
        100. * batch_idx / len(dataloader), loss.data[0]))
