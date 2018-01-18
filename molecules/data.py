import os
import torch
import numpy as np
from torch.utils.data import Dataset


class FSPeptide(Dataset):
    """
    FS-Peptide Dataset.
    """
    def __init__(self, data, labels, transform=None):
        """
        Parameters:
        ----------
        data : str
            Path to the data in numpy format.
        label : str
            Path to the labels.
        transform : callable, optional
            Optional transform to be applied on a sample.
        """
        self.data = np.load(data)
        self.labels = np.load(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cont_mat = self.data[index]
        cont_label = self.labels[index]
        sample = {'cont_matrix': cont_mat, 'label': cont_label}

        if self.transform:
            sample = self.transform(sample)

        return sample
