import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        """
        Parameters:
        ----------
        input_size : int
            Input dimension for the data.
        latent_size : int
            Latent space dimension
        kernel* : int, defualt=4
            Convolutional filter size for layer *.
        stride* : int, default=1
            Stride length for convolutional filter at layer *.
        """
        self.input_size = input_size
        self.latent_size = latent_size

        self.cnn_encoder = nn.Sequential(
            nn.Linear(self.input_size, 500),
            nn.ReLU(),

            nn.Linear(500, 250),
            nn.ReLU(),

            nn.Linear(250, 128),
            nn.ReLU()
        )

        self.fc = nn.Linear(128, self.latent_size) # 64*2*2


    def forward(self, input):
        """
        Parameters:
        ----------
        input : float tensor shape=(batch_size, input_size)

        Returns:
        -------
        A float tensor with shape (batch_size, latent_variable_size)
        """
        out = self.cnn_encoder(input)
        out = self.fc(out)
        return out
