import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        output_size : int
            Output dimension for the data. Should equal input_dimension of AE.
        kernel* : int, defualt=4
            Convolutional filter size for layer *.
        stride* : int, default=2
            Stride length for convolutional filter at layer *.
        """
        self.latent_size = latent_size
        self.output_size = output_size

        self.decode = nn.Sequential(
            nn.Linear(self.latent_size, 256),
            nn.ReLU(),

            nn.Linear(256, 500),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(500, self.output_size),
            nn.Sigmoid()
       )

    def forward(self, latent_input):
        """
        Parameters:
        ----------
        input : float tensor shape=(batch_size, input_size)

        Returns:
        -------
        A float tensor with shape (batch_size, output_size)
        """
        out = self.decode(latent_input)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
