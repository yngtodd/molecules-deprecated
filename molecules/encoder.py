import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, kernel1=4, stride1=2, kernel2=4,
                 stride2=2, kernel3=4, stride3=2, kernel4=4, stride4=2,
                 kernel5=4, stride5=2):
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
        stride* : int, default=2
            Stride length for convolutional filter at layer *.
        """
        self.input_size = input_size
        self.latent_size = latent_size
        self.kernel1 = kernel1
        self.stride1 = stride1
        self.kernel2 = kernel2
        self.stride2 = stride2
        self.kernel3 = kernel3
        self.stride3 = stride3
        self.kernel4 = kernel4
        self.stride4 = stride4
        self.kernel5 = kernel5
        self.stride5 = stride5

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_size, 128, self.kernel1, self.stride1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(128, 256, self.kernel2, self.stride2),
            nn.BatchNorm2d(256),
            nn.ELU(),

            nn.Conv2d(256, 256, self.kernel2, self.stride2),
            nn.BatchNorm2d(256),
            nn.ELU(),

            nn.Conv2d(256, 512, self.kernel3, self.stride3),
            nn.BatchNorm2d(512),
            nn.ELU(),

            nn.Conv2d(512, 512, self.kernel4, self.stride4),
            nn.BatchNorm2d(512),
            nn.ELU(),

            nn.Conv2d(512, self.latent_size, self.kernel5, self.stride5),
            nn.BatchNorm2d(self.latent_size),
            nn.ELU()
        )

    def forward(self, input):
        """
        Parameters:
        ----------
        input : float tensor shape=(batch_size, input_size)

        Returns:
        -------
        A float tensor with shape (batch_size, latent_variable_size)
        """

        # Transpose input to the shape of [batch_size, embed_size, seq_len]
        input = t.transpose(input, 1, 2)

        result = self.cnn(input)
        return result.squeeze(2)
