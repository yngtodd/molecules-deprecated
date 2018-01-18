import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_size, output_size, kernel1=4, stride1=2, kernel2=4,
                 stride2=2, kernel3=4, stride3=2, kernel4=4, stride4=2,
                 kernel5=4, stride5=2):
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
            nn.ConvTranspose2d(self.latent_size, 32, self.kernel1, self.stride1),
            nn.AdaptiveMaxPool2d(32),
            nn.ELU(),

            nn.ConvTranspose2d(32, 64, self.kernel2, self.stride2),
            nn.AdaptiveMaxPool2d(64),
            nn.ELU(),

            nn.ConvTranspose2d(64, 64, self.kernel3, self.stride3),
            nn.AdaptiveMaxPool2d(64),
            nn.ELU(),

            nn.ConvTranspose2d(64, 128, self.kernel4, self.stride4),
            nn.AdaptiveMaxPool2d(128),
            nn.ELU(),

            nn.ConvTranspose2d(128, 128, self.kernel5, self.stride5),
            nn.AdaptiveMaxPool2d(128),
            nn.ELU(),

            nn.Linear(128, self.output_size)
        )


        def forward(self, latent_input):
            """
            Parameters:
            ----------
            input : float tensor shape=(batch_size, input_size)
            Returns:
            -------
            A float tensor with shape (batch_size, out_size)
            """
            # Transpose input to the shape of [batch_size, embed_size, seq_len]
            input = torch.transpose(latent_input, 1, 2)

            result = self.cnn(input)
            return result.squeeze(2)
