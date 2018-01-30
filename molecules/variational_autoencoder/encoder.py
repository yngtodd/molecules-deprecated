import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, kernel1=4, stride1=1, kernel2=4,
                 stride2=1, kernel3=4, stride3=1, kernel4=4, stride4=1):
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
        self.kernel1 = kernel1
        self.stride1 = stride1
        self.kernel2 = kernel2
        self.stride2 = stride2
        self.kernel3 = kernel3
        self.stride3 = stride3
        self.kernel4 = kernel4
        self.stride4 = stride4

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, self.kernel1, self.stride1, padding=2),
            nn.AdaptiveMaxPool2d(16),
            nn.SELU(),

            nn.Conv2d(16, 16, self.kernel2, self.stride2, padding=2),
            nn.AdaptiveMaxPool2d(16),
            nn.SELU(),

            nn.Conv2d(16, 32, self.kernel3, self.stride3, padding=2),
            nn.AdaptiveMaxPool2d(32),
            nn.SELU(),

            nn.Conv2d(32, 32, self.kernel4, self.stride4, padding=2),
            nn.AdaptiveMaxPool2d(2),
            nn.SELU()
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
        #print('output size of encoder: {}'.format(out.size()))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
