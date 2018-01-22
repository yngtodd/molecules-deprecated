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

        self.fc = nn.Linear(self.latent_size, 256)

        self.cnn_decoder1 = nn.Sequential(
            nn.ConvTranspose2d(1, 32, self.kernel1, self.stride1, padding=2),
            nn.MaxPool2d(2),
            nn.ELU()
        )

        self.cnn_decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, self.kernel2, self.stride2),
            nn.MaxPool2d(2),
            nn.ELU()
        )

        self.cnn_decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, self.kernel3, self.stride3),
            nn.MaxPool2d(2),
            nn.ELU()
        )

        self.cnn_decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, self.kernel4, self.stride4),
            nn.MaxPool2d(2),
            nn.ELU()
        )

        #self.fc2 = nn.Linear(128, self.output_size)

    def forward(self, latent_input):
        """
        Parameters:
        ----------
        input : float tensor shape=(batch_size, input_size)

        Returns:
        -------
        A float tensor with shape (batch_size, output_size)
        """
        print("latent input in decoder {}".format(latent_input))
        out = self.fc(latent_input)
        #out = torch.transpose(latent_input, 4, 4)
        #out = out.view(out.size(0), 1, 16, 16)
        #print('output of decoder fc1 has shape {}'.format(out.shape))
        #latent_input = latent_input.unsqueeze(0)
        #out = latent_input.view(latent_input.size(0), 1, 4, 4)
        #print("unsqueezed latent input has shape {}".format(latent_input.shape))
        out = out.view(out.size(0), 1, 16, 16)
        print('output of linear layer has shape {}'.format(out.shape))
        out = self.cnn_decoder1(out)
        print('output of decoder1 cnn has shape {}'.format(out.shape))
        #out = out.view(out.size(0), -1)
        #print('input into fully connected layer of decoder has shape {}'.format(out.shape))
        #out = self.fc2(out)
        #print('output of decoder has shape {}\n'.format(out.shape))
        return out
