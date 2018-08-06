import torch
import torch.nn as nn


class FCDecoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(FCDecoder, self).__init__()
        """
        Simple fully connected decoder.
        """
        self.latent_size = latent_size
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.Linear(64, self.output_size)
        )

    def forward(self, input):
        return self.decoder(input)
