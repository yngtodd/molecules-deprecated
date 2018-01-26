import torch
import numpy as np

import matplotlib.pyplot as plt


def plot_sample(model, inputs, image_dims=21):
    """
    Plot a reconstruced image from variational auotencoder.

    Parameters:
    ----------
    model : Pytorch nn.Module
        - Variational autoencoder model.

    inputs : Pytorch tensor
        - Input tensor for the variational autoencoder
    """
    # Getting started...
    plt.imshow(model(inputs).data[0].numpy().reshape(image_dims, image_dims), cmap='inferno')


def plot_latent():
    """
    Plot the latent space of the variational autoencoder.
    """
