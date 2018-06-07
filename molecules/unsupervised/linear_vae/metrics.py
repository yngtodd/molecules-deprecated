import torch
from torch.nn import functional as F


def entropy_kl_loss(recon_x, img, mu, logvar, batch_size, img_dimension):
    """
    Loss function: binary cross entropy plus KL divergence.

    Parameters:
    ----------
    recons_x : Pytorch tensor
        Reconstructed tensor from VAE.

    img : Pytorch tensor
        True image the VAE is trying to reconstruct.

    mu : Pytorch tensor
        Mean tensor from VAE

    logvar : Pytorch tensor
        Log variance from VAE

    batch_size : int
        Batch size used for training.
        * used to normalize KL divergence

    img_dimension : int
        Flattened image dimension.
        * number of pixels in image

    Returns:
    -------
    Binary cross entropy + KL divergence loss : float

    References:
    ----------
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    Appendix B
    https://arxiv.org/abs/1312.6114
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_dimension))
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * img_dimension 
    return BCE + KLD
