import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from encoder import Encoder
from decoder import Decoder
from fc_decoder import FCDecoder
from vae import VAE
from vae import latent_loss
from vae import kl_loss
from data import FSPeptide
from data import UnlabeledContact
from utils import AverageMeter
from utils import to_numpy

from tensorboardX import SummaryWriter

import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Setup experiment.')
parser.add_argument('--batch_size', type=int, default=128, help='data batch size.')
parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use cuda.')
parser.add_argument('--half_precision', type=bool, default=False, help='Whether to use half precision')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='./save_points/kl_bce_latent3/',
                    help='Path to where to save the model weights.')
args = parser.parse_args()

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 441))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 441

    return BCE + KLD


def main():
    use_cuda = args.use_cuda
    half_precision = args.half_precision
    print("Cuda set to {} | Cuda availability: {}".format(use_cuda, torch.cuda.is_available()))

    experiment = "vae_latent3"
    #logger = SummaryWriter(log_dir='./logs', comment=experiment)

    train_data = UnlabeledContact(data='/home/ygx/data/fspeptide/fs_peptide.npy')
    print('Number of samples: {}'.format(len(train_data)))
    trainloader = DataLoader(train_data, batch_size=args.batch_size)

    # Contact matrices are 21x21
    input_size = 441

    encoder = Encoder(input_size=input_size, latent_size=3)
    decoder = Decoder(latent_size=3, output_size=input_size)
    vae = VAE(encoder, decoder, use_cuda=use_cuda, half_precision=half_precision)
    #criterion = nn.BCELoss()

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        vae = vae.cuda()
        #criterion = criterion.cuda().half()
        if half_precision:
            encoder = encoder.half()
            decoder = decoder.half()
            vae = vae.half()

    optimizer = optim.SGD(vae.parameters(), lr = 0.001)

    losses = AverageMeter()
    epoch_loss = 0
    total_loss = 0
    for epoch in range(100):
        for batch_idx, data in enumerate(trainloader):
            inputs = data['cont_matrix']
            inputs = inputs.resize_(args.batch_size, 1, 21, 21)
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
                if half_precision:
                    inputs = inputs.half()
            inputs = Variable(inputs)

            # Compute output
            optimizer.zero_grad()
            dec = vae(inputs)

            # Measure the loss
            #kl = kl_loss(vae.z_mean, vae.z_sigma)
            #loss = criterion(dec, inputs) #+ kl # Adding KL is caussing loss > 1
            loss = loss_function(dec, inputs, vae.z_mean, vae.z_sigma)
            losses.update(loss.data[0], inputs.size(0))

            # Compute the gradient
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

            # Logging
            # Adding graph is a lot of overhead
            #logger.add_graph_onnx(vae)

            # log loss values every iteration
            #logger.add_scalar('data/(train)loss_val', losses.val, batch_idx + 1)
            #logger.add_scalar('data/(train)loss_avg', losses.avg, batch_idx + 1)

            # log the layers and layers gradient histogram and distributions
            #for tag, value in vae.named_parameters():
            #    tag = tag.replace('.', '/')
            #    logger.add_histogram('model/(train)' + tag, to_numpy(value), batch_idx + 1)
            #    logger.add_histogram('model/(train)' + tag + '/grad', to_numpy(value.grad), batch_idx + 1)

            # log the outputs of the autoencoder
            #logger.add_image('model/(train)output', make_grid(dec.data), batch_idx + 1)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.data[0]))

        #if epoch < 10:
            # Get latent encoding
            #latent_array = encoder(inputs).data[0].cpu().numpy()
            #filename = 'latent_epoch' + str(epoch)
            #np.save('./latent_saves/kl_bce_latent3/' + filename, latent_array)

            # Get reconstructed image
            #reconstructed_array = vae(inputs).data[0].cpu().numpy().reshape(21, 21)
            #recon_filename = 'reconstructed_epoch' + str(epoch)
            #np.save('./reconstruct_saves/kl_bce_latent3/' + recon_filename, reconstructed_array)

        if epoch % 10 == 0:
            torch.save(vae.state_dict(), args.save_path + 'epoch' + str(epoch))

            #latent_array = encoder(inputs).data[0].cpu().numpy()
            #filename = 'latent_epoch' + str(epoch)
            #np.save('./latent_saves/kl_bce_latent3/' + filename, latent_array)

            reconstructed_array = vae(inputs).data[0].cpu().float().numpy().reshape(21, 21)
            recon_filename = 'reconstructed_epoch' + str(epoch)
            np.save('./reconstruct_saves/kl_bce_latent3/' + recon_filename, reconstructed_array)

        #logger.close()


if __name__ == '__main__':
    main()
