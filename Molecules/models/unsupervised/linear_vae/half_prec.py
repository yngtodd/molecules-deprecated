import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from encoder import Encoder
from decoder import Decoder
from fc_decoder import FCDecoder
from vae import VAE
from vae import latent_loss
from data import FSPeptide
from data import UnlabeledContact

import numpy as np
import argparse

import time
from meters import AverageMeter


parser = argparse.ArgumentParser(description='Setup experiment.')
parser.add_argument('--data_dir', type=str, default='/home/ygx/data/fspeptide/fs_peptide.npy')
parser.add_argument('--batch_size', type=int, default=128, help='data batch size.')
parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use cuda.')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='/home/ygx/molecules/molecules/variational_autoencoder/save_points/saves_latent3/',
                    help='Path to where to save the model weights.')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

def main():
    use_cuda = args.use_cuda

    train_data = UnlabeledContact(data=args.data_dir)
    print('Number of samples: {}'.format(len(train_data)))
    trainloader = DataLoader(train_data, batch_size=args.batch_size)

    # Contact matrices are 21x21
    input_size = 441

    encoder = Encoder(input_size=input_size, latent_size=3)
    decoder = Decoder(latent_size=3, output_size=input_size)
    vae = VAE(encoder, decoder, use_cuda=use_cuda)
    criterion = nn.MSELoss()

    if use_cuda:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        encoder = encoder.cuda().half()
        decoder = decoder.cuda().half()
        vae = nn.DataParallel(vae)
        vae = vae.cuda().half()
        criterion = criterion.cuda().half()

    optimizer = optim.SGD(vae.parameters(), lr = 0.01)

    clock = AverageMeter(name='clock16', rank=0)
    epoch_loss = 0
    total_loss = 0
    end = time.time()
    for epoch in range(15):
        for batch_idx, data in enumerate(trainloader):
            inputs = data['cont_matrix']
 #           inputs = inputs.resize_(args.batch_size, 1, 21, 21)
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda().half()
            inputs = Variable(inputs)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

            clock.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.data[0]))

    clock.save(path='/home/ygx/libraries/mds/molecules/molecules/linear_vae')

if __name__ == '__main__':
    main()
