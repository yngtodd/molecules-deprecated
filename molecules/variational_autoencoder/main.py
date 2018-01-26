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

import argparse


parser = argparse.ArgumentParser(description='Setup experiment.')
parser.add_argument('--batch_size', type=int, default=128, help='data batch size.')
parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use cuda.')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='/home/ygx/molecules/molecules/variational_autoencoder/save_points/saves_latent3/',
                    help='Path to where to save the model weights.')
args = parser.parse_args()

use_cuda = True 

#train_data = FSPeptide(data='/home/ygx/data/fspeptide/train.npy',
#                       labels='/home/ygx/data/fspeptide/y_train.npy')

train_data = UnlabeledContact(data='/home/ygx/data/fspeptide/fs_peptide.npy')

print('Number of samples: {}'.format(len(train_data)))
trainloader = DataLoader(train_data, batch_size=args.batch_size)


def main():
    # Contact matrices are 21x21
    input_size = 441 

    encoder = Encoder(input_size=input_size, latent_size=3)
    #encoder = nn.DataParallel(encoder, device_ids=None)

    if use_cuda:
        encoder = encoder.cuda()

    decoder = Decoder(latent_size=3, output_size=input_size)
    #decoder = nn.DataParallel(decoder, device_ids=None)

    if use_cuda:
        decoder = decoder.cuda()

    vae = VAE(encoder, decoder, use_cuda=use_cuda)
    #vae = nn.DataParallel(vae, device_ids=None)

    if use_cuda:
        vae = vae.cuda()

    criterion = nn.MSELoss()

    if use_cuda:
        criterion = criterion.cuda()

    optimizer = optim.SGD(vae.parameters(), lr = 0.01)

    epoch_loss = 0
    total_loss = 0
    for epoch in range(100):
        for batch_idx, data in enumerate(trainloader):
            inputs = data['cont_matrix']
            inputs = inputs.resize_(args.batch_size, 1, 21, 21)
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.data[0]))
        
        if epoch % 10 == 0:
            torch.save(vae.state_dict(), args.save_path + 'epoch' + str(epoch))


if __name__ == '__main__':
    main()
