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

import argparse
from hyperspace import hyperdrive


parser = argparse.ArgumentParser(description='Setup experiment.')
parser.add_argument('--batch_size', type=int, default=128, help='data batch size.')
parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use cuda.')
args = parser.parse_args()

# if args.use_cuda == True & torch.cuda.is_available() == False:
#     print("Cuda not available, using CPU!")
#     use_cuda = False
# else:
#     use_cuda = True

use_cuda = False

train_data = FSPeptide(data='/Users/youngtodd/data/fspeptide/train.npy',
                       labels='/Users/youngtodd/data/fspeptide/y_train.npy')

print('Number of samples: {}'.format(len(train_data)))
trainloader = DataLoader(train_data, batch_size=args.batch_size)


def main():
    # Contact matrices are 21x21
    input_size = 441

    encoder = Encoder(input_size=input_size, latent_size=8)
    decoder = Decoder(latent_size=8, output_size=input_size)
    #decoder = FCDecoder(latent_size=8, output_size=input_size)
    vae = VAE(encoder, decoder)

    #print('decoder latent_size: {}'.format(decoder.latent_size))
    print(vae)

    if use_cuda:
        vae.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)

    epoch_loss = 0
    total_loss = 0
    for epoch in range(100):
        for i, data in enumerate(trainloader):
            inputs = data['cont_matrix']
            inputs = inputs.resize_(args.batch_size, 1, 21, 21)
            inputs = inputs.float()
            #print('input shape: {}'.format(inputs.shape))
            if use_cuda:
                inputs.cuda()
            inputs = Variable(inputs)
            optimizer.zero_grad()
            dec = vae(inputs)
            #print('dec shape {}'.format(dec.shape))
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            epoch_loss = loss.data[0]
        print(epoch, epoch_loss)
        total_loss += epoch_loss

    return total_loss


if __name__ == '__main__':
    main()
