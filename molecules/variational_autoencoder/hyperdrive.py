import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from encoder import Encoder
from decoder import Decoder
from vae import VAE
from vae import latent_loss
from data import FSPeptide

import argparse
from hyperspace import hyperdrive


parser = argparse.ArgumentParser(description='Setup experiment.')
parser.add_argument('--results_dir', type=str, help='Path to results directory.')
parser.add_argument('--batch_size', type=int, default=128, help='data batch size.')
parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use cuda.')
parser.add_argument('--deadline', type=int, default=14400, help='Deadline (seconds) to finish within.')
args = parser.parse_args()


train_data = FSPeptide(data='/Users/youngtodd/data/fspeptide/train.npy',
                       labels='/Users/youngtodd/data/fspeptide/y_train.npy')

print('Number of samples: {}'.format(len(train_data)))
trainloader = DataLoader(train_data, batch_size=batch_size)


def objective(params):
    """
    Objective function to be minimized: loss with respect to our hyperparameters.
    """
    enc_kernel1 = int(params[0])
    enc_kernel2 = int(params[1])
    enc_kernel3 = int(params[2])
    dec_kernel1 = int(params[3])
    dec_kernel2 = int(params[4])
    dec_kernel3 = int(params[5])
    
    # Contact matrices are 21x21
    input_dim = 441

    encoder = Encoder(input_size=input_dim, latent_size=8, kernel1=enc_kernel1,
                      kernel2=enc_kernel2, kernel3=enc_kernel3)

    decoder = Decoder(latent_dim=8, output_size=input_size, kernel1=dec_kernel1,
                      kernel2=dec_kernel2, kernel3=dec_kernel3)

    vae = VAE(encoder, decoder)
    criterion = nn.MSELoss()

    use_cuda = args.use_cuda
    if use_cuda:
        encoder = encoder.cuda()
        deconder = decoder.cuda()
        vae = vae.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)

    epoch_loss = 0
    total_loss = 0
    for epoch in range(100):
        for i, data in enumerate(trainloader, 0):
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
            epoch_loss = loss.data[0]
        print(epoch, epoch_loss)
        total_loss += epoch_loss

    return total_loss


def main():
    hparams = [(2, 10),       # encoder kernel1
               (2, 10),       # encoder kernel2
               (2, 10),       # encoder kernel3
               (2, 10),       # decoder kernel1
               (2, 10),       # decoder kernel2
               (2, 10)]       # decoder kernel3

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0,
               deadline=args.deadline)

if __name__ == '__main__':
    main()
