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


parser = argparse.ArgumentParser(description='Setup experiment.')
parser.add_argument('--input_size', type=int, default=441, help='flattened image size.')
parser.add_argument('--latent_size', type=int, default=3, help='latent dimension')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for net')
parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use cuda.')
parser.add_argument('--model_path', type=str, default='/home/ygx/molecules/molecules/variational_autoencoder/save_points/saves_latent3/',
                    help='Path to saved model weights.')
parser.add_argument('--model_name', type=str, default='epoch90.pt', help='name of the saved model')
parser.add_argument('--latent_save_path', type=str, 
                    default='/home/ygx/molecules/molecules/variational_autoencoder/generate_latent/fs_latent3_epoch90/', 
                    help='path to save generated latent dimensions')
parser.add_argument('--recon_save_path', type=str, 
                    default='/home/ygx/molecules/molecules/variational_autoencoder/generate_recon/fs_latent3_epoch90/', 
                    help='path to save reconstructed images')
args = parser.parse_args()


def main():
    """
    Generate images from a saved model
    """
    train_data = UnlabeledContact(data='/home/ygx/data/fspeptide/fs_peptide.npy')
    print('Number of samples: {}'.format(len(train_data)))
    trainloader = DataLoader(train_data, batch_size=args.batch_size)

    encoder = Encoder(input_size=args.input_size, latent_size=args.latent_size)
    decoder = Decoder(latent_size=args.latent_size, output_size=args.input_size)
    vae = VAE(encoder, decoder, use_cuda=args.use_cuda)
    
    # Load saved model
    vae.load_state_dict(torch.load(args.model_path + args.model_name))

    if args.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        vae = vae.cuda()
    
    latent_arrys = []
    recon_arrys = []
    for batch_idx, data in enumerate(trainloader):
        inputs = data['cont_matrix']
        inputs = inputs.resize_(args.batch_size, 1, 21, 21)
        inputs = inputs.float()
        if args.use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)

        latent_array = encoder(inputs).data.cpu().numpy()
        #print('latent_array has shape {}'.format(latent_array.shape))
        latent_arrys.append(latent_array)

        reconstructed_array = vae(inputs).data.cpu().numpy()
        recon_arrys.append(reconstructed_array)
        
        if batch_idx % 100 == 0:
            print('Saving progress: {:.3f}%'.format(batch_idx * 100. / len(trainloader)))

    print('\nNumber of images prepared: {}'.format(len(latent_arrys))) 
    latent_stacked = np.stack(latent_arrys, axis=0) 
    latent_filename = 'latent_imgs' 
    np.save(args.latent_save_path + latent_filename, latent_stacked)

    recon_stacked = np.stack(recon_arrys, axis=0)
    recon_filename = 'recon_imgs' 
    np.save(args.recon_save_path + recon_filename, recon_stacked)


if __name__=='__main__':
    main()
