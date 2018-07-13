# activate theano on gpu
import os;
#os.environ['THEANO_FLAGS'] = "device=gpu";
#import theano;
#theano.config.floatX = 'float32';

import numpy as np;
import sys, os;
import gzip;
from six.moves import cPickle;
from vae_conv import conv_variational_autoencoder;
from keras import backend as K;
import pdb
channels = 1;
batch_size = 32;
conv_layers = 3;
feature_maps = [128,128,128,128];
filter_shapes = [(3,3),(3,3),(3,3),(3,3)];
strides = [(1,1),(2,2),(1,1),(1,1)];
dense_layers = 1;
dense_neurons = [128];
dense_dropouts = [0];
latent_dim = 3;
epochs = 1;
nb_start = 0;
nb_end = 50;

epochs = 1
batch_size = 8
nb_start = 0
nb_end = 50
dim = 21
X_train = np.random.randn(2,dim,dim,1)
X_test = np.random.randn(1,dim,dim,1)

X_train = np.pad(X_train, ((0,0), (1,1), (1,1), (0,0)), 'constant')[:, 1:, 1:, :]
X_test = np.pad(X_test, ((0,0), (1,1), (1,1), (0,0)), 'constant')[:, 1:, 1:, :]
print(X_train)

feature_maps = feature_maps[0:conv_layers];
filter_shapes = filter_shapes[0:conv_layers];
strides = strides[0:conv_layers];

image_size = X_train.shape[1:];
autoencoder = conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
           filter_shapes,strides,dense_layers,dense_neurons,dense_dropouts,latent_dim);

for i in range (nb_start, nb_end):
    if i == 0:
       print("skipping - no previous saved file to load")
    # load model;
    else:
       autoencoder.load("./model/model_%i" %i)
    # train model;
    print X_train.shape
    print X_test.shape
    pdb.set_trace()
    autoencoder.train(X_train[0:],batch_size,epochs=epochs,
              validation_data=(), checkpoint=False,filepath="./savedweights.dat");
    # save model;
    print "pass"
    autoencoder.save(filepath="./model/model_%i" %(i+1));
    # save loss over train & validation;
    np.savetxt('./hist/history.losses_%i' %(i+1), autoencoder.history.losses, delimiter=',');
    np.savetxt('./hist/history.val_losses_%i' %(i+1), autoencoder.history.val_losses, delimiter=',');
    print('completed %i epochs' % ((i+1)*epochs));
