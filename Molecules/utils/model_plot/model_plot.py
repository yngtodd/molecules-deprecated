from __future__ import print_function
import numpy as np;
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Dropout
from keras.layers import Convolution2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras import objectives
import warnings
from keras.utils import plot_model


def build_training_data():
    X = np.zeros([1100000, 28, 28])
    Y = np.zeros([1100000, 1, 1])
    x_raw = X[0:500000]
    print("Dataset dimension:", np.shape(x_raw))
    # no of trajectory files and frames in each file
    n_traj = 50
    f_traj = 10000 
    # fraction of train, test and pred data separation
    sep_train = 0.8     
    sep_test = 0.9    
    sep_pred = 1 
    # choice to flatten data: "0" for NO & "1" for YES
    choice = 0
    # row and column dimension for each frame
    row = 28
    col = 28   
    # padding: use this incase diemsion mismatch for encoders
    # pad_row and pad_col are row or colums to be added 
    pad_row = 0
    pad_col = 0      
    # end define parameters 
    # padding
    row_dim_array = row + pad_row    
    col_dim_array = col + pad_col    
    # reshape data according to the choice of flatteing
    if choice == 0:
    	new_shape = (len(x_raw),row_dim_array,col_dim_array)  
    if choice == 1:
    	new_shape = (len(x_raw),row_dim_array*col_dim_array)
    add_zero = np.zeros(new_shape,dtype = x_raw.dtype)        
    if choice == 0:
    	add_zero[0:x_raw.shape[0],0:x_raw.shape[1],0:x_raw.shape[2]] = x_raw 
    if choice == 1:
    	add_zero[0:x_raw.shape[0],0:x_raw.shape[1]] = x_raw 
    x_raw = add_zero
    sep_1 = int(x_raw.shape[0]*sep_train) 
    sep_2 = int(x_raw.shape[0]*sep_test)    
    sep_3 = int(x_raw.shape[0]*sep_pred)
    x_train_raw = x_raw[:sep_1]
    x_test_raw = x_raw[sep_1:sep_2] 
    x_pred_raw = x_raw[sep_2:sep_3]

    '''
    Convolutional variational autoencoder in Keras
    
    Reference: "Auto-Encoding Variational Bayes" (https://arxiv.org/abs/1312.6114)
    '''

class conv_variational_autoencoder(object):
    '''
    variational autoencoder class
    
    parameters:
      - image_size: tuple
        height and width of images
      - channels: int
        number of channels in input images
      - conv_layers: int
        number of encoding/decoding convolutional layers
      - feature_maps: list of ints
        number of output feature maps for each convolutional layer
      - filter_shapes: list of tuples
        convolutional filter shape for each convolutional layer
      - strides: list of tuples
        convolutional stride for each convolutional layer
      - dense_layers: int
        number of encoding/decoding dense layers
      - dense_neurons: list of ints
        number of neurons for each dense layer
      - dense_dropouts: list of float
        fraction of neurons to drop in each dense layer (between 0 and 1)
      - latent_dim: int
        number of dimensions for latent embedding
      - activation: string (default='relu')
        activation function to use for layers
      - eps_mean: float (default = 0.0)
        mean to use for epsilon (target distribution for embedding)
      - eps_std: float (default = 1.0)
        standard dev to use for epsilon (target distribution for embedding)
       
    methods:
      - train(data,batch_size,epochs=1,checkpoint=False,filepath=None)
        train network on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
      - return_embeddings(data)
        return the embeddings for given data
      - generate(embedding)
        return a generated output given a latent embedding
    '''

    def __init__(self,image_size,channels,conv_layers,feature_maps,filter_shapes,
                 strides,dense_layers,dense_neurons,dense_dropouts,latent_dim,
                 activation='relu',eps_mean=0.0,eps_std=1.0):
        
        # check that arguments are proper length;
        if len(filter_shapes)!=conv_layers:
            raise Exception("number of convolutional layers must equal length of filter_shapes list")
        if len(strides)!=conv_layers:
            raise Exception("number of convolutional layers must equal length of strides list")
        if len(feature_maps)!=conv_layers:
            raise Exception("number of convolutional layers must equal length of feature_maps list")
        if len(dense_neurons)!=dense_layers:
            raise Exception("number of dense layers must equal length of dense_neurons list")
        if len(dense_dropouts)!=dense_layers:
            raise Exception("number of dense layers must equal length of dense_dropouts list")
        
        # even shaped filters may cause problems in theano backend;
        even_filters = [f for pair in filter_shapes for f in pair if f % 2 == 0];
        if K.image_dim_ordering() == 'th' and len(even_filters) > 0:
            warnings.warn('Even shaped filters may cause problems in Theano backend')
        if K.image_dim_ordering() == 'channels_first' and len(even_filters) > 0:
            warnings.warn('Even shaped filters may cause problems in Theano backend')
        
        self.eps_mean = eps_mean
        self.eps_std = eps_std
        self.image_size = image_size
        
        # define input layer
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
            self.input = Input(shape=(channels,image_size[0],image_size[1]))
        else:
            self.input = Input(shape=(image_size[0],image_size[1],channels))
                    
        # define convolutional encoding layers
        self.encode_conv = [];
        layer = Convolution2D(feature_maps[0],filter_shapes[0],padding='same',
                              activation=activation,strides=strides[0])(self.input)
        self.encode_conv.append(layer)
        for i in range(1,conv_layers):
            layer = Convolution2D(feature_maps[i],filter_shapes[i],
                                  padding='same',activation=activation,
                                  strides=strides[i])(self.encode_conv[i-1])
            self.encode_conv.append(layer)
        
        # define dense encoding layers
        self.flat = Flatten()(self.encode_conv[-1])
        self.encode_dense = []
        layer = Dense(dense_neurons[0],activation=activation) (Dropout(dense_dropouts[0])(self.flat))
        self.encode_dense.append(layer)
        for i in range(1,dense_layers):
            layer = Dense(dense_neurons[i],activation=activation) (Dropout(dense_dropouts[i])(self.encode_dense[i-1]))
            self.encode_dense.append(layer)
        
        # define embedding layer
        self.z_mean = Dense(latent_dim)(self.encode_dense[-1])
        self.z_log_var = Dense(latent_dim)(self.encode_dense[-1]) 
        self.z = Lambda(self._sampling, output_shape=(latent_dim,)) ([self.z_mean, self.z_log_var])
                
        # save all decoding layers for generation model
        self.all_decoding=[]
                
        # define dense decoding layers
        self.decode_dense = []
        layer = Dense(dense_neurons[-1], activation=activation)
        self.all_decoding.append(layer)
        self.decode_dense.append(layer(self.z))
        for i in range(1,dense_layers):
            layer = Dense(dense_neurons[-i-1],activation=activation)
            self.all_decoding.append(layer)
            self.decode_dense.append(layer(self.decode_dense[i-1]))
        
        # dummy model to get image size after encoding convolutions
        self.decode_conv = []
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
            dummy_input = np.ones((1,channels,image_size[0],image_size[1]))
        else:
            dummy_input = np.ones((1,image_size[0],image_size[1],channels))

        dummy = Model(self.input, self.encode_conv[-1])
        conv_size = dummy.predict(dummy_input).shape
        layer = Dense(conv_size[1]*conv_size[2]*conv_size[3],activation=activation)
        self.all_decoding.append(layer)
        self.decode_dense.append(layer(self.decode_dense[-1]))
        reshape = Reshape(conv_size[1:])
        self.all_decoding.append(reshape)
        self.decode_conv.append(reshape(self.decode_dense[-1]))
        
        # define deconvolutional decoding layers
        for i in range(1,conv_layers):
            if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
                dummy_input = np.ones((1,channels,image_size[0],image_size[1]))
            else:
                dummy_input = np.ones((1,image_size[0],image_size[1],channels))
            dummy = Model(self.input, self.encode_conv[-i-1])
            conv_size = list(dummy.predict(dummy_input).shape)
            
            if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
                conv_size[1] = feature_maps[-i]
            else:
                conv_size[3] = feature_maps[-i]
            
            layer = Conv2DTranspose(feature_maps[-i-1],filter_shapes[-i],
                                    padding='same',activation=activation,
                                    strides=strides[-i])
            self.all_decoding.append(layer)
            self.decode_conv.append(layer(self.decode_conv[i-1]))
        
        layer = Conv2DTranspose(channels,filter_shapes[0],padding='same',
                                activation='sigmoid',strides=strides[0])
        self.all_decoding.append(layer)
        self.output=layer(self.decode_conv[-1])

        # build model
        self.model = Model(self.input, self.output)
        self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=self.optimizer, loss=self._vae_loss)
        print("Model summary:")
        self.model.summary()
        #plot model
        plot_model(self.model, show_shapes = 'True', show_layer_names = 'True', 
                   to_file='./images/vae_conv.png')
        
        # model for embeddings
        self.embedder = Model(self.input, self.z_mean)
        
        # model for generation
        self.decoder_input = Input(shape=(latent_dim,))
        self.generation = []
        self.generation.append(self.all_decoding[0](self.decoder_input))
        for i in range(1, len(self.all_decoding)):
            self.generation.append(self.all_decoding[i](self.generation[i-1]))
            self.generator = Model(self.decoder_input, self.generation[-1])
        
    def _sampling(self,args):
        '''
        sampling function for embedding layer
        '''
        z_mean,z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=self.eps_mean, stddev=self.eps_std)
        return z_mean + K.exp(z_log_var) * epsilon
        
    def _vae_loss(self,input,output):
        '''
        loss function for variational autoencoder
        '''
        input_flat = K.flatten(input)
        output_flat = K.flatten(output)
        xent_loss = self.image_size[0] * self.image_size[1] * objectives.binary_crossentropy(input_flat,output_flat)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss
        
    def train(self,data,batch_size,epochs=1,validation_data=None,
              checkpoint=False,filepath=None):
        '''
        train network on given data;
        
        parameters:
          - data: numpy array
            input data
          - batch_size: int
            number of records per batch
          - epochs: int (default: 1)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation data
          - checkpoint: boolean (default: False)
            whether or not to save model after each epoch
          - filepath: string (optional)
            path to save model if checkpoint is set to True
        
        outputs:
            None
        '''
        if checkpoint==True and filepath==None:
            raise Exception("Please enter a path to save the network") 

        self.model.fit(data,data,batch_size,epochs=epochs,shuffle=True,
                       validation_data=(data,data),callbacks=[history])
    
    def save(self, filepath):
        '''
        save the model weights to a file
        
        parameters:
          - filepath: string
            path to save model weights
        
        outputs:
            None
        '''
        self.model.save_weights(filepath)
        
    def load(self,filepath):
        '''
        load model weights from a file
        
        parameters:
          - filepath : string
            path from which to load model weights
        
        outputs:
            None
        '''
        self.model.load_weights(filepath)
    
    def decode(self,data):
        '''
        return the decodings for given data
        
        parameters:
          - data: numpy array
            input data
        
        outputs:
            numpy array of decodings for input data
        '''
        return self.model.predict(data)
    
    def return_embeddings(self,data):
        '''
        return the embeddings for given data
        
        parameters:
          - data: numpy array
            input data
        
        outputs:
            numpy array of embeddings for input data
        '''
        return self.embedder.predict(data)

    def generate(self,embedding):
        '''
        return a generated output given a latent embedding
        
        parameters:
          - data: numpy array
            latent embedding
        
        outputs:
            numpy array of generated output
        '''
        return self.generator.predict(embedding);

if __name__ == "__main__":

    import sys
    import os
    import gzip
    from six.moves import cPickle
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # define parameters
    channels = 1
    batch_size = 1000
    conv_layers = 4
    feature_maps = [64,64,64,64]
    filter_shapes = [(3,3),(3,3),(3,3),(3,3)]
    strides = [(1,1),(2,2),(1,1),(1,1)]
    dense_layers = 1
    dense_neurons = [128]
    dense_dropouts = [0]
    latent_dim = 3
    epochs = 1
    nb_start = 0 
    nb_end = 80    

    
    # create directories
    #path_1 = "./fig"
    #path_2 = "./imgs"
    #path_3 = "./hist"
    #path_4 = "./model"
    #if not os.path.exists(path_1):
    #    os.mkdir(path_1, 0755)
    #if not os.path.exists(path_2):
    #    os.mkdir(path_2, 0755)
    #if not os.path.exists(path_3):
    #    os.mkdir(path_3, 0755)
    #if not os.path.exists(path_4):
    #    os.mkdir(path_4, 0755)   
    #print("Completed directories creation or if already exist - then checked")
    

    # load data
    print("Loading data")
    # normalizing input image matrix
    X_train = x_train_raw.astype('float32') / np.amax(x_train_raw)
    X_test = x_test_raw.astype('float32') / np.amax(x_test_raw)
    X_pred = x_pred_raw.astype('float32') / np.amax(x_pred_raw)
    print("Shape of data loaded:", "train:", np.shape(X_train), "test:", np.shape(X_test))
    
    # reshape to 4d tensors
    image_size = X_train.shape[-2:]
    if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
        tensor_shape = (1,image_size[0],image_size[1])
    else:
        tensor_shape = (image_size[0],image_size[1],1)
    X_train = X_train.reshape((X_train.shape[0],) + tensor_shape)
    X_test = X_test.reshape((X_test.shape[0],) + tensor_shape)
    print("Reshaped data:", "train:", np.shape(X_train), "test:", np.shape(X_test))    
        
    # build autoencoder
    print("Building variational autoencoder")
    autoencoder = conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
                  filter_shapes,strides,dense_layers,dense_neurons,dense_dropouts,latent_dim) 

