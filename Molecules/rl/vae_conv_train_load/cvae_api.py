from __future__ import print_function
import os

# TODO: Add theano if statement check
# activate theano on gpu
#os.environ['THEANO_FLAGS'] = "device=gpu"
#import theano
#theano.config.floatX = 'float32'

import numpy as np
import sys
import gzip
from six.moves import cPickle 
from vae_conv import conv_variational_autoencoder
from keras import backend as K
from scipy.stats import norm

# For plotting purposes
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


class CVAE(object):

    def __init__(self, path="./", n_traj=2, f_traj=10000, sep_train=0.8, sep_test=0.9,
                 sep_pred=1, choice=0, row=21, col=21, pad_row=1, pad_col = 1,
                 channels=1, batch_size=1000, conv_layers=3, feature_maps=[128,128,128,128],
                 filter_shapes=[(3,3),(3,3),(3,3),(3,3)], strides=[(1,1),(2,2),(1,1),(1,1)],
                 dense_layers=1, dense_neurons=[128], dense_dropouts=[0], latent_dim=3,
                 epochs=1, nb_start=0, nb_end=50, nb_select=10, load_step=10,
                 n_dec=10, pick=400, n_d=10, n1=0):

        """
        Builds Keras CVAE model and provides API to use model.
        """

        # TODO: Add path variable to allow output to any directory. Default to "./".
        # TODO: Add exception handling for each input and add doc string.

         if not os.path.exists(path):
             raise Exception("Path: " + str(path) + " does not exist!")

        # Define parameters (For training and loading)
        self.path = path

        # No of trajectory files and frames in each file
        self.n_traj = n_traj 
        self.f_traj = f_traj
        # Fraction of train, test and pred data separation 
        self.sep_train = sep_train
        self.sep_test = sep_test    
        self.sep_pred = sep_pred
        # Choice to flatten data: "0" for NO & "1" for YES
        self.choice = choice
        # Row and column dimension for each frame
        self.row = row
        self.col =col   
        # Padding: use this incase diemsion mismatch for encoders 
        # pad_row and pad_col are row or colums to be added 
        # TODO: Consider handling automatically (if shaoe is odd then pad = 1, else pad = 0)
        self.pad_row = pad_row
        self.pad_col = pad_col
    
        # Define parameters for variational autoencoder - convolutional
        self.channels = channels
        self.batch_size = batch_size
        self.conv_layers = conv_layers
        self.feature_maps = feature_maps
        self.filter_shapes = filter_shapes
        self.strides = strides
        self.dense_layers = dense_layers
        self.dense_neurons = dense_neurons
        self.dense_dropouts = dense_dropouts
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.nb_start = nb_start 
        self.nb_end = nb_end

        # Define parameters for loading section
        self.nb_select = nb_select
        self.load_step = load_step
        self.load_start = self.nb_select
        self.load_end = self.nb_end+1    
        # Number of digits for decoding
        self.n_dec = n_dec
        # What image to pick for to decode
        self.pick = pick
        # Figure with 10x10 digits for generator images
        self.n_d = n_d
        self.n1 = n1

        # End define parameters

        self.build_directories()
        # Other class atributes
        

    def load_contact_matrix(self, dat_path, array_path):
        """
        dat_path : str
            Path of cont-mat.dat file.
            EX) dat_path="./../native-contact/data/cont-mat.dat"
        array_path : str
            Path of cont-mat.array file
            EX) array_path="./../native-contact/data/cont-mat.array"
        """
        # Load data for labelling
        self.label = np.loadtxt(dat_path)
        # Open dat file
        self.path_data_array = array_path

        self.read_data()
        self.process_input_data()

        print("Data was successfully read, loaded, and processed")

        # Not implemented
        # Open pickled file
        #with gzip.open('./aligned_fs-peptide_coor.pkl.gz', 'rb') as f3:
        #    (X) = cPickle.load(f3) 
        #x_raw = X
        #print("Dataset dimension:", np.shape(x_raw))

    def read_data(self):
        """
        Internal method.
        """
        # Read dat type large file line by line to save in array
        nf = self.n_traj * self.f_traj
        q = self.row * self.col
        j_count = 0
        k_count = 0   
        samples = (nf)
        row_num = (nf) * self.row
        column_num = (self.col)
        array_f_int = np.zeros(shape=(row_num, column_num))  
        with open(self.path_data_array) as infile:
            for line in infile:    
                array_f_string = line.split()  
                array_f_array = np.array(list(array_f_string), dtype='|S4')  
                array_f_float = array_f_array.astype(np.float)
                array_f_int[j_count] = array_f_float
                if j_count == k_count:  
                    print("Frames read:", (j_count/self.row))
                    k_count = k_count + 10000 * self.row      
                j_count = j_count + 1
                if j_count == (row_num):
                    break
        print("Initial matrix array dimension:", np.shape(array_f_int))
        array_f = np.reshape(array_f_int, (samples, self.row, self.col))
        print("Final matrix array dimension:", np.shape(array_f))
        x_raw = array_f[0:]
        print("Dataset dimension:", np.shape(x_raw))

        self.x_raw = x_raw

    def process_input_data(self):
        """
        Internal method.
        """
        # Process of input data

        # TODO: Add if expression as instructed in __init__ to automate padding.
        # Padding
        row_dim_array = self.row + self.pad_row    
        col_dim_array = self.col + self.pad_col

        # Reshape data according to the choice of flatteing
        if choice == 0:
            new_shape = (len(self.x_raw), row_dim_array, col_dim_array)  
        if choice == 1:
            new_shape = (len(self.x_raw), row_dim_array * col_dim_array)

        add_zero = np.zeros(new_shape, dtype = self.x_raw.dtype)       

        if choice == 0:
            add_zero[0:self.x_raw.shape[0], 0:self.x_raw.shape[1], 0:self.x_raw.shape[2]] = self.x_raw 
        if choice == 1:
            add_zero[0:self.x_raw.shape[0], 0:self.x_raw.shape[1]] = self.x_raw 
        self.x_raw = add_zero

        # Determine size for training, testing & prediction data
        sep_1 = int(self.x_raw.shape[0] * sep_train)
        sep_2 = int(self.x_raw.shape[0] * sep_test)    
        sep_3 = int(self.x_raw.shape[0] * sep_pred)    
        x_train_raw = self.x_raw[:self.sep_1]
        x_test_raw = self.x_raw[sep_1:sep_2] 
        x_pred_raw = self.x_raw[sep_2:sep_3]
        print("Shape to load:", "Train:", np.shape(x_train_raw), "Test:", np.shape(x_test_raw), "Prediction:", np.shape(x_pred_raw))

        # load data
        print("Loading data")
        
        # Normalizing input image matrix
        X_train = x_train_raw.astype('float32') / np.amax(x_train_raw)
        X_test = x_test_raw.astype('float32') / np.amax(x_test_raw)
        X_pred = x_pred_raw.astype('float32') / np.amax(x_pred_raw)
        print("Shape of data loaded:", "Train:", np.shape(X_train), "Test:", np.shape(X_test), "Prediction:", np.shape(X_pred))

        # TODO: Reshape prediction shape X_pred ? Ask Sindhu
        # Reshape to 4d tensors
        image_size = X_train.shape[-2:]
        if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
           tensor_shape = (1,image_size[0],image_size[1])
        else:
           tensor_shape = (image_size[0],image_size[1],1)
        # OLD
        #X_train = X_train.reshape((X_train.shape[0],) + tensor_shape)
        #X_test = X_test.reshape((X_test.shape[0],) + tensor_shape)
        #print("Reshaped data:", "Train:", np.shape(X_train), "Test:", np.shape(X_test))
        #
        #NEW A. Brace 7/16/2018, comment: Ask Sindhu why he didn't reshape X_pred.
        X_train = X_train.reshape((X_train.shape[0],) + tensor_shape)
        X_test = X_test.reshape((X_test.shape[0],) + tensor_shape)
        X_pred = X_pred.reshape((X_pred.shape[0],) + tensor_shape)
        print("Reshaped data:", "Train:", np.shape(X_train), "Test:", np.shape(X_test), "Prediction:", np.shape(X_pred))

        self.X_train = X_train
        self.X_test = X_test
        self.X_pred = X_pred
        self.sep_1 = sep_1
        self.sep_2 = sep_2
        self.sep_3 = sep_3
        self.image_size = image_size

    def build_directories(self):
        """
        Internal method.
        """
        # Create directories
        self.path = self.path + "/cvae"
        path_1 = self.path + "/fig"
        path_2 = self.path + "/imgs"
        path_3 = self.path + "/hist"
        path_4 = self.path + "/model"

        if not os.path.exists(self.path):
            os.mkdir(self.path, 0755)
        if not os.path.exists(path_1):
           os.mkdir(path_1, 0755)
        if not os.path.exists(path_2):
           os.mkdir(path_2, 0755)
        if not os.path.exists(path_3):
           os.mkdir(path_3, 0755)
        if not os.path.exists(path_4):
           os.mkdir(path_4, 0755)
        print("Completed directories creation or if already exist - then checked")

    def compile(self):
        """
        Builds autoencoder.
        """
        print("Building convolutional variational autoencoder")

        # set up parameter
        self.feature_maps = self.feature_maps[0:self.conv_layers]
        self.filter_shapes = self.filter_shapes[0:self.conv_layers]
        self.strides = self.strides[0:self.conv_layers]
        self.autoencoder = conv_variational_autoencoder(self.image_size,
                                                        self.channels,
                                                        self.conv_layers,
                                                        self.feature_maps,
                                                        self.filter_shapes,
                                                        self.strides,
                                                        self.dense_layers,
                                                        self.dense_neurons,
                                                        self.dense_dropouts,
                                                        self.latent_dim)

    def train(self):
        """
        Train, save & load.
        """
        for i in range (self.nb_start, self.nb_end):

            # Load model
            if i == 0:       
               print("Skipping - no previous saved file to load")
            else:
               self.autoencoder.load(self.path + "/model/model_%i" %i)

            # Train model
            self.autoencoder.train(self.X_train[0:], self.batch_size, epochs=self.epochs,
                              validation_data=(self.X_test[0:], self.X_test[0:]),
                              checkpoint=False, filepath=self.path + "/savedweights.dat")
            # Save model
            self.autoencoder.save(filepath=self.path + "/model/model_%i" %(i+1))

            # Save loss over train & validation
            np.savetxt(self.path + '/hist/history.losses_%i' %(i+1), self.autoencoder.history.losses, delimiter=',')
            np.savetxt(self.path + '/hist/history.val_losses_%i' %(i+1), self.autoencoder.history.val_losses, delimiter=',')
            print("Completed %i epochs" % ((i+1) * self.epochs))

    def history(self):
        """
        Call method after training.

        Compile loss value.
        Saves history in "cvae/hist/hist_tot".

        Plot loss value.
        Plot train & validation loss.
        Saves figures in "cvae/fig/history.png".
        """

        # TODO: Add exception if "./hist/history.losses_%i" does not exist (inside for loop).

        hist = np.zeros(((self.nb_end - self.nb_start), 3))
        for i in range ((self.nb_start + 1), (self.nb_end + 1)):
            hist_loss = np.loadtxt(self.path + "/hist/history.losses_%i" %i)
            hist_val_loss = np.loadtxt(self.path + "/hist/history.val_losses_%i" %i)
            tmp = np.array([i, hist_loss, hist_val_loss])
            hist[i-1] = tmp
        np.savetxt(self.path + '/hist/hist_tot', hist, delimiter=' ')

        plt.switch_backend('agg')
        plt.semilogx(hist[:, 0], hist[:, 1], color="blue", linewidth=1.5, linestyle="-", label="train_loss")
        plt.semilogx(hist[:, 0], hist[:, 2], color="red",  linewidth=3.5, linestyle=":", label="test_loss")
        plt.legend(loc='upper right')
        # plt.ylim(np.amin(hist[:,1:3]/np.amax(hist[:, 1:3])),np.amax(hist[:,1:3]/np.amax(hist[:, 1:3])))
        plt.savefig(self.path + '/fig/history.png', dpi=600)
        plt.clf()

    def analyze(self, data_set, model_selection):
        """
        data_set : str
            'train', 'test', or 'pred'
        model_selection : int
            select file number
            EX) Want "/model/model_2", then select model_selection = 2

        Generates plots (.png files) stored in "/cvae/fig/".
        Save evaluated data in '/imgs/decoded_train_%i.out'.
        """
        # TODO: Add exception handling for input data

        #Load, encode, decode, save both encode and decode

        # 1) Select data set
        # Load data to analyze
        data = np.array([])
        if data_set == 'train':
            data = self.X_train[0:]
        else if data_set == 'test':
            data = self.X_test[0:]
        else if data_set == 'pred':
            data = self.X_pred[0:]
                 
        print("Loading", model_selection)
        # TODO: Add exception handling checking that the file exists
        # 2) Loading model
        self.load_weights(self.path + "/model/model_%i" %model_selection)   
       
        print("Decode image for train data")
        # 3) Decode images        
        decoded_imgs_full = self.decode(data)
        # 4) Save decoded array to file   
        np.savetxt(self.path + '/imgs/decoded_train_%i.out' %model_selection,
                   np.reshape(decoded_imgs_full[:, 0:self.row, 0:self.col, :], 
                   (len(decoded_imgs_full), (self.row *self.col))), fmt='%f')

         print("Encode image for train data")
         # Encode images
         # 5) Project inputs on the latent space
         x_pred_encoded = self.encode(data)
         # 6) Save encoded array to file 
         np.savetxt(self.path + '/imgs/encoded_train_%i.out' %model_selection, x_pred_encoded, fmt='%f')

    def load_weights(self, weight_path):
        """
        weight_path : str      
        """
        self.autoencoder.load(weight_path)   

    def decode(self, data):
        return self.autoencoder.decode(data)

    def encode(self, data):
        return self.autoencoder.return_embeddings(data)

    def decode_pred(self):
        return self.autoencoder.decode(X_pred)

    def encode_pred(self):
        return self.autoencoder.return_embeddings(data)

    def analyze_all(self):
        """
        Generates plots (.png files) stored in "/cvae/fig/".
        """
        # TODO: Break up functions into several smaller plotting funcions
        #Load, encode, decode, save both encode and decode

        # 1) Select data set (just one line)
        # Load data to analyze
        conv_full_train = self.X_train[0:]
        conv_full_test = self.X_test[0:]
        conv_full_pred = self.X_pred[0:]
        label = label[:len(self.x_raw)]
        y_train_0 = label[:self.sep_1, 0]
        y_train_2 = label[:self.sep_1, 2]
        y_test_0 = label[self.sep_1:self.sep_2, 0]
        y_test_2 = label[self.sep_1:self.sep_2, 2]
        y_pred_0 = label[self.sep_2:self.sep_3, 0]
        y_pred_2 = label[self.sep_2:self.sep_3, 2]
      
        # For generator images (for latent space = nD)
        z_axis = np.arange(self.latent_dim - 2)

        for load in range(self.load_start, self.load_end, self.load_step):
            # need
            #########################
            print("Loading", load)
            # TODO: Add exception handling checking that the file exists
            # 2) Loading model
            self.autoencoder.load(self.path + "/model/model_%i" %load)    
       
            print("Decode image for train data")
            # 3) Decode images        
            decoded_imgs_full = self.autoencoder.decode(conv_full_train)
            # Save decoded array to file   
            np.savetxt(self.path + '/imgs/decoded_train_%i.out' %load,
                       np.reshape(decoded_imgs_full[:, 0:self.row, 0:self.col, :], 
                       (len(decoded_imgs_full), (self.row *self.col))), fmt='%f')  
            ###########################
            # Plot decoded images
            plt.switch_backend('agg')
            plt.figure(figsize=(20, 4))

            for i in range (self.n_dec):

                # Display original
                ax = plt.subplot(2, self.n_dec, i + 1)                                                         
                plt.imshow(conv_full_train[i + self.pick, 0:self.row , 0:self.col, :].reshape(self.row, self.col))
                np.savetxt(self.path + '/imgs/original_imgs_train_%i_%i.out' %(i, load), 
                           (conv_full_train[i + self.pick, 0:self.row , 0:self.col, :].reshape(self.row, self.col)))
                plt.colorbar(orientation='vertical')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
                # Display reconstruction
                ax = plt.subplot(2, self.n_dec, i + 1 + self.n_dec)
                plt.imshow(decoded_imgs_full[i + self.pick, 0:self.row, 0:self.col, :].reshape(self.row, self.col))
                np.savetxt(self.path + '/imgs/decoded_imgs_train_%i_%i.out' %(i, load), 
                           (decoded_imgs_full[i + self.pick, 0:self.row, 0:self.col, :].reshape(self.row, self.col)))
                plt.colorbar(orientation='vertical')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.savefig(self.path + '/fig/decoded_train_%i.png' %load, dpi=600)
            plt.clf()
          
            print("Decode image for test data")
            # Decode images      
            decoded_imgs_full = self.autoencoder.decode(conv_full_test)
            # Save decoded array to file   
            np.savetxt(self.path + '/imgs/decoded_test_%i.out' %load,
                       np.reshape(decoded_imgs_full[:, 0:self.row, 0:self.col, :], 
                       (len(decoded_imgs_full), (self.row * self.col))), fmt='%f')  

            # Plot decoded images
            plt.figure(figsize=(20, 4))

            for i in range (self.n_dec):
                # Display original
                ax = plt.subplot(2, self.n_dec, i + 1)                                                         
                plt.imshow(conv_full_train[i + self.pick, 0:self.row, 0:self.col, :].reshape(self.row, self.col))
                np.savetxt(self.path + '/imgs/original_imgs_test_%i_%i.out' %(i,load), 
                            (conv_full_train[i + self.pick, 0:self.row, 0:self.col, :].reshape(self.row, self.col)))
                plt.colorbar(orientation='vertical')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
                # Display reconstruction
                ax = plt.subplot(2, self.n_dec, i + 1 + self.n_dec)
                plt.imshow(decoded_imgs_full[i + self.pick, 0:self.row, 0:self.col, :].reshape(self.row, self.col))
                np.savetxt(self.path + '/imgs/decoded_imgs_test_%i_%i.out' %(i, load), 
                            (decoded_imgs_full[i+self.pick, 0:self.row, 0:self.col, :].reshape(self.row, self.col)))
                plt.colorbar(orientation='vertical')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.savefig(self.path + '/fig/decoded_test_%i.png' %load, dpi=600)
            plt.clf()  
	           
            print("Encode image for train data")
            # Encode images
            # 4) Project inputs on the latent space
            x_pred_encoded = self.autoencoder.return_embeddings(conv_full_train)
            # 5) Save encoded array to file 
            np.savetxt(self.path + '/imgs/encoded_train_%i.out' %load, x_pred_encoded, fmt='%f')

            # PLOT in another subclass


            # Plot 1: 
            Dmax = y_train_2
            [n,s] = np.histogram(Dmax, 11)
            d = np.digitize(Dmax, s)
            #[n,s] = np.histogram(-np.log10(Dmax), 11)
            #d = np.digitize(-np.log10(Dmax), s)
            cmi = plt.get_cmap('jet')
            cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax))
            #cNorm = mpl.colors.Normalize(vmin=140, vmax=240)
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # scatter3D requires a 1D array for x, y, and z
            # ravel() converts the 100x100 array into a 1x10000 array
            p = ax.scatter3D(np.ravel(x_pred_encoded[:, 0]),
                             np.ravel(x_pred_encoded[:, 1]),
                             np.ravel(x_pred_encoded[:, 2]), 
                             marker='o', c=scalarMap.to_rgba(Dmax))
            ax.set_xlim3d(np.amin(np.ravel(x_pred_encoded[:, 0])), np.amax(np.ravel(x_pred_encoded[:, 0])))
            ax.set_ylim3d(np.amin(np.ravel(x_pred_encoded[:, 1])), np.amax(np.ravel(x_pred_encoded[:, 1])))
            ax.set_zlim3d(np.amin(np.ravel(x_pred_encoded[:, 2])), np.amax(np.ravel(x_pred_encoded[:, 2])))
            ax.set_xlabel('VAE 0')
            ax.set_ylabel('VAE 1')
            ax.set_zlabel('VAE 2')
            scalarMap.set_array(Dmax)
            fig.colorbar(scalarMap)
            plt.savefig(self.path + '/fig/encoded_train_%i.png' %load, dpi=600)
            plt.clf()

            print("Encode image for test data")
            # Encode images 
            # Project inputs on the latent space
            x_pred_encoded = self.autoencoder.return_embeddings(conv_full_test)
            # Save encoded array to file
            np.savetxt(self.path + '/imgs/encoded_test_%i.out' %load, x_pred_encoded, fmt='%f')

            # Plot 2: 
            Dmax = y_test_2
            [n,s] = np.histogram(Dmax, 11)
            d = np.digitize(Dmax, s)
            #[n,s] = np.histogram(-np.log10(Dmax), 11) 
            #d = np.digitize(-np.log10(Dmax), s)
            cmi = plt.get_cmap('jet')
            cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax))
            #cNorm = mpl.colors.Normalize(vmin=140, vmax=240)
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # scatter3D requires a 1D array for x, y, and z
            # ravel() converts the 100x100 array into a 1x10000 array
            p = ax.scatter3D(np.ravel(x_pred_encoded[:, 0]),
                             np.ravel(x_pred_encoded[:, 1]),
                             np.ravel(x_pred_encoded[:, 2]), 
                             marker='o', c=scalarMap.to_rgba(Dmax))
            ax.set_xlim3d(np.amin(np.ravel(x_pred_encoded[:, 0])), np.amax(np.ravel(x_pred_encoded[:, 0])))
            ax.set_ylim3d(np.amin(np.ravel(x_pred_encoded[:, 1])), np.amax(np.ravel(x_pred_encoded[:, 1])))
            ax.set_zlim3d(np.amin(np.ravel(x_pred_encoded[:, 2])), np.amax(np.ravel(x_pred_encoded[:, 2])))
            ax.set_xlabel('VAE 0')
            ax.set_ylabel('VAE 1')
            ax.set_zlabel('VAE 2')
            scalarMap.set_array(Dmax)
            fig.colorbar(scalarMap)
            plt.savefig(self.path + '/fig/encoded_test_%i.png' %load, dpi=600)
            plt.clf()
	          
            print("Generate image")
            # Building generator
            # Build a digit generator that can sample from the learned distribution  
            # Display a 2D manifold of the digits
            # Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
            #   to produce values of the latent variables z, since the prior of the latent space is Gaussian
            figure = np.zeros((self.row  * self.n_d, self.col * self.n_d))
            grid_x = norm.ppf(np.linspace(0.05, 0.95, self.n_d))
            grid_y = norm.ppf(np.linspace(0.05, 0.95, self.n_d))
    
            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):    
                    self.n1 = self.n1 + 1
                    z_sample = np.append([xi, yi], [z_axis])
                    z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, self.latent_dim)
                    x_decoded = self.autoencoder.generate(z_sample)
                    digit = x_decoded[0, 0:self.row, 0:self.col, :].reshape(self.row , self.col)    
                    # Saving generated array to file   
                    # np.savetxt('./generated/digit_%i.out' %self.n1, digit, fmt='%f')
                    figure[i * self.row: (i + 1) * self.row,
                           j * self.col: (j + 1) * self.col] = digit
            plt.figure(figsize=(10, 10))
            plt.imshow(figure)
            plt.savefig(self.path + '/fig/generated_%i.png' %load, dpi=600)
            plt.clf()
