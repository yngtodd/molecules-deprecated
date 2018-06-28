# activate theano on gpu
from __future__ import print_function
import os;
os.environ['THEANO_FLAGS'] = "device=gpu";
import theano;
theano.config.floatX = 'float32';

import numpy as np;
import sys, os;
import gzip;
from six.moves import cPickle; 
from vae_conv import conv_variational_autoencoder;
from keras import backend as K;
from scipy.stats import norm;

# define parameters;
# no of trajectory files and frames in each file;
n_traj = 1; 
f_traj = 110*10000; 
# fraction of train, test and pred data separation; 
sep_train = 0.8;     
sep_test = 0.9;    
sep_pred = 1; 
# choice to flatten data: "0" for NO & "1" for YES;
choice = 0; 
# row and column dimension for each frame; 
row = 28;    
col =28;   
# padding: use this incase diemsion mismatch for encoders; 
# pad_row and pad_col are row or colums to be added; 
pad_row = 0; 
pad_col = 0;      
# define parameters for variational autoencoder - convolutional;
channels = 1;
batch_size = 1000;
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
# loading section; 
nb_select = 10;
load_step = 10;
load_start = nb_select;
load_end = nb_end+1;      
# number of digits for decoding;
n_dec = 10;  
# what image to pick for to decode;
pick = 400;   
# figure with 10x10 digits for generator images;
n_d = 10;    
n1 = 0;        
# end define parameters; 

# opening file;
# load data for labelling;
label = np.loadtxt("/home/odb/dl/keras/1FME-0/data/1FME-0_cont-mat.dat");
# open pickled file;
#with gzip.open('./aligned_fs-peptide_coor.pkl.gz', 'rb') as f3:
#    (X) = cPickle.load(f3) 
#x_raw = X;   
#print "dataset dimension:", np.shape(x_raw);
# open dat file;
path_data_array = "/home/odb/dl/keras/1FME-0/data/1FME-0_cont-mat.array";
# read dat type large file line by line to save in array
nf = n_traj*f_traj;      
q = row*col;    
j_count = 0;    
k_count = 0;   
samples = (nf);
row_num = (nf)*row;
column_num = (col);
array_f_int = np.zeros(shape=(row_num,column_num));  
with open(path_data_array) as infile:
    for line in infile:    
        array_f_string = line.split();  
        array_f_array = np.array(list(array_f_string), dtype='|S4');  
        array_f_float = array_f_array.astype(np.float);    
        array_f_int[j_count] = array_f_float;
        if j_count == k_count:  
            print('Frames read:', (j_count/row))
            k_count = k_count + 10000*row;      
        j_count = j_count + 1;   
        if j_count == (row_num):
            break;
print(('Initial matrix array dimension:'), np.shape(array_f_int))
array_f = np.reshape(array_f_int, (samples, row, col));
print(('Final matrix array dimension:'), np.shape(array_f))
x_raw = array_f[0:];
print("Dataset dimension:", np.shape(x_raw))

##########################################################################################################
##########################################################################################################
##########################################################################################################

# process of input data;
# padding;   
row_dim_array = row + pad_row;    
col_dim_array = col + pad_col;    
# reshape data according to the choice of flatteing;
if choice == 0:
    new_shape = (len(x_raw),row_dim_array,col_dim_array)  
if choice == 1:
    new_shape = (len(x_raw),row_dim_array*col_dim_array)
add_zero = np.zeros(new_shape,dtype = x_raw.dtype);        
if choice == 0:
    add_zero[0:x_raw.shape[0],0:x_raw.shape[1],0:x_raw.shape[2]] = x_raw 
if choice == 1:
    add_zero[0:x_raw.shape[0],0:x_raw.shape[1]] = x_raw 
x_raw = add_zero;
# determine size for training, testing & prediction data;    
sep_1 = int(x_raw.shape[0]*sep_train); 
sep_2 = int(x_raw.shape[0]*sep_test);    
sep_3 = int(x_raw.shape[0]*sep_pred);    
x_train_raw = x_raw[:sep_1];
x_test_raw = x_raw[sep_1:sep_2]; 
x_pred_raw = x_raw[sep_2:sep_3];
print("Shape to load:", "train:", np.shape(x_train_raw), "test:", np.shape(x_test_raw), "prediction:", np.shape(x_pred_raw))

# start variational autoencoder - convolutional; 
# create directories;
path_1 = "./fig"
path_2 = "./imgs"
path_3 = "./hist"
path_4 = "./model"
if not os.path.exists(path_1):
   os.mkdir(path_1, 0755);
if not os.path.exists(path_2):
   os.mkdir(path_2, 0755);
if not os.path.exists(path_3):
   os.mkdir(path_3, 0755);
if not os.path.exists(path_4):
   os.mkdir(path_4, 0755);   
print("Completed directories creation or if already exist - then checked")
    
# load data;
print("Loading data")    
# normalizing input image matrix;
X_train = x_train_raw.astype('float32') / np.amax(x_train_raw);
X_test = x_test_raw.astype('float32') / np.amax(x_test_raw);
X_pred = x_pred_raw.astype('float32') / np.amax(x_pred_raw);
print("Shape of data loaded:", "train:", np.shape(X_train), "test:", np.shape(X_test))
    
# reshape to 4d tensors;
image_size = X_train.shape[-2:];
if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
   tensor_shape = (1,image_size[0],image_size[1])
else:
   tensor_shape = (image_size[0],image_size[1],1)
X_train = X_train.reshape((X_train.shape[0],) + tensor_shape);
X_test = X_test.reshape((X_test.shape[0],) + tensor_shape);
print("Reshaped data:", "train:", np.shape(X_train), "test:", np.shape(X_test))    
        
# build autoencoder;
print("Building variational autoencoder")
# set up parameter;
feature_maps = feature_maps[0:conv_layers];
filter_shapes = filter_shapes[0:conv_layers];
strides = strides[0:conv_layers];
autoencoder = conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
           filter_shapes,strides,dense_layers,dense_neurons,dense_dropouts,latent_dim); 

# load data to analyze;
conv_full_train = X_train[0:];
conv_full_test = X_test[0:];
conv_full_pred = X_pred[0:];
label = label[:len(x_raw)];
y_train_0 = label[:sep_1,0];
y_train_2 = label[:sep_1,2];
y_test_0 = label[sep_1:sep_2,0];
y_test_2 = label[sep_1:sep_2,2];
y_pred_0 = label[sep_2:sep_3,0];
y_pred_2 = label[sep_2:sep_3,2];
# pixel size of decoded figure; 
row_dim = row_dim_array-pad_row;        
col_dim = col_dim_array-pad_col;
# for generator images (for latent space = nD);
z_axis = np.arange(latent_dim-2);

# print "plot starts";
for load in range(load_start, load_end, load_step):
    print("**********************************************loading", load)
    # loading model;
    autoencoder.load("./model/model_%i" %load);    
    ####################################################################
    print("Decode image for train data")
    # decode images;         
    decoded_imgs_full = autoencoder.decode(conv_full_train);
    # save decoded arary to file;    
    np.savetxt('./imgs/decoded_train_%i.out' %load, np.reshape(decoded_imgs_full[:, 0:row_dim, 0:col_dim, :], 
                (len(decoded_imgs_full), (row_dim*col_dim))), fmt='%f');   
    # plot decoded images;
    import matplotlib.pyplot as plt;
    plt.switch_backend('agg');
    plt.figure(figsize=(20, 4));
    for i in range (n_dec):
        # display original;
        ax = plt.subplot(2, n_dec, i + 1);                                                           
        plt.imshow(conv_full_train[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim));
        np.savetxt('./imgs/original_imgs_train_%i_%i.out' %(i,load), 
                    (conv_full_train[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim)));
        plt.colorbar(orientation='vertical');
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False);
    
        # display reconstruction;
        ax = plt.subplot(2, n_dec, i + 1 + n_dec);
        plt.imshow(decoded_imgs_full[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim));
        np.savetxt('./imgs/decoded_imgs_train_%i_%i.out' %(i,load), 
                    (decoded_imgs_full[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim)));
        plt.colorbar(orientation='vertical');
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False);
    plt.savefig('./fig/decoded_train_%i.png' %load, dpi=600);
    plt.clf();     
    ####################################################################
    print("Decode image for test data")
    # decode images;         
    decoded_imgs_full = autoencoder.decode(conv_full_test);
    # save decoded arary to file;    
    np.savetxt('./imgs/decoded_test_%i.out' %load, np.reshape(decoded_imgs_full[:, 0:row_dim, 0:col_dim, :], 
                (len(decoded_imgs_full), (row_dim*col_dim))), fmt='%f');   
    # plot decoded images;
    import matplotlib.pyplot as plt;
    plt.figure(figsize=(20, 4));
    for i in range (n_dec):
        # display original;
        ax = plt.subplot(2, n_dec, i + 1);                                                           
        plt.imshow(conv_full_train[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim));
        np.savetxt('./imgs/original_imgs_test_%i_%i.out' %(i,load), 
                    (conv_full_train[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim)));
        plt.colorbar(orientation='vertical');
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False);
    
        # display reconstruction;
        ax = plt.subplot(2, n_dec, i + 1 + n_dec);
        plt.imshow(decoded_imgs_full[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim));
        np.savetxt('./imgs/decoded_imgs_test_%i_%i.out' %(i,load), 
                    (decoded_imgs_full[i+pick, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim)));
        plt.colorbar(orientation='vertical');
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False);
    plt.savefig('./fig/decoded_test_%i.png' %load, dpi=600);
    plt.clf();     
	####################################################################    
    print("Encode image for train data")
    # encode images; 
    # project inputs on the latent space;
    x_pred_encoded = autoencoder.return_embeddings(conv_full_train);
    # save encoded array to file ;   
    np.savetxt('./imgs/encoded_train_%i.out' %load, x_pred_encoded, fmt='%f');
    # plot 1: 
    Dmax = y_train_2;
    [n,s] = np.histogram(Dmax, 11); 
    d = np.digitize(Dmax, s);
    #[n,s] = np.histogram(-np.log10(Dmax), 11); 
    #d = np.digitize(-np.log10(Dmax), s);
    from matplotlib import cm;
    import matplotlib as mpl;
    cmi = plt.get_cmap('jet');
    cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax));
    #cNorm = mpl.colors.Normalize(vmin=140, vmax=240);
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);
    import numpy as np;
    from mpl_toolkits.mplot3d import Axes3D;
    import matplotlib.pyplot as plt;
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    # scatter3D requires a 1D array for x, y, and z;
    # ravel() converts the 100x100 array into a 1x10000 array;
    p = ax.scatter3D(np.ravel(x_pred_encoded[:, 0]),
                np.ravel(x_pred_encoded[:, 1]),
                np.ravel(x_pred_encoded[:, 2]), 
                marker='o', c=scalarMap.to_rgba(Dmax));
    ax.set_xlim3d(np.amin(np.ravel(x_pred_encoded[:, 0])), np.amax(np.ravel(x_pred_encoded[:, 0])));
    ax.set_ylim3d(np.amin(np.ravel(x_pred_encoded[:, 1])), np.amax(np.ravel(x_pred_encoded[:, 1])));
    ax.set_zlim3d(np.amin(np.ravel(x_pred_encoded[:, 2])), np.amax(np.ravel(x_pred_encoded[:, 2])));
    ax.set_xlabel('VAE 0');
    ax.set_ylabel('VAE 1');
    ax.set_zlabel('VAE 2');
    scalarMap.set_array(Dmax);
    fig.colorbar(scalarMap);
    plt.savefig('./fig/encoded_train_%i.png' %load, dpi=600);
    plt.clf();
	####################################################################    
    print("Encode image for test data")
    # encode images; 
    # project inputs on the latent space;
    x_pred_encoded = autoencoder.return_embeddings(conv_full_test);
    # save encoded array to file ;   
    np.savetxt('./imgs/encoded_test_%i.out' %load, x_pred_encoded, fmt='%f');
    # plot 1: 
    Dmax = y_test_2;
    [n,s] = np.histogram(Dmax, 11); 
    d = np.digitize(Dmax, s);
    #[n,s] = np.histogram(-np.log10(Dmax), 11); 
    #d = np.digitize(-np.log10(Dmax), s);
    from matplotlib import cm;
    import matplotlib as mpl;
    cmi = plt.get_cmap('jet');
    cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax));
    #cNorm = mpl.colors.Normalize(vmin=140, vmax=240);
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);
    import numpy as np;
    from mpl_toolkits.mplot3d import Axes3D;
    import matplotlib.pyplot as plt;
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    # scatter3D requires a 1D array for x, y, and z;
    # ravel() converts the 100x100 array into a 1x10000 array;
    p = ax.scatter3D(np.ravel(x_pred_encoded[:, 0]),
                np.ravel(x_pred_encoded[:, 1]),
                np.ravel(x_pred_encoded[:, 2]), 
                marker='o', c=scalarMap.to_rgba(Dmax));
    ax.set_xlim3d(np.amin(np.ravel(x_pred_encoded[:, 0])), np.amax(np.ravel(x_pred_encoded[:, 0])));
    ax.set_ylim3d(np.amin(np.ravel(x_pred_encoded[:, 1])), np.amax(np.ravel(x_pred_encoded[:, 1])));
    ax.set_zlim3d(np.amin(np.ravel(x_pred_encoded[:, 2])), np.amax(np.ravel(x_pred_encoded[:, 2])));
    ax.set_xlabel('VAE 0');
    ax.set_ylabel('VAE 1');
    ax.set_zlabel('VAE 2');
    scalarMap.set_array(Dmax);
    fig.colorbar(scalarMap);
    plt.savefig('./fig/encoded_test_%i.png' %load, dpi=600);
    plt.clf();
	####################################################################    
    print("Generate image")
    # building generator; 
    # build a digit generator that can sample from the learned distribution;    
    # display a 2D manifold of the digits;   
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian; 
    figure = np.zeros((row_dim * n_d, col_dim * n_d));
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n_d));
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n_d));
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):    
            n1 = n1 + 1; 
            z_sample = np.append([xi, yi], [z_axis]);
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim);
            x_decoded = autoencoder.generate(z_sample);
            digit = x_decoded[0, 0:row_dim, 0:col_dim, :].reshape(row_dim, col_dim);    
    # saving generated array to file;     
    # np.savetxt('./generated/digit_%i.out' %n1, digit, fmt='%f');
            figure[i * row_dim: (i + 1) * row_dim,
                    j * col_dim: (j + 1) * col_dim] = digit;
    plt.figure(figsize=(10, 10));
    plt.imshow(figure);
    plt.savefig('./fig/generated_%i.png' %load, dpi=600);
    plt.clf();  







