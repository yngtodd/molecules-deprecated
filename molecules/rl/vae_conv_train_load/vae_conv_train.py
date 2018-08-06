# activate theano on gpu
import os;
#os.environ['THEANO_FLAGS'] = "device=gpu";
#import theano;
#theano.config.floatX = 'float32';

import numpy as np;
import sys
import gzip;
from six.moves import cPickle; 
from vae_conv import conv_variational_autoencoder;
from keras import backend as K;
#import pdb
#pdb.set_trace()   
# define parameters;
# no of trajectory files and frames in each file;
n_traj = 2; 
f_traj = 10000; 
# fraction of train, test and pred data separation; 
sep_train = 0.8;     
sep_test = 0.9;    
sep_pred = 1; 
# choice to flatten data: "0" for NO & "1" for YES;
choice = 0; 
# row and column dimension for each frame; 
row = 21;    
col = 21;   
# padding: use this incase diemsion mismatch for encoders; 
# pad_row and pad_col are row or colums to be added; 
pad_row = 1; 
pad_col = 1;      
# define parameters for variational autoencoder - convolutional;
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
# end define parameters; 

# opening file;
# open pickled file;
#with gzip.open('./aligned_fs-peptide_coor.pkl.gz', 'rb') as f3:
#    (X) = cPickle.load(f3) 
#x_raw = X;   
#print "dataset dimension:", np.shape(x_raw);
# open dat file;
path_data_array = "./../native-contact/data/cont-mat.array";
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
            print 'frames read:', (j_count/row);
            k_count = k_count + 10000*row;      
        j_count = j_count + 1;   
        if j_count == (row_num):
            break;
print('initial matrix array dimension:'), np.shape(array_f_int);
array_f = np.reshape(array_f_int, (samples, row, col));
print('final matrix array dimension:'), np.shape(array_f);
x_raw = array_f[0:];
print "dataset dimension:", np.shape(x_raw);

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
print "shape to load:", "train:", np.shape(x_train_raw), "test:", np.shape(x_test_raw), "prediction:", np.shape(x_pred_raw);

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
print "compledted directories creation or if already exist - then checked";
    
# load data;
print "loading data";      
# normalizing input image matrix;
X_train = x_train_raw.astype('float32') / np.amax(x_train_raw);
X_test = x_test_raw.astype('float32') / np.amax(x_test_raw);
X_pred = x_pred_raw.astype('float32') / np.amax(x_pred_raw);
print "shape of data loaded:", "train:", np.shape(X_train), "test:", np.shape(X_test);
    
# reshape to 4d tensors;
image_size = X_train.shape[-2:];
if K.image_dim_ordering() == 'th' or K.image_dim_ordering() == 'channels_first':
   tensor_shape = (1,image_size[0],image_size[1])
else:
   tensor_shape = (image_size[0],image_size[1],1)
X_train = X_train.reshape((X_train.shape[0],) + tensor_shape);
X_test = X_test.reshape((X_test.shape[0],) + tensor_shape);
print "reshaped data:", "train:", np.shape(X_train), "test:", np.shape(X_test);    
        
# build autoencoder;
print "building variational autoencoder";
# set up parameter;
feature_maps = feature_maps[0:conv_layers];
filter_shapes = filter_shapes[0:conv_layers];
strides = strides[0:conv_layers];
autoencoder = conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
           filter_shapes,strides,dense_layers,dense_neurons,dense_dropouts,latent_dim);

# UNIQUE
###############################################################
# train, save & load;
for i in range (nb_start, nb_end):    
    if i == 0:       
       print("skipping - no previous saved file to load")
    # load model; 
    else:        
       autoencoder.load("./model/model_%i" %i)
    # train model;
    autoencoder.train(X_train[0:],batch_size,epochs=epochs,
              validation_data=(X_test[0:], X_test[0:]), checkpoint=False,filepath="./savedweights.dat");  
    # save model;  
    autoencoder.save(filepath="./model/model_%i" %(i+1));
    # save loss over train & validation;
    np.savetxt('./hist/history.losses_%i' %(i+1), autoencoder.history.losses, delimiter=',');
    np.savetxt('./hist/history.val_losses_%i' %(i+1), autoencoder.history.val_losses, delimiter=',');
    print('completed %i epochs' % ((i+1)*epochs));   

# plot loss value;
# compile loss value; 
hist = np.zeros(((nb_end-nb_start),3));
for i in range ((nb_start+1), (nb_end+1)):
  hist_loss = np.loadtxt("./hist/history.losses_%i" %i);
  hist_val_loss = np.loadtxt("./hist/history.val_losses_%i" %i);
  tmp = np.array([i, hist_loss, hist_val_loss]);
  hist[i-1] = tmp;   
np.savetxt('./hist/hist_tot', hist, delimiter=' ');
# plot train & validation loss;
import matplotlib.pyplot as plt;
plt.switch_backend('agg');
plt.semilogx(hist[:, 0], hist[:, 1], color="blue", linewidth=1.5, linestyle="-", label="train_loss");
plt.semilogx(hist[:, 0], hist[:, 2], color="red",  linewidth=3.5, linestyle=":", label="test_loss");
plt.legend(loc='upper right');
# plt.ylim(np.amin(hist[:,1:3]/np.amax(hist[:, 1:3])),np.amax(hist[:,1:3]/np.amax(hist[:, 1:3])));
plt.savefig('./fig/history.png', dpi=600);
plt.clf();
    
###############################################################

