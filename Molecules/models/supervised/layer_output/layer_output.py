#%matplotlib inline
from __future__ import print_function
import numpy as np
import gzip
from six.moves import cPickle
import sys    
from keras import backend as K  
from keras.utils import np_utils
#import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Deconvolution2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model
from keras import objectives
from keras.datasets import mnist
from keras.models import Sequential



class LayerOutput(object):
    def __init__(self, nb_classes=10, batch_size=50, nb_epoch=1, img_rows=28,
		 img_cols=28, nb_filters_1=1, nb_filters=64, pool_size=(1, 2),
 		 kernel_size=(5, 5), nb_start=0, nb_end=1):

        # TODO: Add exception handling for pool_size and kernel_size

        if(nb_classes < 0):
            raise Exception("Invalid input: nb_classes must be greater than 0!")
        if(batch_size < 0):
            raise Exception("Invalid input: batch_size must be greater than 0!")
        if(nb_epoch < 0):
            raise Exception("Invalid input: nb_epoch must be greater than 0!")
        if(img_rows < 0):
            raise Exception("Invalid input: img_rows must be greater than 0!")
        if(img_cols < 0):
            raise Exception("Invalid input: img_cols must be greater than 0!")
        if(nb_filters_1 < 0):
            raise Exception("Invalid input: nb_filters_1 must be greater than 0!")
        if(nb_filters < 0):
            raise Exception("Invalid input: nb_filters must be greater than 0!")
        if(nb_start < 0):
            raise Exception("Invalid input: nb_start must be greater than 0!")
        if(nb_start < nb_end):
            raise Exception("Invalid input: nb_end must be greater than nb_start!")

	    self.nb_classes = nb_classes
	    #batch size
	    self.batch_size = batch_size
	    # epoch numbers
	    self.nb_epoch = nb_epoch
	    # input image dimensions
	    self.img_rows, self.img_cols = img_rows, img_cols
	    # number of convolutional filters to use
	    self.nb_filters_1 = nb_filters_1
	    self.nb_filters = nb_filters
	    # size of pooling area for max pooling
	    self.pool_size = pool_size
	    # convolution kernel size
	    self.kernel_size = kernel_size

	    self.nb_start = nb_start
	    self.nb_end = nb_end
    
    def load(self, file_name='./input_data/mnist.pkl.gz'):

   	    # open compressed pickled data         
    	with gzip.open(file_name, 'rb') as f3:    
    	    (self.x_train, self.y_train), (self.x_test, self.y_test) = cPickle.load(f3)


	    print("Train set:", np.shape(self.x_train), np.shape(self.y_train)) 
	    print("Test set:", np.shape(self.x_test), np.shape(self.y_test)) 
	    #print("Pred set:", np.shape(x_pred), np.shape(y_pred))

        self.a = np.amax(self.x_train)
	    self.b = np.amax(self.x_test)
	    #c = np.amax(x_pred)
	    print(self.a, self.b)#, c

        # determining size for training, testing & prediction data    
	    #sep_1 = X.shape[0]*7/10 
	    #sep_2 = X.shape[0]*9/10    

	    #sep_1 = 40000 
	    #sep_2 = sep_1+10000

	    self.img_rows = x_train.shape[1]    
	    self.img_cols = x_train.shape[2]   

	    # assigning training set 
	    #x_train = X[:sep_1]
	    #y_train = Y[:sep_1]  
	
	    # assigning testing set 
	    #x_test = X[sep_1:sep_2]
	    #y_test = Y[sep_1:sep_2] 
	
	    # assigning prediction set 
	    #x_pred = X[sep_2: ]
	    #y_pred = Y[sep_2: ]   
	

    def prepare_data(self):
	    # input image dimensions
	    self.img_rows, self.img_cols, self.img_chns = self.img_rows, self.img_cols, 1
	
	    if K.image_dim_ordering() == 'th':
	        original_img_size = (self.img_chns, self.img_rows, self.img_cols)
	        input_shape = (1, self.img_rows, self.img_cols)
	    else:
	        original_img_size = (self.img_rows, self.img_cols, self.img_chns)
	        self.input_shape = (self.img_rows, self.img_cols, 1)
	
	    self.x_train = self.x_train.astype('float32') / self.a
	    self.x_test = self.x_test.astype('float32') / self.b
	    #x_pred = x_pred.astype('float32') / c
	    self.x_train = self.x_train.reshape((self.x_train.shape[0],) + original_img_size)
	    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
	    #x_pred = x_pred.reshape((x_pred.shape[0],) + original_img_size)

    	print('Shape of training set data:', np.shape(self.x_train), np.shape(self.y_train))    
	    print('Shape of testing set data:', np.shape(self.x_test), np.shape(self.y_test))    
	    #print('Shape of prediction set data:', np.shape(self.x_pred), np.shape(self.y_pred))

	    # convert class vectors to binary class matrices 
	    self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
	    self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)

    def build_model(self):
        # TODO: Add parameters for hyperparameter tuning
	    self.model = Sequential()
	    self.model.add(Convolution2D(self.nb_filters_1, self.kernel_size[0], self.kernel_size[1],
                        	         border_mode='valid',
                        	         input_shape=self.input_shape))
	    self.model.add(Activation('relu'))
	    self.model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1]))
	    self.model.add(Activation('relu'))
	    self.model.add(MaxPooling2D(pool_size=self.pool_size))
	    self.model.add(Dropout(0.25))
	    self.model.add(Flatten())
	    self.model.add(Dense(128))
	    self.model.add(Activation('relu'))
	    self.model.add(Dropout(0.5))
	    self.model.add(Dense(self.nb_classes))
	    self.model.add(Activation('softmax'))

    def summary(self):
        self.model.summary();

    def compile(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics);


    def build_directories(self):
        """
        Builds directory "./model_1/" to store model weights.
        """
        print("Building Directories...")

        path_1 = "./saved_models/model_1/"

        if not os.path.exists(path_1):
            os.mkdir(path_1, 0755)
    	print("Completed directories creation or if already exist - then checked")


# saving models: start & end
# fitting model with target dataset, saving model & loading for next fitting    
    def fit(self):
	    for i in range(self.nb_start, self.nb_end):    
    	    if i == 0:        
                print('Skipping - no previous saved file to load')
            else:        
                # TODO: This code has a bug here. Where does this file come from
        	    vae.load_weights('./saved_models/model_1/1FME-0_model_3D_%i' % i)       
        
    	    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
             	      verbose=1, validation_data=(x_test, y_test))
    	    score = model.evaluate(x_test, y_test, verbose=1)
    	    print(score)
    	    print('Completed %i epochs' % ((i+1)*nb_epoch)) 




    def save_layer(self):
	    X = self.x_train[0:50]
	    # with a Sequential model
	    get_layer_output = K.function([self.model.layers[0].input],
	                                  [self.model.layers[1].output])
	    layer_output = get_layer_output([X])[0]
	
	    print("Layer shape:", np.shape(layer_output))
	
	    # saving decoded arary to file    
	    np.savetxt('./saved_models/model_1/layer_output_conv1_th.out', np.reshape(layer_output, (len(layer_output), 
						            (layer_output.shape[2]*layer_output.shape[3]))), fmt='%f')      
	
	    #print layer_output

