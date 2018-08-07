from __future__ import print_function
import numpy as np
import gzip
from six.moves import cPickle
import sys
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.utils import plot_model

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from sklearn import svm
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier 

# for reproducibility;
np.random.seed(1337)

# Save history from log;        
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc"))
        self.val_acc.append(logs.get("val_acc"))


# Trains four layered convolutational network on dataset.
class Convnet(object):

    def __init__(self, file_name=None, batch_size = 128, nb_classes = 2, nb_epoch = 1, nb_start = 0, 
                 nb_end = 400, nb_filters = [32, 64, 128, 256], pool_size = (2, 1), kernel_size = (3, 1),
                 dense = 128, dropout = 0.25):

        # file_name = './data/aligned_1FME-0_coor_classification.pkl.gz'
        if file_name == None:
            raise Exception("Required Argument: file_name, must contain path of .pkl.gz file containing data ... terminating program")
        if (not os.path.exists(file_name)):
            raise Exception("Path " + str(file_name) + " does not exist!")
        if(batch_size < 0):
            raise Exception("Invalid input: batch_size must be greater than 0!")
        if(nb_classes < 0):
            raise Exception("Invalid input: nb_classes must be greater than or equal to 0!")
        if(nb_epoch < 0):
            raise Exception("Invalid input: nb_epoch must be greater than or equal to 0!")
        if(nb_start < 0):
            raise Exception("Invalid input: nb_start must be greater than or equal to 0!")
        if((nb_end < 0) or (nb_end < nb_start)):
            raise Exception("Invalid input: nb_end must be greater than or equal to 0 and less than nb_start!")
        if(dense < 0):
            raise Exception("Invalid input: dense must be greater than or equal to 0!")
        if((dropout < 0) or (dropout > 1)):
            raise Exception("Invalid input: dense must be greater than or equal to 0 and less than or equal to 1!")

        # define parameters
        self.file_name = file_name
        # minibatch size
        self.batch_size = batch_size
        # number of label class
        self.nb_classes = nb_classes
        # epoch numbers per model saving
        self.nb_epoch = nb_epoch
        # saving models: start & end
        self.nb_start = nb_start
        self.nb_end = nb_end # total epochs = (nb_end-nb_start)*nb_epoch
        # number of filters
        self.nb_filters = nb_filters
        # size of max pooling
        self.pool_size = pool_size
        # kernel size
        self.kernel_size = kernel_size
        # Dense layer
        self.dense = dense
        # dropout
        self.dropout = dropout

        self.X, self.Y, self.X_train, self.y_train, self.X_test, 
        self.y_test, self.X_pred, self.y_pred = load(filename)
        # input image dimensions
        self.img_rows = self.X.shape[1]
        self.img_cols = self.X.shape[2]
    
        self.history = LossHistory()
 
        # ordering in input data
        if K.image_dim_ordering() == 'th':
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.img_rows, self.img_cols)
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.img_rows, self.img_cols)
            self.X_pred = self.X_pred.reshape(self.X_pred.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], self.img_rows, self.img_cols, 1)
            self.X_test = self.X_test.reshape(self.X_test.shape[0], self.img_rows, self.img_cols, 1)
            self.X_pred = self.X_pred.reshape(self.X_pred.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

    def load(self):    
        """
        From model_plot code.
        X = np.zeros([1100000, 28, 28])
        Y = np.zeros([1100000, 1, 1])
        X = X[0:59000]
        Y = Y[0:59000]
        """
        # open compressed pickled data         
        with gzip.open(self.file_name, 'rb') as f:    
            (X, Y) = cPickle.load(f)

        # determining size for training, testing & prediction data    
        sep_1 = X.shape[0]*0.7 
        sep_2 = X.shape[0]*0.9

        # assigning training set 
        X_train = X[:int(sep_1)]
        y_train = Y[:int(sep_1)]  
        # assigning testing set
        X_test = X[int(sep_1):int(sep_2)]
        y_test = Y[int(sep_1):int(sep_2)] 
        # assigning prediction set 
        X_pred = X[int(sep_2):]
        y_pred = Y[int(sep_2):]
        print('Shape of training set data:', 'input', np.shape(X_train), 'label', np.shape(y_train))    
        print('Shape of testing set data:', 'input', np.shape(X_test), 'label', np.shape(y_test))  
        print('Shape of prediction set data:', 'input', np.shape(X_pred), 'label', np.shape(y_pred))

        return (X, Y, X_train, y_train, X_test, y_test, X_pred, y_pred)

    def prepare_data(self):
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_pred = self.X_pred.astype('float32')
        self.X_train /= np.amax(self.X_train)
        self.X_test /= np.amax(self.X_test)    
        self.X_pred /= np.amax(self.X_pred)
        print('Train samples', self.X_train.shape)
        print('Test samples', self.X_test.shape)
        print('Prediction samples', self.X_pred.shape)

        # convert class vectors to binary class matrices;
        self.Y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        self.Y_test = np_utils.to_categorical(self.y_test, self.nb_classes)     
        self.Y_pred = np_utils.to_categorical(self.y_pred, self.nb_classes)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(self.nb_filters[0], kernel_size=self.kernel_size, 
                       activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(self.nb_filters[1], self.kernel_size, activation='relu'))
        self.model.add(Conv2Dself.(nb_filters[1], self.kernel_size, activation='relu'))
        self.model.add(Conv2D(self.nb_filters[1], self.kernel_size, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Dropout(self.dropout))
        self.model.add(Flatten())
        self.model.add(Dense(self.dense, activation='relu'))
        self.model.add(Dropout(2*self.dropout))
        self.model.add(Dense(self.nb_classes, activation='softmax'))

    def summary(self):
        self.model.summary()

    def compile(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)   

    def build_directories(self):
        """
        Builds directories "./model/" and "./hist"
        to store model weights and model statistics (loss, acc) respectively.
        """
        print "Building Directories..."

        path_1 = "./saved_models/model/"
        path_2 = "./saved_models/hist/"

        if not os.path.exists(path_1):
            os.mkdir(path_1, 0755)
        if not os.path.exists(path_2):
            os.mkdir(path_2, 0755)

        print("Completed directories creation or if already exist - then checked")
    
    # fit, save & load
    def fit(self):

        for i in range (self.nb_start, self.nb_end):    
            if i == 0:        
                print('skipping - no previous saved file to load')
            else:        
                self.model.load_weights('./saved_models/model/protein_model_%i' % i)       
            self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                    verbose=2, validation_data=(self.X_test, self.Y_test), callbacks=[self.history])       
            self.model.save_weights('./saved_models/model/protein_model_%i' % (i+1))
            np.savetxt('./saved_models/hist/history.losses_%i' %(i+1), self.history.losses, delimiter=',')
            np.savetxt('./saved_models/hist/history.val_losses_%i' %(i+1), self.history.val_losses, delimiter=',')
            np.savetxt('./saved_models/hist/history.acc_%i' %(i+1), self.history.acc, delimiter=',')
            np.savetxt('./saved_models/hist/history.val_acc_%i' %(i+1), self.history.val_acc, delimiter=',')
            print('Completed %i epochs' % ((i+1)*self.nb_epoch))

        # plot loss value; 
        self.hist = np.zeros(((self.nb_end - self.nb_start),5))
        for i in range ((self.nb_start+1), (self.nb_end+1)):
            hist_loss = np.loadtxt("./saved_models/hist/history.losses_%i" %i)
            hist_val_loss = np.loadtxt("./saved_models/hist/history.val_losses_%i" %i)
            hist_acc = np.loadtxt("./saved_models/hist/history.acc_%i" %i)
            hist_val_acc = np.loadtxt("./saved_models/hist/history.val_acc_%i" %i)
            tmp = np.array([i, hist_loss, hist_val_loss, hist_acc, hist_val_acc])
            self.hist[i-1] = tmp

    def plot_loss(self):   
        plt.switch_backend('agg')
        plt.semilogx(self.hist[:, 0], self.hist[:, 1], color="blue", linewidth=1.5, linestyle="-", label="train_loss")
        plt.semilogx(self.hist[:, 0], self.hist[:, 2], color="red",  linewidth=3.5, linestyle=":", label="test_loss")
        plt.legend(loc='upper right')
        plt.savefig('./images/history_loss.png', dpi=600)
        plt.clf()

    def plot_acc(self):
        plt.switch_backend('agg')
        plt.semilogx(self.hist[:, 0], self.hist[:, 3], color="blue", linewidth=1.5, linestyle="-", label="train_acc")
        plt.semilogx(self.hist[:, 0], self.hist[:, 4], color="red",  linewidth=3.5, linestyle=":", label="test_acc")
        plt.legend(loc='upper right')
        plt.savefig('./images/history_acc.png', dpi=600)
        plt.clf()

    def evaluate(self):
        # evaluate model training on test data;    
        self.score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('Test accuracy:', self.score[1])

    def predict(self):    
        # predicting dataset using trained model;    
        print(self.model.predict_classes(self.X_test[ :20], batch_size=20))

    def print_f1_score(self):
        # load;
        load = 50
        self.model.load_weights('./saved_models/model/protein_model_%i' % load)  

        pred = self.model.predict_classes(self.X_pred[ :], batch_size=self.X_pred.shape[0])
        print("Macro f1_score:", f1_score(pred, self.y_pred, average='macro'))
        print("Micro f1_score:",f1_score(pred, self.y_pred, average='micro'))
        print("Weighted f1_score:",f1_score(pred, self.y_pred, average='weighted'))
        print("Binary f1_score:",f1_score(pred, self.y_pred, average='binary'))

    def build_svm_classifier(self, file_name= None):

        # file_name = '/Users/odb/Desktop/coor/SVM_dist-mat/1FME-0_svm.pkl'
        if file_name == None:
            raise Exception("Required Argument: file_name, must contain path of the .pkl file to dump svm model weights ... terminating program")

        # classifier with svm & fit
        clf_svm = svm.SVC()
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1]*self.X_train.shape[2]*self.X_train.shape[3]))
        clf_svm.fit(X_train, self.y_train)
        
        # open file, pickle dumping & closing file
        joblib.dump(clf_svm, file_name)

    def print_svm_f1_score(self, file_name=None):
        # file_name = '/Users/odb/Desktop/coor/SVM_dist-mat/1FME-0_svm.pkl'
        if file_name == None:
            raise Exception("Required Argument: file_name, must contain path of the .pkl file with svm model weights ... terminating program")
        # open pickled classifier;    
        clf_svm = joblib.load(file_name)   
        X_pred = np.reshape(self.X_pred, (self.X_pred.shape[0], self.X_pred.shape[1]*self.X_pred.shape[2]*self.X_pred.shape[3]))
        pred = clf_svm.predict(X_pred[0:])
  
        # determining accuracy
        k = len(pred)         
        l = 0    
        for i in range (0, k):
            if (pred[i]-y_pred[i]) != 0:        
                l = l + 1      
        a = k - l    
        t = a/float(k)
        print('Accuracy: %f' % t) 
        print("Macro f1 score:", f1_score(pred, self.y_pred, average='macro'))

    def build_rf_classifier(self, file_name):

        # file_name = '/Users/odb/Desktop/coor/RF_dist-mat/1FME-0_rf.pkl'
        if file_name == None:
            raise Exception("Required Argument: file_name, must contain path of the .pkl file to dump rf model weights ... terminating program")

        # Create the random forest object which will include all the parameters
        # for the fit
        forest = RandomForestClassifier(n_estimators = 100)

        # Fit the training data to the Survived labels and create the decision trees
        X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1]*self.X_train.shape[2]*self.X_train.shape[3]))
        forest.fit(X_train, self.y_train)            
       
        # open file, pickle dumping & closing file   
        joblib.dump(forest, file_name) 

    def print_rf_f1_score(self, file_name=None):
        # file_name = '/Users/odb/Desktop/coor/RF_dist-mat/1FME-0_rf.pkl'
        if file_name == None:
            raise Exception("Required Argument: file_name, must contain path of the .pkl file with rf model weights ... terminating program")

        # open pickled classifier    
        forest = joblib.load(file_name)    
        X_pred = np.reshape(self.X_pred, (self.X_pred.shape[0], self.X_pred.shape[1]*self.X_pred.shape[2]*self.X_pred.shape[3]))
        pred = forest.predict(self.X_pred[0:])        
  
        # determining accuracy
        k = len(pred)         
        l = 0    
        for i in range (0, k):
            if (pred[i]-y_pred[i]) != 0:        
                l = l + 1      
        a = k - l    
        t = a/float(k)
        print('Accuracy: %f' % t) 
        print("Macro f1 score:", f1_score(pred, self.y_pred, average='macro'))

    def plot_model(self):
        plot_model(self.model, show_shapes = 'True', show_layer_names = 'True', to_file='./images/convnet.png')
