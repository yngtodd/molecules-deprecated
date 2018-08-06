#%pylab inline
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#self.encoded_path = "./encoded_train_50.out"
#self.data_path = "./pp_fs-peptide.npy"
class Plot(object):

    def _init__(self, encoded_path=None, data_path=None):
        """
        encoded_path : string
            - path of the encoded .out file. Should be located in ./output_data
        data_path : string
            - path of the data .npy. Should be located in ./output_data
        """
        if(encoded_path == None or data_path == None):
            raise ValueError("Must input encoded_path and data_path as parameters.")
        if (not os.path.exists(encoded_path)):
            raise Exception("Path " + str(encoded_path) + " does not exist!")
        if (not os.path.exists(data_path)):
            raise Exception("Path " + str(data_path) + " does not exist!")
       
        self.encoded_path = encoded_path
        self.data_path = data_path


   
    def encode_images(self):
   	    print("Encode image for train data")
    	# encode images
    	# project inputs on the latent space
    	self.x_pred_encoded = np.loadtxt(self.encoded_path)
    	#x_pred_encoded = x_pred_encoded[10000:110000]
    	data_input = np.load(self.data_path)
    	#data_input = data_input[10000:110000]
    	label = data_input.sum(axis=1)
    	label = np.reshape(label, (len(label), 1))
    	sep_train = 0.8
    	sep_test = 0.9    
    	sep_pred = 1
    	sep_1 = int(data_input.shape[0]*sep_train) 
    	sep_2 = int(data_input.shape[0]*sep_test)    
    	sep_3 = int(data_input.shape[0]*sep_pred) 
    	y_train_0 = label[:sep_1,0]
    	self.y_train_2 = label[:sep_1,0]
    	y_test_0 = label[sep_1:sep_2,0]
    	y_test_2 = label[sep_1:sep_2,0]
    	y_pred_0 = label[sep_2:sep_3,0]
    	y_pred_2 = label[sep_2:sep_3,0]

    def plot(self):
    	# plot 1: 
    	Dmax = self.y_train_2
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
    	p = ax.scatter3D(np.ravel(self.x_pred_encoded[:, 0]),
                         np.ravel(self.x_pred_encoded[:, 1]),
                         np.ravel(self.x_pred_encoded[:, 2]), 
                     	 marker='.', c=scalarMap.to_rgba(Dmax))
    	ax.set_xlim3d(np.amin(np.ravel(self.x_pred_encoded[:, 0])), np.amax(np.ravel(self.x_pred_encoded[:, 0])))
    	ax.set_ylim3d(np.amin(np.ravel(self.x_pred_encoded[:, 1])), np.amax(np.ravel(self.x_pred_encoded[:, 1])))
    	ax.set_zlim3d(np.amin(np.ravel(self.x_pred_encoded[:, 2])), np.amax(np.ravel(self.x_pred_encoded[:, 2])))
    	ax.set_xlabel('VAE 0')
    	ax.set_ylabel('VAE 1')
    	ax.set_zlabel('VAE 2')
    	scalarMap.set_array(Dmax)
    	fig.colorbar(scalarMap)
    	plt.savefig('./images/encoded_train.png', dpi=600)
    	plt.show()
    	#plt.clf()

