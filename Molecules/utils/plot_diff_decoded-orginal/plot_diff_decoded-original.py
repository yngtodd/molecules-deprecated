from __future__ import print_function
import numpy as np
import gzip
from six.moves import cPickle
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt



class Decoded_plot(object):

    def __init__(self, residue_diff=28, coor=3, batch_size=1000, start=0, sep_train=0.8, 
		 sep_test=0.9, sep_pred=1, chose=1):
    
        if(residue_diff < 0):
            raise Exception("Invalid input: residue_diff must be greater than 0!")
        if(coor < 0):
            raise Exception("Invalid input: coor must be greater than 0!")
        if(batch_size < 0):
            raise Exception("Invalid input: batch_size must be greater than 0!")
        if(start < 0):
            raise Exception("Invalid input: start must be greater than 0!")
        if(sep_train < 0 or sep_train > 1):
            raise Exception("Invalid input: sep_train must be between 0 and 1!")
        if(sep_pred < 0 or sep_pred > 1):
            raise Exception("Invalid input: sep_pred must be between 0 and 1!")
        if(chose != 0 and chose != 1 and chose != 2):
            raise Exception("Invalid input: chose must be either 0, 1, or 2!")

	    # define parameters
	    self.residue_diff = residue_diff
	    self.coor = coor
	    self.batch_size = batch_size
	    # how data was separated into train, test or prediction set while running running code 
	    self.start = start
	    self.sep_train = sep_train 
	    self.sep_test = sep_test
	    self.sep_pred = sep_pred
	    # which dataset was loaded? chose '0' = train, '1' = test, '2' = pred
	    self.chose = chose

    # original_path='./aligned_1FME-0_coor.pkl.gz'
    # decoded_path='./decoded_test_80.out'
    def load(self, original_path=None, decoded_path=None):
        """
        original_path : string
            - path of the coor.pkl.gz file. Should be located in ./output_data or ./input_data
        decoded_path : string
            - path of the decoded .out. Should be located in ./output_data or ./input_data
        """
        if(original_path == None or decoded_path == None):
            raise ValueError("Must input original_path and decoded_path as parameters.")
        if (not os.path.exists(original_path)):
            raise Exception("Path " + str(original_path) + " does not exist!")
        if (not os.path.exists(decoded_path)):
            raise Exception("Path " + str(decoded_path) + " does not exist!")

	    # load original data
	    # open pickled file
    	print("Loading original data:")
	    with gzip.open(original_path, 'rb') as f3:
	        (X) = cPickle.load(f3)     
	    self.CA_xyz_tot = X[0:500000]
	    print("Shape of original data", np.shape(self.CA_xyz_tot))

	    # load decoded data
	    print("Loading decoded data:")
	    self.CA_xyz_tot_decoded = np.loadtxt(decoded_path)
	    print("Shape of loaded data", np.shape(self.CA_xyz_tot_decoded))
	    # reshaping loaded decoded data
	    self.CA_xyz_tot_decoded_nf = np.reshape(self.CA_xyz_tot_decoded, 
					      (self.CA_xyz_tot_decoded.shape[0], self.residue_diff, (self.coor)))
	    print("Shape of decoded reshaped data:", np.shape(self.CA_xyz_tot_decoded_nf))


    def calc_orginal_decoded_value(self):
	    # calculate original value of decoded data
	    if self.chose == 0:
	        # for training data
	        xyz_max = np.amax(self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.start):
					      int(self.CA_xyz_tot.shape[0]*self.sep_train), :, :])
	    if self.chose == 1:
	        # for testing data
	        xyz_max = np.amax(self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.sep_train):
					      int(self.CA_xyz_tot.shape[0]*self.sep_test), :, :])
	    if self.chose == 2:
	        # for prediction data
	        xyz_max = np.amax(self.CA_xyz_tot[self.CA_xyz_tot.shape[0]*self.sep_test:
					      self.CA_xyz_tot.shape[0]*self.sep_pred, :, 0:3])
	    print("Maximum value in any direction in concerned data set", xyz_max)  
	    print("Change to original value from normalized loaded data")
	    self.CA_xyz_tot_decoded_nf[:, :, :] = self.CA_xyz_tot_decoded_nf[:, :, :]*xyz_max   


    def calc_diff_value(self):
	    # calculate difference between original & decoded
	    # create zero value arrays
	    diff = np.zeros((self.CA_xyz_tot_decoded.shape[0],residue_diff,coor))
	    diff_pc = np.zeros((self.CA_xyz_tot_decoded.shape[0],residue_diff,coor)) 
	    # calculate difference - both in nominal & in percentage
	    if self.chose == 0:
	        # for training data 
	        diff[:, :, :] = self.CA_xyz_tot_decoded_nf[:, :, :]-self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.start)
	                                                              :int(self.CA_xyz_tot.shape[0]*self.sep_train), :, :]
	        diff_pc = np.absolute(diff[:, :, :]/self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.start)
	                                                              :int(CA_xyz_tot.shape[0]*sep_train), :, :]*100)
	    if self.chose == 1:
	        # for testing data
	        diff[:, :, :] = self.CA_xyz_tot_decoded_nf[:, :, :]-self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.sep_train)
	                                                              :int(self.CA_xyz_tot.shape[0]*self.sep_test), :, :]
	        diff_pc = np.absolute(diff[:, :, :]/self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.sep_train)
	                                                              :int(self.CA_xyz_tot.shape[0]*self.sep_test), :, :]*100)
	    if self.chose == 2:
	        # for prediction data
	        diff[:, :, 0:3] = self.CA_xyz_tot_decoded_nf[:, :, 0:3]-self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.sep_test)
	                                                              :int(self.CA_xyz_tot.shape[0]*self.sep_pred), :, 0:3]
	        diff_pc = np.absolute(diff[:, :, 0:3]/self.CA_xyz_tot[int(self.CA_xyz_tot.shape[0]*self.sep_test)
	                                                              :int(CA_xyz_tot.shape[0]*sep_pred), :, 0:3]*100)    
	    # create array for count of frames/samples
	    count = np.arange(diff_pc.shape[0]*diff_pc.shape[1]*diff_pc.shape[2])
	    # create array for storing count of frames/samples & difference in percentage
	    self.diff_1 = np.zeros((diff_pc.shape[0]*diff_pc.shape[1]*diff_pc.shape[2], 3))
	    diff = np.reshape(diff, (diff.shape[0]*diff.shape[1]*diff.shape[2]))
	    diff_pc = np.reshape(diff_pc, (diff_pc.shape[0]*diff_pc.shape[1]*diff_pc.shape[2]))
	    self.diff_1[:, 0] = count
	    self.diff_1[:, 1] = diff 
	    self.diff_1[:, 2] = diff_pc 
	    print("Total number of coordinate data points", len(self.diff_1))
	

    def plot1(self):
	    # plot calculated difference
	    # plot histogram
	    # number of bins 
	    n_bin = 180
	    #[nhist, shist] = np.histogram(np.absolute(self.diff[:, :, 0:3]), 25)
	    [nhist, shist] = np.histogram(self.diff_1[ :, 1], n_bin)
	    print("Number of points in respective bin (%):")
	    print(nhist[81:111]/float(len(self.diff_1))*100)
	    print("Bin values:")
	    print(shist[81:111])
	    plt.semilogy(shist[1: ], (nhist/float(len(self.diff_1))*100), marker='o', linestyle='-.', color='r')
	    plt.title('x = diff. in coordinate value (in angstrom)')
	    plt.xlim(-4,4)
	    plt.ylim(1.5,5)
	    plt.show()


    def plot2(self):
	    # plot calculated difference
	    # plot histogram
	    # number of bins 
	    n_bin = 180
	    #[nhist, shist] = np.histogram(np.absolute(self.diff[:, :, 0:3]), 25)
	    [nhist, shist] = np.histogram(self.diff_1[ :, 2], n_bin)
	    print("Number of points in respective bin (%):")
	    print(nhist[:14]/float(len(self.diff_1))*100)
	    print("Bin values:")
	    print(shist[:14])
	    plt.semilogy(shist[1: ], (nhist/float(len(self.diff_1))*100), marker='o', linestyle='--', color='r')
	    plt.title('x = relative "%" diff. in coordinate value')
	    plt.xlim(0,100)
	    plt.ylim(0.08,100)
	    plt.show()
	
