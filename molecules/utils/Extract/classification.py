from __future__ import print_function
import numpy as np
import os, gzip, pickle, shutil
from six.moves import cPickle
import sys
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import random


class ExtractClassification(object):    
   
    def __init__(self, path_data_array="./input_data/native-contact/data/cont-mat.array",
                        n_traj=28, f_traj=10000, row=21, col=21):    
        """
            file_name = directory location of cont-mat.array
        """
        if (not os.path.exists(path_data_array)):
            raise Exception("Path " + str(path_data_array) + " does not exist!")
        if(n_traj < 0):
            raise Exception("Invalid input: n_traj must be greater than 0!")
        if(f_traj < 0):
            raise Exception("Invalid input: f_traj must be greater than 0!")
        if(row < 0):
            raise Exception("Invalid input: row must be greater than 0!")
        if(col < 0):
            raise Exception("Invalid input: col must be greater than 0!")

        # define parameters;
        self.path_data_array = path_data_array
        self.n_traj = n_traj
        self.f_traj = f_traj
        # row and column dimension for each frame; 
        self.row = row
        self.col = col

    def build_array(self):
        # read line by line to save in array
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
                print("Frames read: ", (j_count/row))
                k_count = k_count + 10000 * self.row      
            j_count = j_count + 1
            if j_count == (row_num):
                break

        print("Initial matrix array dimension:", np.shape(array_f_int))
        self.X = np.reshape(array_f_int, (samples, self.row, self.col))
        print("Final matrix array dimension:", np.shape(self.X))
        # select number of sample for data;
        self.X = self.X[0:1100000]
        # open contact map value for each frame;
        cont_mat = np.loadtxt(self.path_data_array)
        self.Y = cont_mat[0:len(self.X)]
        print("Input data shape:", np.shape(self.X))
        print("Input label shape:", np.shape(self.Y))


    def plot(self):
       # plot histogram of npcontacts 
       [nhist, shist] = np.histogram(self.Y[ : ,1], 25);
       plt.semilogy(shist[1: ], nhist, 'r-');
       plt.show();


    def calculate_states(self):
        # calculate folded & unfolded state
        self.fold = list()
        self.unfold = list()
        # assign labels based on contact map value for each frame
        self.dat_1 = (self.Y[:,1:2])
        # assign label: fold=1 or unfold=0 (based on contact map value < or > 0.9)
        for i in range (0, len(self.dat_1)):
            # folded state
            if self.dat_1[i] > 0.9:
                self.dat_1[i] = 1
                fold.append(i)     
           else:
               # unfolded state
               self.dat_1[i] = 0
               unfold.append(i)   
        self.dat_1 = np.reshape(self.dat_1, (len(self.dat_1))) 
        print("Number of folded state:", len(self.fold)) 
        print("Number of unfolded state:", len(self.unfold))  

    def shuffle(self):    
        # randomly choose folded & unfolded states with equal numbers
        fold_cp = self.fold
        unfold_cp = self.unfold
        print("Choose %i number of folded or unfolded random labels" % len(self.unfold))    
        z = len(self.unfold)
        self.data = np.zeros((z*2, self.X.shape[1], self.X.shape[2]))
        self.label = np.zeros((z*2))
        self.index = np.zeros((z*2))
        print("Start random choices")
        # start random selections for folded configurations    
        for i in range (0, z): 
            a = random.choice(fold_cp[:]) 
            self.index[i] = a    
            fold_cp.remove(a)   
        print("Finished choosing random choice for folded configurations")
        # start random selections for unfolded configurations    
        for i in range (z, z*2): 
            b = random.choice(unfold_cp[:])
            self.index[i] = b
            unfold_cp.remove(b)
        print("Finished choosing random choice for unfolded configurations")
        # suffle elements      
        self.index.tolist()
        random.shuffle(self.index)    
        print("Shuffle finished")   

    def save(self,  data_format=None):
        if (data_format == None):
            print("Selecting default npy format type")
            data_format = 'npy'
        if ((data_format is not 'npy') and (data_format is not 'pkl')):
            raise Exception(str(data_format) + " is not a vaild data format.\nPlease chooose either 'npy' or 'pkl'.")

        file_name = "./input_data/cont-mat." + data_format


        # store input data & lables in separate files    
        w = 0
        for i in range (0, z*2):
            self.data[i,] = self.X[int(self.index[i,])]
            self.label[i] = self.dat_1[int(index[i])]    
            if i == w*10000:
                print("Completing:", i) 
            w = w + 1
        print("New input & label created with randomly suffle")    
        # check newly created input & label
        print("Data shape:", np.shape(self.data))
        print("Labels shape:", np.shape(self.label)

        if (data_format == 'npy'):
            label_file_name = "./input_data/label.npy"
            np.save(file_name, self.data)
            np.save(label_file_name, self.label)
        elif (data_format == 'pkl'):
            result = (self.data, self.label)
            # pickling extracted data
            # open file, pickle dumping & closing file    
            file_object = open(file_name, 'wb')
            pickle.dump(result, file_object)
            file_object.close()
            # compress pickle file
            with open(file_name, 'rb') as f_in, gzip.open(file_name + ".gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
