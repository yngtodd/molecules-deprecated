from __future__ import print_function
import numpy as np;
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt; 

class FSpeptide_plot(object):

    def __init__(self):
        pass
    # original_path="./data/fs_peptide_original.npy"
    # decoded_path="./data/fs-peptide_decoded_test_150.out"
    def load(self, original_path=None, decoded_path=None):
        """
        original_path : string
            - path of the .npy file. Should be located in ./output_data or ./input_data
        decoded_path : string
            - path of the decoded .out. Should be located in ./output_data or ./input_data
        """
        if(original_path == None or decoded_path == None):
            raise ValueError("Must input original_path and decoded_path as parameters.")
        if (not os.path.exists(original_path)):
            raise Exception("Path " + str(original_path) + " does not exist!")
        if (not os.path.exists(decoded_path)):
            raise Exception("Path " + str(decoded_path) + " does not exist!")

       self.original = np.load(original_path)
       self.decoded = np.loadtxt(decoded_path)
       print(self.original.shape)
       print(self.decoded.shape)


   def calculate_diff(self):
       original_mod = np.reshape(self.original, (self.original.shape[0], self.original.shape[1]*self.original.shape[2]))
       print(original_mod.shape)
  
       self.diff = original_mod-decoded
       np.savetxt("./output_data/fs_peptide_diff.out", self.diff)
       print(self.diff.shape)

    def plot_diff(self):
        # plot calculated difference
        # plot histogram 
        # number of bins 
        n_bin = 1000
        #[nhist, shist] = np.histogram(np.absolute(diff[:, :, 0:3]), 25)
        [nhist, shist] = np.histogram(self.diff, n_bin)
        #print "number of points in respective bin (%):" 
        #print nhist/float(len(diff))*100
        #print "bin values:"
        #print shist
        plt.semilogy(shist[1: ], (nhist/float(len(self.diff))), marker='.', linestyle='-.', color='r')
        plt.title('x = diff. in coordinate value (in angstrom)')
        #plt.xlim(-4,4)
        #plt.ylim(1.5,5)
        plt.show()
        np.savetxt("./output_data/nhist_fs-peptide_diff.out", nhist)
        np.savetxt("./output_data/shist_fs-peptide_diff.out", shist[1:])
        #diff_abs = abs(original_mod-decoded) # CHANGE
        self.diff_abs = abs(self.diff)
        np.savetxt("./output_data/fs_peptide_diff_abs.out", self.diff_abs)
        print(self.diff_abs.shape)


    def plot_diff_abs(self):
        # plot calculated difference
        # plot histogram
        # number of bins 
        n_bin = 1000
        #[nhist, shist] = np.histogram(np.absolute(diff[:, :, 0:3]), 25)
        [nhist, shist] = np.histogram(self.diff_abs, n_bin)
        #print "number of points in respective bin (%):"
        #print nhist/float(len(diff))*100
        #print "bin values:"
        #print shist
        plt.semilogy(shist[1: ], (nhist/float(len(self.diff_abs))), marker='.', linestyle='-.', color='r')
        plt.title('x = diff. in coordinate value (in angstrom)')
        #plt.xlim(-4,4)
        #plt.ylim(1.5,5)
        plt.show()

        np.savetxt("./output_data/nhist_fs-peptide_diff_abs.out", nhist)
        np.savetxt("./output_data/shist_fs-peptide_diff_abs.out", shist[1:])

