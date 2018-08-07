#get_ipython().magic(u'matplotlib inline')
from __future__ import print_function
import gzip;
import os, sys;
import numpy as np;
import matplotlib.pyplot as plt;
plt.style.use('ggplot');

# calculate distance matrix;
import MDAnalysis as mdanal;
from MDAnalysis.analysis import distances;

# structure_path = "./HIV_aligned_CA_0.pdb"
# trajectory_path = './HIV_aligned_CA.trr'

class ExtractDistanceMatrix(object):
    
    def __init__(self, structure_path=None, trajectory_path=None, n=1, f = 10000):
 
        """
        structure_path : str
            directory path containing the desired .pdb file (include pdb file in path).
        trajectory_path : str
            directory path containing the desired .xtc, .dcd trajectory files (do not include trajectory files).
            Trajectory files should be labeled trajectory-i.xtc where i is the number of the file.
        n : int
            number of trajectories (trajectory files)
        f : int
            number of frames per trajectory.
        """

        if(structure_path == None or trajectory_path == None):
            raise Exception("Must input structure_path and trajectory_path as parameters.")
        if (not os.path.exists(structure_path)):
            raise Exception("Path " + str(structure_path) + " does not exist!")
        if (not os.path.exists(trajectory_path)):
            raise Exception("Path " + str(trajectory_path) + " does not exist!")
        if(n < 0):
            raise Exception("Invalid input: n must be greater than 0!")
        if(f < 0):
            raise Exception("Invalid input: f must be greater than 0!")
        # TODO: build funciton to parse n and f from structure_path and trajectory_path
        self.structure_path = structure_path
        self.trajectory_path = trajectory_path
        # number of trajectory files
        self.n = n

        # number of frames per trajectories;
        self.f = f

        # defining holder for distance matrix;
        self.ca_tot = [];

    def calculate_distance_matrix(self):
        # calculate contact map over frames;
        for i in range(1, (self.n+1)):    
            # specify path of structure & trajectory files;    
            u0 =mdanal.Universe(self.structure_path, self.trajectory_path + '/trajectory-%i.xtc' % i)
            # crude definition of salt bridges as contacts between CA atoms;
            CA1 = "(name CA and resid 55)";
            CA2 = "(name CA and resid 154)";
            CA10 = u0.select_atoms(CA1);
            CA20 = u0.select_atoms(CA2);
            #CA0 = u0.select_atoms(CA);
            # print progress;
            print("Read user defined atoms for frames:", i * self.f);
            # calculate contact map over all frames; 
            for ts in u0.trajectory[0:self.f]:
                # calculating and saving native contact dat files per frame;     
                # set up analysis of native contacts ("salt bridges"); salt bridges have a distance <8 Angstrom;
               ca_dist = distances.distance_array(CA10.coordinates(), CA20.coordinates()); 
               self.ca_tot = np.append(self.ca_tot,ca_dist);

    def build_directories(self):
        """
        Builds directories "./native-contact/" containing "/data/"
        to hold distance matrix data.
        """
        print("Building Directories...")

        path_1 = "./input_data/native-contact/"
        path_2 = "./input_data/native-contact/data/"

        if not os.path.exists(path_1):
            os.mkdir(path_1, 0755)
        if not os.path.exists(path_2):
            os.mkdir(path_2, 0755)

        print("Completed directories creation or if already exist - then checked")

    def save(self)
        #reshaping distance matrix & saving;
        self.ca_tot = np.reshape((self.ca_tot), (len(self.ca_tot), 1)); 
        print np.shape(self.ca_tot);
        np.save("./input_data/native-contact/data/dist-mat.npy", self.ca_tot);


    def plot(self):
        # plot histogram of native contacts;    
        dat_check = np.load('./input_data/native-contact/data/dist-mat.npy');
        [nhist, shist] = np.histogram(dat_check[ : ,0], 25);
        plt.semilogy(shist[1: ], nhist, 'r-');
        plt.show();
