#get_ipython().magic(u'matplotlib inline')
from __future__ import print_function
import gzip
import os, sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# calculate native contacts & contact map
import MDAnalysis as mdanal
from MDAnalysis.analysis import contacts

# TODO: Turn into function that takes path of structure & trajectory files and the parameters defined below.
# TODO: Automate calculation of n,f

# stucture_path = '../../raw_MD_data/original/100-fs-peptide-400K.pdb'
# trajectory_path = '../../raw_MD_data/original/trajectory-%i.xtc'

class ExtractNativeContact(object):

    def __init__(self, structure_path=None, trajectory_path=None, n=28, f=10000):

	    """
        structure_path = directory path containing the desired .pdb file (include pdb file in path).
        trajectory_path = directory path containing the desired .xtc, .dcd trajectory files (do not include trajectory files).
                          Trajectory files should be labeled trajectory-i.xtc where i is the number of the file.
        n = number of trajectories (trajectory files)
        f = number of frames per trajectory.
        """
	    if(structure_path == None or trajectory_path == None):
            raise ValueError("Must input structure_path and trajectory_path as parameters.")
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
        self.n = n
        self.f = f

    def build_directories(self):
    	"""
    	Builds directories "./input_data/native-contact/" containing "/raw/" and "/data/"
    	to hold contact matrix data .dat and .array files.
    	"""
    	print("Building Directories...")

   	    path_1 = "./input_data/native-contact/"
    	path_2 = "./input_data/native-contact/raw/"
    	path_3 = "./input_data/native-contact/data/"

    	if not os.path.exists(path_1):
            os.mkdir(path_1, 0755)
    	if not os.path.exists(path_2):
            os.mkdir(path_2, 0755)
    	if not os.path.exists(path_3):
            os.mkdir(path_3, 0755)

    	print("Completed directories creation or if already exist - then checked")

    def calculate_contact_maps(self):
    	# for counting purpose
    	k = 0
    	# calculate contact map over frames
    	for i in range(1, (self.n+1)):    
       
            # specify path of structure & trajectory files    
            u0 =mdanal.Universe(self.structure_path, self.trajectory_path + '/trajectory-%i.xtc' % i)
            # crude definition of salt bridges as contacts between CA atoms
            CA = "(name CA and resid 1:24)"
            #CA = "(name CA and resid 42:76)"
            CA0 = u0.select_atoms(CA)
            #CA0 = u0.select_atoms(CA)
       
            # print progress
            print("Read user defined atoms for frames:", i * self.f)
       
            # calculate contact map over all frames 
            for j in range(0, (self.f)):
            	# calculating and saving native contact dat files per frame     
            	# set up analysis of native contacts ("salt bridges"); salt bridges have a distance <8 Angstrom
            	ca = contacts.ContactAnalysis1(u0, selection=(CA, CA), refgroup=(CA0, CA0), radius=8.0, 
                                           outfile='./input_data/native-contact/raw/cont-mat_%i.dat' % k)    
            	ca.run(store=True, start=self.j, stop=self.j+1, step=1)

            	# save ncontact figures per frame or function of residues
            	#ca.plot_qavg(filename="./fig_res/ncontact_res_%i.pdf" % k)
            	# save ncontact over time
            	#ca.plot(filename="./fig_frame/ncontact_time_%i.pdf" % k)
            	# read zipped native contact array files

            	inF_array = gzip.GzipFile("./input_data/native-contact/raw/cont-mat_%i.array.gz" % k, 'rb')
            	s_array = inF_array.read()
            	inF_array.close()
            	# copy to another file
            	outF_array = file("./input_data/native-contact/raw/cont-mat_%i.array" % k, 'wb')
            	outF_array.write(s_array)
            	outF_array.close()
            	# remove zipped array file
            	os.remove("./input_data/native-contact/raw/cont-mat_%i.array.gz" %k)
            	# to next file name numerics
            	k += 1


    def generate_array_file(self):
    	# create one contact map from all contact map files
    	# for counting purpose
    	l = 0
    	for i in range(0, self.n * self.f):
            if i == self.f * l:
            	print("Compressing frame:", i)
            	l+= 1
            fin = open("./input_data/native-contact/raw/cont-mat_%i.array" % i, "r")
            data1 = fin.read()
            fin.close()
            fout = open("./input_data/native-contact/data/cont-mat.array", "a")
            fout.write(data1)
            fout.close() 
        print("Contact map file created");

    def generate_dat_file(self):
    	# create one native contact from all native contact files
    	# for counting purpose
    	l = 0
    	for i in range(0, self.n * self.f):
            if i == self.f * l:
            	print("Compressing frame:", i)
            	l+= 1
            fin = open("./input_data/native-contact/raw/cont-mat_%i.dat" % i, "r")
            data1 = fin.read()
            fin.close()
            fout = open("./input_data/native-contact/data/cont-mat.dat", "a")
            fout.write(data1)
            fout.close() 
        print("Native contact file created")

    def build_contact_map(self):
        generate_array_file()
        generate_dat_file()
    """
    def extract_native_contact(self):
        build_directories()
        structure_path = './raw_MD_data/original/100-fs-peptide-400K.pdb'
        trajectory_path = './raw_MD_data/original'
        calculate_contact_maps(structure_path, trajectory_path)
        build_contact_map()
    """

    def plot_native_contacts(self):
   	 # plot histogram of native contacts
    	dat_check = np.loadtxt('./input_data/native-contact/data/cont-mat.dat')
    	[nhist, shist] = np.histogram(dat_check[ : ,1], 25)
    	plt.semilogy(shist[1: ], nhist, 'r-')
    	plt.show()

    def contact_map_shape(self):
        # check contact map shape
        map_check = np.loadtxt('./input_data/native-contact/data/cont-mat.array')
        print("Contact map shape:", np.shape(map_check))

