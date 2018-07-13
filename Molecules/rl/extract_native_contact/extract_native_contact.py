from __future__ import print_function
import gzip;
import os, sys;
import numpy as np;
from math import sqrt 
import MDAnalysis as mdanal;
from MDAnalysis.analysis import contacts;

import glob
def count_traj_files(path, extension):
    if not os.path.exists(path):
	raise Exception ("Path "+ str(path) + " does not exist.")
    """
    path : string
        Path of directory containing trajectory files.
    extension : string
        File extension type for trajectory files.
        EX) 'dcd', 'xtc', ...
    """
    return len(glob.glob1(path,"*."+extension)) 

class ExtractNativeContact(object):
    def __init__(self, data_path, traj_extension, n=None):
	"""
	data_path : str
	    path containing the pdb and trajectory (xtc, dcd) files.
	traj_extension : str
	    file extension of the trajectory files in data_path
	n : int
	    number of trajectory files to be processed. If not selected
	    then all available in data_path will be selected.
	"""
	if traj_extension != 'xtc' or traj_extension != 'dcd':
	    raise Exception("traj_extension must be 'xtc' or 'dcd'.")
	
	#data_path = "/home/a05/Package_6_22/raw_MD_data/original/";
	self.data_path = data_path
	self.traj_extension = traj_extension
	if n == None:
	    n = count_traj_files(data_path, self.traj_extension)
	else:
	    self.n = n

	# create directories for results;
	self.path_0 = "./results/";
	self.path_1 = "./results/native-contact/";
	self.path_2 = "./results/native-contact/raw/";
	self.path_3 = "./results/native-contact/data/";
    def build_directories(self):  
	# creating directories for results;
	if not os.path.exists(self.path_0):
	    os.mkdir(self.path_0, 0755);
	if not os.path.exists(self.path_1):
	    os.mkdir(self.path_1, 0755);
	if not os.path.exists(self.path_2):
	    os.mkdir(self.path_2, 0755);
	if not os.path.exists(self.path_3):
	    os.mkdir(self.path_3, 0755);
	print("Directories created or if already exists - then checked")
 
 
 
# calculate native contacts & contact map;

# number of frames per trajectories;
#f = int(0.01*10000);
# for file naming purpose;
k = 0;
# end define parameters
# calculate contact map over frames;
for i in range(1, (n+1)):    
    # specify path of structure & trajectory files;    
    print "Creating Universe"
    u0 =mdanal.Universe(data_path + '100-fs-peptide-400K.pdb', data_path + 'trajectory-%i.xtc' % i);
    f = len(u0.trajectory);
    print('trajectory no:'), i;
    print('number of frames'), f;
    # crude definition of salt bridges as contacts between CA atoms;
    #CA = "(name CA and resid 237-248 283-288 311-319 345-349 394-399)";
    #CA = "(name CA and resid 42:76)";
    CA = "(name CA and resid 1:24)";
    #CA = "(name CA and resid 42:76)";
    CA0 = u0.select_atoms(CA);
    print "Defining carbon alphas"
    #CA0 = u0.select_atoms(CA);
    # print progress;
#    print('read user defined atoms for frames:'), k;
    # calculate contact map over all frames; 
    for j in range(0, (f)):
        # calculating and saving native contact dat files per frame;     
        # set up analysis of native contacts ("salt bridges"); salt bridges have a distance <8 Angstrom;
        ca = contacts.ContactAnalysis1(u0, selection=(CA, CA), refgroup=(CA0, CA0), radius=8.0, 
                                       outfile= path_2 + 'cont-mat_%i.dat' % k)    
        ca.run(store=True, start=j, stop=j+1, step=1);
        # save ncontact figures per frame or function of residues;
        #ca.plot_qavg(filename="./fig_res/ncontact_res_%i.pdf" % k);
        # save ncontact over time;
        #ca.plot(filename="./fig_frame/ncontact_time_%i.pdf" % k);
        # read zipped native contact array files;
        inF_array = gzip.GzipFile(path_2 + "cont-mat_%i.array.gz" % k, 'rb');   
        s_array = inF_array.read();
	inF_array.close();
	arr = s_array
        arr = np.fromstring(s_array, dtype='float32', sep=' ')
        arr = np.reshape(arr, (int(sqrt(arr.shape[0])), int(sqrt(arr.shape[0]))))
	for i in range(0, arr.shape[0]):
    	    arr[i][i] = 0.
	    if i == arr.shape[0] - 1:
		break
	    else:
		arr[i][i+1] = 0.
		arr[i+1][i] = 0.
	temp = ''
	for ind in range(0, arr.shape[0]):
	    for inj in range(0, arr.shape[0]):
	        temp += str( arr[ind][inj])    
	   	temp += ' '
	    temp += '\n'
	s_array = temp
        # copy to another file;
        outF_array = file(path_2 + "cont-mat_%i.array" % k, 'wb');
        outF_array.write(s_array);
        outF_array.close(); 
        # remove zipped array file;
        os.remove(path_2 + "cont-mat_%i.array.gz" %k);
        # to next file name numerics;
        k += 1;
    print('read user defined atoms for frames:'), k;
 
 
# create one contact map from all contact map files;
# for counting purpose;
l = 0;
for i in range(0, k):
    if i==10000*l:
        print "compressing frame:", i;
        l+= 1;
    fin = open(path_2 + "cont-mat_%i.array" % i, "r")
    data1 = fin.read()
    fin.close()
    fout = open(path_3 + "cont-mat.array", "a")
    fout.write(data1)
    fout.close() 
print "contact map file created";
# create one native contact from all native contact files;
# for counting purpose;
l = 0;
for i in range(0, k):
    if i==10000*l:
        print "compressing frame:", i;
        l+= 1;
    fin = open(path_2 + "cont-mat_%i.dat" % i, "r")
    data1 = fin.read()
    fin.close()
    fout = open(path_3 + "cont-mat.dat", "a")
    fout.write(data1)
    fout.close() 
print "native contact file created";
 
#import matplotlib.pyplot as plt
# plot histogram of native contacts;    
#dat_check = np.loadtxt(path_3 + 'cont-mat.dat');
#[nhist, shist] = np.histogram(dat_check[ : ,1], 25);
#plt.semilogy(shist[1: ], nhist, 'r-');
#plt.savefig(path_1+'native-contact.png', dpi=600);
#plt.show();
#plt.clf();
 
 
# check contact map shape;
map_check = np.loadtxt(path_3 + 'cont-mat.array');
print type(map_check)
print "contact map shape:", np.shape(map_check)
