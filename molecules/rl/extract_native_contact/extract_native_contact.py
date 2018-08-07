from __future__ import print_function
import gzip;
import os, sys;
import numpy as np;
from math import sqrt 
import MDAnalysis as mdanal;
from MDAnalysis.analysis import contacts;

# TODO: Add count_traj_files to utils
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

def grab_file_name(f):
    """
    f : str
        name of file
    """
    index = 0
    for i in f:
        if i == '-':
            break
        index += 1
    return f[:index]
        

class ExtractNativeContact(object):
    def __init__(self, data_path, structure_file, traj_file, n=None):
        """
        data_path : str
            path containing the pdb and trajectory (xtc, dcd) files.
        structure_file : str
            name of structure (pdb) file in data_path.
            EX) protein.pdb
        traj_file : str
            name of trajectory (xtc, dcd) file in data_path
            EX) protein-1.dcd
        n : int
            number of trajectory files to be processed. If not selected
            then all available in data_path will be selected.
        """
        if traj_file[-3:] != 'xtc' and traj_file[-3:] != 'dcd':
            raise Exception("traj_path must have extension 'xtc' or 'dcd'.")
        if structure_file[-3:] != 'pdb':
            raise Exception("structure_path must have extension 'pdb'.")

        self.structure_file = structure_file
        self.traj_file = traj_file
        #data_path = "/home/a05/Package_6_22/raw_MD_data/original/";
        self.data_path = data_path
        if n == None:
            n = count_traj_files(self.data_path, self.traj_file[-3:])
        self.n = n

        # create directories for results;
        #self.path_0 = self.data_path + "results/";
        self.path_1 = self.data_path + "native-contact/"
        self.path_2 = self.path_1 + "raw/"
        self.path_3 = self.path_1 + "data/"

    def build_directories(self):  
        # creating directories for results;
        #if not os.path.exists(self.path_0):
        #    os.mkdir(self.path_0, 0755);
        if not os.path.exists(self.path_1):
            os.mkdir(self.path_1, 0755);
        if not os.path.exists(self.path_2):
            os.mkdir(self.path_2, 0755);
        if not os.path.exists(self.path_3):
            os.mkdir(self.path_3, 0755);
        print("Directories created or if already exists - then checked")
 
    # calculate native contacts & contact map;
    def calculate_contact_matrices(self):
        # for file naming purpose;
        k = 0
        # end define parameters
        # calculate contact map over frames;
        for i in range(1, (self.n+1)):    
            # specify path of structure & trajectory files;    
            print("Creating Universe")
            # TODO: Automatically get correct pdb file
            # TODO: Automatically get trajectory files name
            # TODO: Automatically get CA residues
            u0 =mdanal.Universe(self.data_path + self.structure_file, self.data_path + grab_file_name(self.traj_file) + '-%i' % i + self.traj_file[-4:])
            self.f = len(u0.trajectory)
            print('Trajectory no:', i)
            print('Number of frames', self.f)
            # crude definition of salt bridges as contacts between CA atoms;
            #CA = "(name CA and resid 237-248 283-288 311-319 345-349 394-399)";
            #CA = "(name CA and resid 42:76)";
            # put in backbone
            CA = "(name CA and resid 1:24)"
            #CA = "(name CA and resid 42:76)";
            CA0 = u0.select_atoms(CA)
            print("Defining carbon alphas")
            #CA0 = u0.select_atoms(CA);
            # print progress;
            # print('read user defined atoms for frames:'), k;
            # calculate contact map over all frames; 
            for j in range(0, (self.f)):
                # calculating and saving native contact dat files per frame;     
                # set up analysis of native contacts ("salt bridges"); salt bridges have a distance <8 Angstrom;
                ca = contacts.ContactAnalysis1(u0, selection=(CA, CA), refgroup=(CA0, CA0), radius=8.0, 
                                               outfile= self.path_2 + 'cont-mat_%i.dat' % k)    
                ca.run(store=True, start=j, stop=j+1, step=1);
                # save ncontact figures per frame or function of residues;
                #ca.plot_qavg(filename="./fig_res/ncontact_res_%i.pdf" % k);
                # save ncontact over time;
                #ca.plot(filename="./fig_frame/ncontact_time_%i.pdf" % k);
                # read zipped native contact array files;
                inF_array = gzip.GzipFile(self.path_2 + "cont-mat_%i.array.gz" % k, 'rb');   
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
                outF_array = file(self.path_2 + "cont-mat_%i.array" % k, 'wb');
                outF_array.write(s_array);
                outF_array.close(); 
                # remove zipped array file;
                os.remove(self.path_2 + "cont-mat_%i.array.gz" %k);
                # to next file name numerics;
                k += 1;
            print('Read user defined atoms for frames:', k);
 
    def generate_array_file(self): 
        # create one contact map from all contact map files;
        # for counting purpose;
        l = 0;
        for i in range(0, self.f * self.n):
            if i==10000*l:
                print("Compressing frame:", i)
                l += 1;
            fin = open(self.path_2 + "cont-mat_%i.array" % i, "r")
            data1 = fin.read()
            fin.close()
            fout = open(self.path_3 + "cont-mat.array", "a")
            fout.write(data1)
            fout.close() 
        print("Contact map file created")

    def generate_dat_file(self):
        # create one native contact from all native contact files;
        # for counting purpose;
        l = 0;
        for i in range(0, self.f * self.n):
            if i==10000*l:
                print("Compressing frame:", i)
                l+= 1;
            fin = open(self.path_2 + "cont-mat_%i.dat" % i, "r")
            data1 = fin.read()
            fin.close()
            fout = open(self.path_3 + "cont-mat.dat", "a")
            fout.write(data1)
            fout.close() 
        print("Native contact file created");
 
    def generate_contact_matrix(self):
        self.build_directories()
        self.calculate_contact_matrices()
        self.generate_array_file()
        self.generate_dat_file()

    def plot_native_contacts(self):
        #import matplotlib.pyplot as plt
        # plot histogram of native contacts;    
        #dat_check = np.loadtxt(path_3 + 'cont-mat.dat');
        #[nhist, shist] = np.histogram(dat_check[ : ,1], 25);
        #plt.semilogy(shist[1: ], nhist, 'r-');
        #plt.savefig(path_1+'native-contact.png', dpi=600);
        #plt.show();
        #plt.clf();
        print("Not implemented. Uncomment code to use.")
  
    def map_check(self):
        # Check contact map shape
        map_check = np.loadtxt(self.path_3 + 'cont-mat.array')
        print(type(map_check))
        print("Contact map shape:", np.shape(map_check))
