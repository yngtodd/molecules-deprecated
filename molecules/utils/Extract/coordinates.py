from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mdanal
plt.style.use('ggplot')
import gzip, pickle, shutil
import os, sys

class ExtractCoordinates(object):

    def __init__(self, structure_path=None, trajectory_path=None, n=2, f=10000,
                 residue_start=1, residue_end=23, coor=3):
     
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
        if(residue_start < 0):
            raise Exception("Invalid input: residue_start must be greater than 0!")
        if(residue_end < 0):
            raise Exception("Invalid input: residue_end must be greater than 0!")
        if(residue_end < residue_start):
            raise Exception("Invalid input: residue_end must be greater than residue_start!")
        if(coor < 0):
            raise Exception("Invalid input: coor must be greater than 0!")
        
        # TODO: build funciton to parse n and f from structure_path and trajectory_path
        self.structure_path = structure_path
        self.trajectory_path = trajectory_path
        # number of trajectory files
        self.n = n

        # number of frames per trajectories
        self.f = f
        self.residue_start = residue_start
        self.residue_end = residue_end
        self.residue_diff = ((self.residue_end-self.residue_start+1)-2)
        # number of cartesian coordinates
        self.coor = coor
        # create zero arrays    
        self.CA_xyz_tot = np.zeros((self.n * self.f, self.residue_diff, self.coor)) 

    def write_poitions(self):
        print("Start to extract raw coordinates from the trajectory files")
        # for counting purpose
        k = 0
        # run over frames
        for i in range(0, (self.n)):    
            # specify path of structure & trajectory files    
            u0 =mdanal.Universe(self.structure_path, self.trajectory_path + '/trajectory-%i.xtc' %(i+1))
            # crude definition of salt bridges as contacts between CA atoms
            CA = "(name CA and resid 1:24)"
            CA0 = u0.select_atoms(CA)
            # create zero arrays
            CA_xyz_frame = np.zeros((self.f,self.residue_diff,self.coor))
            # write positions of each selected atoms of each frame
            for j in range(0, (self.f)):
                for ts in u0.trajectory[(j+0):(j+1):(1)]:
                    CA_xyz_frame[j] = (CA0.positions)
            # save selected atom coordinate
            self.CA_xyz_tot[i*self.f:(i+1)*self.f] = CA_xyz_frame
        print("Finished writing positions")

    def summary(self):
        print("Position matrix array dimension:",np.shape(self.CA_xyz_tot))
        print("Minimum values of atom position along three directions:")   
        x_min = np.amin(self.CA_xyz_tot[:, :, 0])
        y_min = np.amin(self.CA_xyz_tot[:, :, 1])
        z_min = np.amin(self.CA_xyz_tot[:, :, 2])
        print("Along direction:")
        print("x: ", x_min, " y: ", y_min, " z: ", z_min)
        print("Shift origin by nearest integer number (or original minimum values) along each positive direction:")
        self.CA_xyz_tot[:, :, 0] +=  np.ceil(abs(x_min))
        self.CA_xyz_tot[:, :, 1] +=  np.ceil(abs(y_min))
        self.CA_xyz_tot[:, :, 2] +=  np.ceil(abs(z_min))
        print("Shift along each direction:", " x: ", np.ceil(abs(x_min)), " y: ", np.ceil(abs(y_min)), " z: ", np.ceil(abs(z_min)))
        print("Minimum values of atom position along three directions after origin shift:")
        x_min = np.amin(self.CA_xyz_tot[:, :, 0])
        y_min = np.amin(self.CA_xyz_tot[:, :, 1])
        z_min = np.amin(self.CA_xyz_tot[:, :, 2])
        print("Along direction:")
        print("x: ", x_min, " y: ", y_min, " z: ", z_min)

        print("Maximum values of atom position along three directions after origin shift:")
        x_max = np.amax(self.CA_xyz_tot[:, :, 0])
        y_max = np.amax(self.CA_xyz_tot[:, :, 1])
        z_max = np.amax(self.CA_xyz_tot[:, :, 2])
        print("Along direction:")
        print("x: ", x_max, " y: ", y_max, " z: ", z_max)    


        #print("This is for flattening the data")
        #self.CA_xyz_tot = np.reshape(self.CA_xyz_tot, (self.n * self.f, (self.residue_diff * self.coor)))
        #print("New shape after flatteing data:", np.shape(self.CA_xyz_tot))

    def build_directories(self):
        """
        Builds directories "./coor/" containing "/data/"
        to hold coordinate data.
        """
        print("Building Directories...")

        path_1 = "./input_data/coor/"
        path_2 = "./input_data/coor/data/"

        if not os.path.exists(path_1):
            os.mkdir(path_1, 0755)
        if not os.path.exists(path_2):
            os.mkdir(path_2, 0755)

        print("Completed directories creation or if already exist - then checked")

    def save(self, file_name=None):
        if (file_name == None):
            print("Selecting default file_name")
            file_name = "./input_data/coor/data/protein_coor.pkl"

        build_directories()
        # pickling extracted data
        # open file, pickle dumping & closing file    
        file_object = open(file_name, 'wb')
        pickle.dump(self.CA_xyz_tot, file_object)
        file_object.close()
        # compress pickle file
        with open(file_name, 'rb') as f_in, gzip.open(file_name + ".gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
