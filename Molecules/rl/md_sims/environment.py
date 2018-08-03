from __future__ import print_function
import os
import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

# From Molecules
import sys
sys.path.append('../')
from extract_native_contact.extract_native_contact import ExtractNativeContact
from vae_conv_train_load.cvae_api import CVAE

# For clustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import Counter

# For reward function
import MDAnalysis as mdanal
from MDAnalysis.analysis.rms import RMSD
from scipy import stats

# For calc_native_contact()
from MDAnalysis.analysis import contacts
import gzip

def get_cluster_indices(labels, cluster=-1):
    """
    labels : DBSCAN.labels_ object (numpy array)
    cluster : int
        cluster label whose in dices are desired. 
        cluster=-1 default automatically selects outliers. 
    """
    indices = []
    for i,j in enumerate(labels):
        if j == cluster:
            indices.append(i)
    return indices

def calc_native_contact(native_pdb, out_path, dat_file='cont-mat.dat', array_file='cont-mat.array'):
    u_native = mdanal.Universe(native_pdb)
    CA = "(name CA and resid 1:24)"
    CA0 = u_native.select_atoms(CA)
    ca = contacts.ContactAnalysis1(u_native, selection=(CA, CA), refgroup=(CA0, CA0), radius=8.0, 
                                   outfile= out_path + '/' + dat_file)    
    ca.run(store=True, start=0, stop=1, step=1)
    inF_array = gzip.GzipFile(out_path + '/' + array_file +'.gz', 'rb')
    s_array = inF_array.read()
    inF_array.close()
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
    # copy to another file
    outF_array = file(out_path + '/' + array_file, 'wb')
    outF_array.write(s_array)
    outF_array.close()
    # remove zipped array file
    os.remove(out_path + '/' + array_file +'.gz')


class environment(object):
    def __init__(self, cvae_weights_path, sim_steps=20000, traj_out_freq=100, native_pdb=None):
        
        # State variables
        self.rmsd_state = []
        self.num_native_contacts = []
        self.obs_in_cluster = []
        self.num_dbscan_cluster = -2
        
        # IO variables
        self.dcd_file = 'output-1.dcd'
        self.pdb_file = 'output.pdb'
        # For testing purposes
        self.initial_pdb = ['/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-0.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-1.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-2.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-3.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-4.pdb']
        
        if native_pdb == None:
            # For testing purposes
            self.native_pdb = '/home/a05/data/fs-peptide/raw_MD_data/fs-peptide.pdb'
        else:
            self.native_pdb = native_pdb
        self.native_protein = mdanal.Universe(self.native_pdb)
	 
	self.cvae_weights_path = cvae_weights_path
	self.sim_steps = sim_steps
	self.traj_out_freq = traj_out_freq    
        self.pdb_stack = []
        self.rmsd_threshold = 5.0 # Set to random seed?
        # DBSCAN params
	self.d_eps = 0.1
	self.d_min_samples = 10
    
    def initial_state(self, path):
        # Run MD simulation
        self.MDsimulation(path)
        self.internal_step(path=path, i_episode=0)            
        return np.array(self.rmsd_state)
        
    
    def get_state(self):
        return np.array(self.rmsd_state)
    
    
    def reward(self):
        # Before calc assert that each vector is the same length
        if len(self.rmsd_state) != len(self.num_native_contacts):
            raise Exception("Shape mismatch")
        if len(self.rmsd_state) != len(self.obs_in_cluster):
            raise Exception("Shape mismatch")
         
        reward = 0.0    
        n = self.sim_steps/self.traj_out_freq
        for i in range(n):
	    num = float(self.num_native_contacts[i]) + self.rmsd_threshold
	    den = float(self.obs_in_cluster[i]) + self.rmsd_state[i]
            reward += num/den
	return (self.num_dbscan_cluster*reward/n)
    
    def step(self, action, path, i_episode):
        # Take action
        #return state, reward, done
        self.rmsd_threshold = action
        self.MDsimulation(path, self.dcd_file, self.initial_pdb[0])
        self.internal_step(path, i_episode)
        return (np.array(self.rmsd_state), self.reward(), len(self.pdb_stack) == 0) 
        
    
    def MDsimulation(self, path, out_dcd_file=None, pdb_in=None, 
                     ff='amber14-all.xml', 
                     water_model='amber14/tip3pfb.xml'):
        if not os.path.exists(path):
            raise Exception("Path: " + str(path) + " does not exist!")
        if out_dcd_file==None:
            out_dcd_file=self.dcd_file
        if pdb_in==None:
            if len(self.pdb_stack) == 0:
                pdb_in=self.initial_pdb[0]
            else:
                pdb = self.pdb_stack[-1]
                self.pdb_stack.pop()
                             
        pdb = PDBFile(pdb_in)
        
        forcefield = ForceField(ff, water_model)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                                         nonbondedCutoff=1.0*nanometer, constraints=HBonds)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)
        # Start back from output.pdb last frame
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        # Saves step every "traj_out_freq" to a DCD file. 
        simulation.reporters.append(DCDReporter(path + out_dcd_file, self.traj_out_freq))
        # For every step saved in the DCD we save a corresponding PDB file.
        for i in range(self.sim_steps/self.traj_out_freq):
            simulation.reporters.append(PDBReporter(path + "pdb_data/output-%i.pdb" % i, self.traj_out_freq))
            simulation.step(self.traj_out_freq)
            simulation.reporters.pop()

        # Writes the final PDB file to the same directory where the DCD file is saved.
        fin = open(path + "pdb_data/output-%i.pdb" % (self.sim_steps/self.traj_out_freq - 1))
        final_pdb_data = fin.read()
        fin.close()
        fout = open(path + "/output.pdb", 'w')
        fout.write(final_pdb_data)
        fout.close()
            
    def internal_step(self, path, i_episode):
        # Calculate contact matrix
        cm = ExtractNativeContact(path, self.pdb_file, self.dcd_file)
        cm.generate_contact_matrix()
        
        # Pass contact matrix through CVAE and retrieve encoded_data
        cvae = CVAE(path=path, sep_train=0, sep_test=0, sep_pred=1, f_traj=self.sim_steps/self.traj_out_freq)
        cvae.load_contact_matrix(path + "native-contact/data/cont-mat.dat",
                                 path + "native-contact/data/cont-mat.array")
        cvae.compile()
        cvae.load_weights(self.cvae_weights_path)
        encoded_data = cvae.encode_pred()
        np.save("./results/final_output/intermediate_data/encoded_data_rl_%i.npy" % i_episode, encoded_data)
        
        # Calculate rmsd values for each PDB file sampled.
        self.rmsd_state = []
        for i in range(self.sim_steps/self.traj_out_freq):
            path_1 = path + "/pdb_data/output-%i.pdb" % i
            u = mdanal.Universe(path_1)
            R = RMSD(u, self.native_protein)
            R.run()
            self.rmsd_state.append(R.rmsd[0,2])
       
        # Calculate number of native contacts for state
        self.num_native_contacts = []
        # Build native contact matrix
	if i_episode == 0:
            calc_native_contact(native_pdb=self.native_pdb,
                            	out_path='./results/final_output',
				dat_file='native-cont-mat.dat',
				array_file='native-cont-mat.array')
	else:
	    calc_native_contact(native_pdb=self.native_pdb,
                                out_path='./results/final_output')
        
        fin = open('./results/final_output/native-cont-mat.array', "r")
        native_cont_mat = fin.read()
        fin.close()
        native_cont_mat = np.fromstring(native_cont_mat, dtype='float32', sep=' ')
        n = int(sqrt(native_cont_mat.shape[0]))
        for i in range(self.sim_steps/self.traj_out_freq):
            fin = open(path + "native-contact/raw/cont-mat_%i.array" % i)
            ith_cont_mat = fin.read()
            fin.close()
            ith_cont_mat = np.fromstring(ith_cont_mat, dtype='float32', sep=' ')
            counter = 0
            row = 0
            while row < n + 2:
                col = row + 2
                shift = row * n
                while col < n:
                    if ith_cont_mat[shift + col] == native_cont_mat[shift + col]:
                        counter += 1
                    col += 1 
                
                row += 1
            self.num_native_contacts.append(counter)     
        
        # Perform DBSCAN clustering on all the data produced in the ith RL iteration.
        db = DBSCAN(eps=self.d_eps, min_samples=self.d_min_samples).fit(encoded_data)
        # Compute number of observations in the DBSCAN cluster of the ith PDB
        self.obs_in_cluster = []
        labels_dict = Counter(db.labels_)
        # Compute number of DBSCAN clusters for reward function
        self.num_dbscan_clusters = len(labels_dict)
        for label in db.labels_:
            self.obs_in_cluster.append(labels_dict[label])
            
        for cluster in Counter(db.labels_):
            indices = get_cluster_indices(labels=db.labels_, cluster=cluster)
            path_to_pdb = []
            rmsd_values = []
            for ind in indices:
                path_1 = path + "/pdb_data/output-%i.pdb" % ind
                # For DBSCAN outliers
                if cluster == -1:
                    if self.rmsd_state[ind] < self.rmsd_threshold:
                        # Start next rl iteration with this pdb path_1
                        print("RMSD threshold:", self.rmsd_threshold)
                        print("RMSD to native contact for DBSCAN outlier at index %i :" % ind, self.rmsd_state[ind])
                        pdb_stack.append(path_1)
                # For RMSD outliers within DBSCAN clusters
                else:
                    rmsd_values.append(self.rmsd_state[ind])
                    path_to_pdb.append((path_1, ind))
            # For RMSD outliers within DBSCAN clusters
            if cluster != -1:
                rmsd_array = np.array(rmsd_values)
                rmsd_zscores = stats.zscore(rmsd_array)
                ind = 0
                for zscore in rmsd_zscores:
                    # z-score of -3 marks outlier for a normal distribution.
                    # Assuming Normal Distribution of RMSD values because 
                    # CVAE yields normally distributed clusters.
                    if zscore <= -3:
                        print("RMSD to native contact for DBSCAN clustered outlier at index %i :" % path_to_pdb[ind][1], rmsd_values[ind])
                        self.pdb_stack.append(path_to_pdb[ind][0]) 
                    ind += 1
            
