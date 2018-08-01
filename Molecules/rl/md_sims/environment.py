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


class environment(object):
    def __init__(self, cvae_weights_path, sim_steps=20000, traj_out_freq=100, native_pdb=None):
        
        # State variables
        self.rmsd_state = []
        self.num_native_contacts = []
        self.obs_in_cluster = []
        self.dcd_file = 'output-0.dcd'
        self.pdb_file = 'output.pdb'
        self.initial_pdb = ['/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-0.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-1.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-2.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-3.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-4.pdb']
        
        if native_pdb == None:
            # For testing purposes
            native_pdb = '/home/a05/data/fs-peptide/raw_MD_data/fs-peptide.pdb'
        else:
            self.native_pdb = native_pdb
            
        pdb_stack = []
        self.rmsd_threshold
    
    def initial_state(self, path):
        # Run MD simulation
        self.MDsimulation(path, self.dcd_file, self.initial_pdb[0])
        
        # Calculate contact matrix
        path_1 = path + "%i/sim_%i_%i/" % (0,0,0)
        cm = ExtractNativeContact(path_1, pdb_file, dcd_file)
        cm.generate_contact_matrix()
        
        # Pass contact matrix through CVAE and retrieve encoded_data
        cvae = CVAE(path=path_1, sep_train=0, sep_test=0, sep_pred=1, f_traj=self.sim_steps/self.traj_out_freq)
        cvae.load_contact_matrix(path_1 + "native-contact/data/cont-mat.dat",
                                 path_1 + "native-contact/data/cont-mat.array")
        cvae.compile()
        cvae.load_weights(self.cvae_weights_path)
        encoded_data = cvae.encode_pred()
        np.save("./results/final_output/intermediate_data/encoded_data_rl_%i.npy" % 0, encoded_data)
        
        # Perform DBSCAN clustering on all the data produced in the ith RL iteration.
        db = DBSCAN(eps=d_eps, min_samples=d_min_samples).fit(encoded_data)
        for cluster in Counter(db.labels_):
            indices = get_cluster_indices(labels=db.labels_, cluster=cluster)
            rmsd_values = []
            path_to_pdb = []
            for ind in indices:
                path_1 = path + "%i/sim_%i_%i/pdb_data/output-%i.pdb" % (0, 0, 0, ind)
                u = mdanal.Universe(path_1)
                R = RMSD(u, self.native_protein)
                R.run()
                # For DBSCAN outliers
                if cluster == -1:
                    if R.rmsd[0,2] < self.rmsd_threshold:
                        # Start next rl iteration with this pdb path_1
                        print("RMSD threshold:", self.rmsd_threshold)
                        print("RMSD to native contact for outlier at index %i :" % ind, R.rmsd[0,2])
                        pdb_stack.append(path_1)
                # For RMSD outliers within DBSCAN clusters
                else:
                    rmsd_values.append(R.rmsd[0,2])
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
                        print("RMSD to native contact for clustered outlier at index %i :" % path_to_pdb[ind][1], rmsd_values[ind])
                        pdb_stack.append(path_to_pdb[ind][0]) 
                    ind += 1
        
    
    def get_state(self):
        pass
    
    def action(self):
        # Take action based on state
        pass
    
    def reward(self):
        pass
 
    
    def step(self, action):
        # Take action
        #return state, reward, done, _
        pass
    
    def MDsimulation(self, path, out_dcd_file, pdb_in, 
                     ff='amber14-all.xml', 
                     water_model='amber14/tip3pfb.xml'):
        if not os.path.exists(path):
            raise Exception("Path: " + str(path) + " does not exist!")
     
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