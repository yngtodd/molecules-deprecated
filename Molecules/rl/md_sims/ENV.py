from __future__ import print_function
import os
import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import sys
sys.path.append('../')
from extract_native_contact.extract_native_contact import ExtractNativeContact
from vae_conv_train_load.cvae_api import CVAE

# For clustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import Counter

# RMSD painted clusters
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# For reward function
import MDAnalysis as mdanal
from MDAnalysis.analysis.rms import RMSD
from scipy import stats

def get_all_encoded_data(dir_path, num):
    data = []
    if num == 0:
        data.append(np.load(dir_path + "/encoded_data_rl_%i.npy" % num))
        return data[0]
    for i in range(num + 1):
        data.append(np.load(dir_path + "/encoded_data_rl_%i.npy" % i))
    data = np.array(data)
    return np.reshape(data, (data.shape[0] * data.shape[1], data.shape[-1]))

def scatter_plot_rmsd(data, title, save_path, rmsd_values, vmin=None, vmax=None):
    [n,s] = np.histogram(rmsd_values, 11)
    d = np.digitize(rmsd_values, s)
    cmi = plt.get_cmap('jet')
    if vmin == None and vmax == None:
        cNorm = mpl.colors.Normalize(vmin=min(rmsd_values), vmax=max(rmsd_values))
    else:
        cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter3D(np.ravel(data[:, 0]),
                     np.ravel(data[:, 1]),
                     np.ravel(data[:, 2]),
                     marker='o', c=scalarMap.to_rgba(rmsd_values))
    ax.set_xlim3d(np.amin(np.ravel(data[:, 0])), np.amax(np.ravel(data[:, 0])))
    ax.set_ylim3d(np.amin(np.ravel(data[:, 1])), np.amax(np.ravel(data[:, 1])))
    ax.set_zlim3d(np.amin(np.ravel(data[:, 2])), np.amax(np.ravel(data[:, 2])))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    scalarMap.set_array(rmsd_values)
    fig.colorbar(scalarMap)
    plt.title(title)
    plt.savefig(save_path, dpi=600)
    plt.cla()
    plt.close(fig)

def scatter_plot(data, title, save_path, color='b'):
    """
    data : numpy array
        must be of dimension (n,3).
    title : str
        title of desired plot.
    save_path : str
        file name of save location desired. Containing directory must
        already exist.
    color : str of list
        color scheme desired.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(title)
    plt.xlim(np.amin(data[:, 0]), np.amax(data[:, 0]))
    plt.ylim(np.amin(data[:, 1]), np.amax(data[:, 1]))
    ax.set_zlim(np.amin(data[:, 2]), np.amax(data[:, 2]))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, marker='o')
    plt.savefig(save_path)
    plt.cla()
    plt.close(fig)

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

class ENV(object):

    def __init__(self, cvae_weights_path, iterations=1, 
                 sim_num=1, sim_steps=20000, traj_out_freq=100,
                 native_pdb=None, initial_pdb=None):
        if initial_pdb == None:
            # For testing purposes
            self.initial_pdb = ['/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-0.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-1.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-2.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-3.pdb',
                                '/home/a05/data/fs-peptide/raw_MD_data/native-state/fs-peptide-4.pdb']
        else:
             self.initial_pdb = initial_pdb
        if native_pdb == None:
            # For testing purposes
            native_pdb = '/home/a05/data/fs-peptide/raw_MD_data/fs-peptide.pdb'
        else:
            self.native_pdb = native_pdb

        self.native_protein = mdanal.Universe(native_pdb)
        self.iterations = iterations
        self.sim_num = sim_num
        self.sim_steps = sim_steps
        self.traj_out_freq = traj_out_freq

        if len(self.initial_pdb) < self.sim_num:
            raise Exception("PDB mismatch. sim_num must match number of initial pdb files given.")
        if not os.path.exists(cvae_weights_path):
            raise Exception("Path " + str(cvae_weights_path) + " does not exist!")
        self.cvae_weights_path = cvae_weights_path

        if not os.path.exists("./results"):
            os.mkdir("./results", 0755)
        if not os.path.exists("./results/final_output"):
            os.mkdir("./results/final_output")
        if not os.path.exists("./results/final_output/intermediate_data"):
            os.mkdir("./results/final_output/intermediate_data")
            
        # RL Environment variables
      
    
    def run_simulation(self, path, out_dcd_file, pdb_in=None, 
                       initial_rl_loop=False, ff='amber14-all.xml', 
                       water_model='amber14/tip3pfb.xml'):
        if not os.path.exists(path):
            raise Exception("Path: " + str(path) + " does not exist!")
       
        # TODO: Add other parameters for simulation + exception handling
        if initial_rl_loop == True:
            pdb_file = self.initial_pdb[0]
            self.initial_pdb = self.initial_pdb[1:]
            pdb = PDBFile(pdb_file)
        else:
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

    def execute(self):
        pdb_file = 'output.pdb'
        dcd_file = 'output-1.dcd'
        pdb_stack = []
        # spawn_pdb is a place holder to allow code to run.
        # in the future it must be changed to an RL spwan or random PDB file.
        spawn_pdb = self.initial_pdb[0]
        # Parameters for DBSCAN clustering.
        d_eps = 0.1
        d_min_samples = 10
        # Naive RMSD threshold.
        rmsd_threshold = 5.0

        for i in range(0, self.iterations):
            path = "./results/iteration_rl_"
            if not os.path.exists(path + "%i" % i):
                os.mkdir(path + "%i" % i, 0755)
            for j in range(0, self.sim_num):
                path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
                if not os.path.exists(path_1):
                    os.mkdir(path_1, 0755)
                    os.mkdir(path_1 + "/cluster", 0755)
                    os.mkdir(path_1 + "/pdb_data", 0755)
                # TODO: Optimize so that the simulation jobs are split over
                #       the available GPU nodes. May be possible with python
                #       subprocess. It would be a good idea to pull 
                #       self.run_simulation(path_1) out of the inner for loop
                if i == 0:
                    self.run_simulation(path_1, dcd_file, initial_rl_loop = True)
                else:
                    if len(pdb_stack) == 0:
                        self.run_simulation(path_1, dcd_file, spawn_pdb)
                        print("Using spawn PDB.")
                    else:
                        self.run_simulation(path_1, dcd_file, pdb_in=pdb_stack[-1])
                        if len(pdb_stack) == 1:
                            spawn_pdb = pdb_stack[-1]
                            rmsd_threshold += 0.50
                            pdb_stack.pop()
        
            # Calculate contact matrix .array and .dat files for each simulation
            # run. Files are placed in native-contact/data inside each simulation
            # directory.
            # TODO: Parallelize
            for j in range(0, self.sim_num):
                path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
                cm = ExtractNativeContact(path_1, pdb_file, dcd_file)
                cm.generate_contact_matrix()
                
            # Process contact matrix with CVAE algorithm for each simulation.
            # Requires pre-trained CVAE.
            # TODO: compile CVAE outside of loop and pass in weights.
            #       then pass in cont-mat files on the fly and update the data.
            # TODO: Parallelize
            total_data = []
            for j in range(0, self.sim_num):
                path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
                cvae = CVAE(path=path_1, sep_train=0, sep_test=0, sep_pred=1, f_traj=self.sim_steps/self.traj_out_freq)
                cvae.load_contact_matrix(path_1 + "native-contact/data/cont-mat.dat",
                                         path_1 + "native-contact/data/cont-mat.array")
                cvae.compile()
                cvae.load_weights(self.cvae_weights_path)
                encoded_data = cvae.encode_pred()
  
                print("Encoded data shape:", encoded_data.shape)
                total_data.append(encoded_data)    

            print("total_data len:", len(total_data))
            total_data = np.array(total_data)
            total_data = np.reshape(total_data, (total_data.shape[0] * total_data.shape[1], total_data.shape[-1]))
            print("total_data shape:", total_data.shape)
            np.save("./results/final_output/intermediate_data/encoded_data_rl_%i.npy" % i, np.array(total_data))
      
            # Perform DBSCAN clustering on all the data produced in the ith RL iteration.
            db = DBSCAN(eps=d_eps, min_samples=d_min_samples).fit(total_data)
            for cluster in Counter(db.labels_):
                print(Counter(db.labels_))
                print("Current cluster:", cluster)
                indices = get_cluster_indices(labels=db.labels_, cluster=cluster)
                print("indices length:", len(indices))
                rmsd_values = []
                path_to_pdb = []
                for ind in indices:
                    sim_ind = ind / (self.sim_steps/self.traj_out_freq)
                    pdb_ind = ind % (self.sim_steps/self.traj_out_freq)
                    path_1 = path + "%i/sim_%i_%i/pdb_data/output-%i.pdb" % (i, i, sim_ind, pdb_ind)
                    u = mdanal.Universe(path_1)
                    R = RMSD(u, self.native_protein)
                    R.run()
                    # For DBSCAN outliers
                    if cluster == -1:
                        if R.rmsd[0,2] < rmsd_threshold:
                            # Start next rl iteration with this pdb path_1
                            print("RMSD threshold:", rmsd_threshold)
                            print("RMSD to native contact for outlier at index %i :" % ind, R.rmsd[0,2])
                            pdb_stack.append(path_1)
                    # For RMSD outliers within DBSCAN clusters
                    else:
                        rmsd_values.append(R.rmsd[0,2])
                        path_to_pdb.append((path_1, pdb_ind))
                # For RMSD outliers within DBSCAN clusters
                if cluster != -1:
                    rmsd_array = np.array(rmsd_values)
                    rmsd_zscores = stats.zscore(rmsd_array)
                    print("rmsd_values:", rmsd_array.shape)
                    print("rmsd_zscores:", rmsd_zscores.shape)
                    ind = 0
                    for zscore in rmsd_zscores:
                        # z-score of -3 marks outlier for a normal distribution.
                        # Assuming Normal Distribution of RMSD values because 
                        # CVAE yields normally distributed clusters.
                        if zscore <= -3:
                            print("RMSD to native contact for clustered outlier at index %i :" % path_to_pdb[ind][1], rmsd_values[ind])
                            pdb_stack.append(path_to_pdb[ind][0]) 
                        ind += 1
       
            print("PDB files left to investigate:", len(pdb_stack))
            # Base line for RL
            rmsd_threshold -= 0.40
        #END for     
        
        # Paint with RMSD to native state
        rmsd_values = []
        for i in range(0, self.iterations):
            for j in range(0, self.sim_num):   
                for k in range(0, self.sim_steps/self.traj_out_freq):
                    path = "./results/iteration_rl_%i/sim_%i_%i/pdb_data/output-%i.pdb" % (i, i, j, k) 
                    u = mdanal.Universe(path)
                    R = RMSD(u, self.native_protein)
                    R.run()
                    rmsd_values.append(R.rmsd[0,2])

        path = "./results/final_output/intermediate_data/"
        # Get data saved during RL iterations.
        all_encoded_data = get_all_encoded_data(path, self.iterations - 1)
        print("Final encoded data shape:", all_encoded_data.shape)
        scatter_plot(all_encoded_data, 'Latent Space (Before Clustering)', "./results/final_output/scatter.png")

        # Compute DBSCAN
        db = DBSCAN(eps=d_eps, min_samples=d_min_samples).fit(all_encoded_data)
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        print(Counter(db.labels_))
        # DBSCAN cluster plot
        scatter_plot(all_encoded_data, 
                     'Latent Space (Number of Clusters: %d, Params: eps=%.2f, min_samples=%i)' % (n_clusters_, d_eps, d_min_samples),
                     "./results/final_output/dbscan_clusters.png", color=db.labels_)
         
        # RMSD to native state plot
        scatter_plot_rmsd(all_encoded_data, 
                          "Final Latent Space", 
                          './results/final_output/rmsd_native_clusters.png',
                          rmsd_values)	
        # ALT: Could load full encoded_data and then set int_encoded_data to portions of it each loop iteration.
        for i in range(0, self.iterations):
            print(i)
            int_encoded_data = get_all_encoded_data(path, i)
            int_rmsd_data = rmsd_values[:self.sim_num*(self.sim_steps/self.traj_out_freq)*(i + 1)]
            print("int_encoded_data:", len(int_encoded_data))
            print("int_rmsd_data:", len(int_rmsd_data))
            db = DBSCAN(eps=d_eps, min_samples=d_min_samples).fit(int_encoded_data)
            n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            print('Estimated number of clusters: %d' % n_clusters_)
            print(Counter(db.labels_))
            scatter_plot(int_encoded_data,
                         'Intermediate Latent Space (Number of Clusters: %d, RL Loop: %i)' % (n_clusters_, i),
                         path + "dbscan_clusters_rl_%i.png" % i, 
                         color=db.labels_)

            scatter_plot_rmsd(int_encoded_data,
                              "Intermediate Latent Space (RL Loop: %i)" % i,
                               path + "cluster_rmsd_rl_%i.png" % i,
                               rmsd_values=int_rmsd_data,
                               vmin=min(rmsd_values),
                               vmax=max(rmsd_values))

        print("PDB files left to investigate:", len(pdb_stack))
        
# Script for testing
#rl = RL(cvae_weights_path="../model_150.dms", iterations=5, sim_num=5)
#rl.execute()