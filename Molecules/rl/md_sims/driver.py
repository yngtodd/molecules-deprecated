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
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')
    plt.savefig(save_path)
    plt.clf()

class RL(object):

    def __init__(self, cvae_weights_path, iterations=10, sim_num=10, sim_steps=20000, traj_out_freq=100, initial_pdb=None):
	if initial_pdb == None:
	     # For testing purposes
	     self.initial_pdb = '/home/a05/data/fs-peptide/raw_MD_data/fs-peptide.pdb'
	else:
	     self.initial_pdb = initial_pdb
        
	self.iterations = iterations
	self.sim_num = sim_num
        self.sim_steps = sim_steps
	self.traj_out_freq = traj_out_freq

        if not os.path.exists(cvae_weights_path):
            raise Exception("Path " + str(cvae_weights_path) + " does not exist!")
        self.cvae_weights_path = cvae_weights_path

	if not os.path.exists("./results"):
            os.mkdir("./results", 0755)
    
    def run_simulation(self, path, out_pdb_file, out_dcd_file, ff='amber14-all.xml', water_model='amber14/tip3pfb.xml'):
        if not os.path.exists(path):
	    raise Exception("Path: " + str(path) + " does not exist!")

	# TODO: Add other parameters for simulation + exception handling
	pdb_file = self.initial_pdb
	pdb = PDBFile(pdb_file)
	#print "Load PDB"
	forcefield = ForceField(ff, water_model)
	#print "Define force field"
	# nonBondedeMethod=PME
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
			      	             nonbondedCutoff=1.0*nanometer, constraints=HBonds)
	#print "Define system"
	integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
	#print "Define integrator"
	simulation = Simulation(pdb.topology, system, integrator)
	#print "Define simulation"
	# Start back from output.pdb last frame
	simulation.context.setPositions(pdb.positions)
        #print "Set positions"
	simulation.minimizeEnergy()
	#print "Minimize energy"
	simulation.reporters.append(PDBReporter(path + out_pdb_file, self.sim_steps))
	simulation.reporters.append(DCDReporter(path + out_dcd_file, self.traj_out_freq))
	#print "PDB report"
	#simulation.reporters.append(StateDataReporter(stdout, 100, step=True,
	#        potentialEnergy=True, temperature=True))
	#print "State data report"
	simulation.step(self.sim_steps)
	#print "Define steps"

    def execute(self):
	pdb_file = 'output.pdb'
	dcd_file = 'output-1.dcd'
	scatter_data = []
	# Put DCD reporter in a loop and put only a fixed number (10000) frames
	# in each output-i.dcd file. Where i ranges from (1,n).
	for i in range(1, self.iterations + 1):
	    path = "./results/iteration_rl_"
	    if not os.path.exists(path + "%i" % i):
            	os.mkdir(path + "%i" % i, 0755)
	    #reward_data = np.array([])
	    for j in range(1, self.sim_num + 1):
		path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
		if not os.path.exists(path_1):
                    os.mkdir(path_1, 0755)
		    os.mkdir(path_1 + "/cluster", 0755)
		# TODO: Optimize so that the simulation jobs are split over
		#       the available GPU nodes. May be possible with python
		#	subprocess. It would be a good idea to pull 
		#	self.run_simulation(path_1) out of the inner for loop.
		self.run_simulation(path_1, pdb_file, dcd_file)
	   
	    # Calculate contact matrix .array and .dat files for each simulation
	    # run. Files are place in native-contact/data inside each simulation
	    # directory.
	    for j in range(1, self.sim_num + 1):
		path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
		cm = ExtractNativeContact(path_1, pdb_file, dcd_file)
		cm.generate_contact_matrix()
	   	 
	    # Process contact matrix with CVAE algorithm for each simulation.
            # Requires pre-trained CVAE.
	    # TODO: compile CVAE outside of loop and pass in weights.
	    # 	    then pass in cont-mat files on the fly and update the data.
	    for j in range(1, self.sim_num + 1):
		path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
                cvae = CVAE(path=path_1, sep_train=0, sep_test=0, sep_pred=1, f_traj=self.sim_steps/self.traj_out_freq)
                cvae.load_contact_matrix(path_1 + "native-contact/data/cont-mat.dat",
                                         path_1 + "native-contact/data/cont-mat.array")
		cvae.compile()
                cvae.load_weights(self.cvae_weights_path)
                encoded_data = cvae.encode_pred()

		# Clustering
                print("Encoded data shape:",encoded_data.shape)
		np.save(path_1 + "/cluster/encoded_data.npy", encoded_data)
	 	scatter_data.append(encoded_data)
		scatter_plot(encoded_data, 'Latent Space (Before Clustering)', path_1+"/cluster/scatter.png")	
		 # Compute DBSCAN
        	db = DBSCAN(eps=0.1, min_samples=3).fit(encoded_data)
        	n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
		print('Estimated number of clusters: %d' % n_clusters_)
		print(Counter(db.labels_))
		colors = db.labels_
		scatter_plot(encoded_data, 'Latent Space (Number of Clusters: %d, Parameters: eps=%.2f, min_samples=%i)' % (n_clusters_, 0.1, 3), path_1+"/cluster/clusters.png", color=colors)

	        # Generate contact matrix
	        # Pass CM's to CVAE
	        # Evaluate reward function
	        # Kill some models and spawn new ones 
	if not os.path.exists("./results/final_output"):
	    os.mkdir("./results/final_output")
	all_encoded_data = np.array(scatter_data)
	all_encoded_data = np.reshape(all_encoded_data, (all_encoded_data.shape[0] * all_encoded_data.shape[1], all_encoded_data.shape[-1]))
	np.save("./results/final_output/all_encoded_data.npy", all_encoded_data)
	print("Final encoded data shape:", all_encoded_data.shape)	
	scatter_plot(all_encoded_data, 'Latent Space (Before Clustering)', "./results/final_output/scatter.png")	

	# Compute DBSCAN
        db = DBSCAN(eps=0.1, min_samples=3).fit(all_encoded_data)
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        print(Counter(db.labels_))
        colors = db.labels_
        scatter_plot(all_encoded_data, 'Latent Space (Number of Clusters: %d, Parameters: eps=%.2f, min_samples=%i)' % (n_clusters_, 0.1, 3), "./results/final_output/clusters.png", color=colors)


rl = RL(cvae_weights_path="../model_150.dms", iterations=1, sim_num=4)
rl.execute()
