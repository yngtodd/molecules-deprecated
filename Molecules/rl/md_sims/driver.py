import os
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import sys
sys.path.append('../')
from extract_native_contact.extract_native_contact import ExtractNativeContact 

class RL(object):

    def __init__(self, iterations=10, sim_num=10, initial_pdb=None):
	if initial_pdb == None:
	    # For testing purposes
	    self.initial_pdb = '/home/a05/data/fs-peptide/raw_MD_data/fs-peptide.pdb'
	else:
	    self.initial_pdb = initial_pdb
	
	self.iterations = iterations
	self.sim_num = sim_num
	if not os.path.exists("./results"):
            os.mkdir("./results", 0755)
    
    def run_simulation(self, path, out_pdb_file, out_dcd_file, steps=1000, traj_out_freq=100, ff='amber14-all.xml', water_model='amber14/tip3pfb.xml'):
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
	simulation.reporters.append(PDBReporter(path + out_pdb_file, steps))
	simulation.reporters.append(DCDReporter(path + out_dcd_file, traj_out_freq))
	#print "PDB report"
	#simulation.reporters.append(StateDataReporter(stdout, 100, step=True,
	#        potentialEnergy=True, temperature=True))
	#print "State data report"
	simulation.step(steps)
	#print "Define steps"

    def execute(self):
	pdb_file = 'output.pdb'
	dcd_file = 'output-1.dcd'
	for i in range(1, self.iterations + 1):
	    path = "./results/iteration_rl_"
	    if not os.path.exists(path + "%i" % i):
            	os.mkdir(path + "%i" % i, 0755)
	    for j in range(1, self.sim_num + 1):
		path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
		if not os.path.exists(path_1):
                    os.mkdir(path_1, 0755)
		# TODO: Optimize so that the simulation jobs are split over
		#       the available GPU nodes. May be possible with python
		#	subprocess. It would be a good idea to pull 
		#	self.run_simulation(path_1) out of the inner for loop.
		self.run_simulation(path_1, pdb_file, dcd_file)

	    for j in range(1, self.sim_num + 1):
		path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
		cm = ExtractNativeContact(path_1, pdb_file, dcd_file)
		cm.generate_contact_matrix()
	    # Generate contact matrix
	    # Pass CM's to CVAE
	    # Evaluate reward function
	    # Kill some models and spawn new ones    	

rl = RL(iterations=2, sim_num=4)
rl.execute()
