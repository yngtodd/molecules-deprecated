import os
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

class RL(object):

    def __init__(self, sim_num=1000, iterations=10, initial_pdb=None):
	if initial_pdb == None:
	    # For testing purposes
	    self.initial_pdb = '/home/a05/data/fs-peptide/raw_MD_data/fs-peptide.pdb'
	else:
	    self.initial_pdb = initial_pdb
	
	self.sim_num = sim_num
	self.iterations = iterations
	if not os.path.exists("./results"):
            os.mkdir("./results", 0755)
    
    def run_simulation(self, path, ff='amber14-all.xml', water_model='amber14/tip3pfb.xml'):
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
	simulation.reporters.append(PDBReporter(path + 'output.pdb', 100))
	simulation.reporters.append(DCDReporter(path + 'output.dcd', 100))
	#print "PDB report"
	#simulation.reporters.append(StateDataReporter(stdout, 100, step=True,
	#        potentialEnergy=True, temperature=True))
	#print "State data report"
	simulation.step(1000)
	#print "Define steps"

    def execute(self):
	# Counting purpose
	for i in range(1, self.iterations + 1):
	    path = "./results/iteration_rl_"
	    if not os.path.exists(path + "%i" % i):
            	os.mkdir(path + "%i" % i, 0755)
	    for j in range(1, self.sim_num + 1):
		path_1 = path + "%i/sim_%i_%i/" % (i,i,j)
		if not os.path.exists(path_1):
                    os.mkdir(path_1, 0755)
		self.run_simulation(path_1)
	    # Generate contact matrix
	    # Pass CM's to CVAE
	    # Evaluate reward function
	    # Kill some models and spawn new ones    	

rl = RL(sim_num=4, iterations=2)
rl.execute()
