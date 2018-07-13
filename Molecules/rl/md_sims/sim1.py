from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

pdb_file = '/home/a05/data/E1B48/E4B048ww.pdb'
pdb = PDBFile(pdb_file)
print "Load PDB"
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
print "Define forcefield"
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=0.9*nanometer, constraints=HBonds)
print "Define system"
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
print "Define integrator"
simulation = Simulation(pdb.topology, system, integrator)
print "Define simulation"
simulation.context.setPositions(pdb.positions)
print "Set positions"
simulation.minimizeEnergy()
print "Minimize energy"
simulation.reporters.append(PDBReporter('output.pdb', 1000))
print "PDB report"
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True))
print "State data report"
simulation.step(10)
print "Define steps"

