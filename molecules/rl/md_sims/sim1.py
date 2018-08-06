from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

pdb_file = '/home/a05/data/fs-peptide/raw_MD_data/fs-peptide.pdb'
pdb = PDBFile(pdb_file)
print "Load PDB"
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
print "Define forcefield"
# nonBondedeMethod=PME
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
        nonbondedCutoff=1.0*nanometer, constraints=HBonds)
print "Define system"
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
print "Define integrator"
simulation = Simulation(pdb.topology, system, integrator)
print "Define simulation"
# Start back from output.pdb last frame
simulation.context.setPositions(pdb.positions)
print "Set positions"
simulation.minimizeEnergy()
print "Minimize energy"
simulation.reporters.append(PDBReporter('output.pdb', 100))
simulation.reporters.append(DCDReporter('output.dcd', 100))
print "PDB report"
#simulation.reporters.append(StateDataReporter(stdout, 100, step=True,
#        potentialEnergy=True, temperature=True))
print "State data report"
simulation.step(1000)
print "Define steps"

