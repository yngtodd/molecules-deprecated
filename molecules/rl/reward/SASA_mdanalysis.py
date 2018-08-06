
# coding: utf-8

# In[ ]:


import MDAnalysis
import numpy as np


# In[ ]:


from MDAnalysis import *
u = Universe('./fs-peptide.pdb', './trajectory-1.xtc')


# In[ ]:


type(u.trajectory)


# In[ ]:


from __future__ import print_function
import numpy as np
import mdtraj as md
import MDAnalysis
from MDAnalysis import *

u = Universe('./fs-peptide.pdb', './trajectory-1.xtc')
xyz_an = u.trajectory[-1].positions

last_frame = md.load_frame('./trajectory-1.xtc', 9999, top='./fs-peptide.pdb')
xyz_traj = last_frame.xyz[0]

print("an:",type(xyz_an))
print("traj:", type(xyz_traj))
print("an:",xyz_an.shape)
print("traj:", xyz_traj.shape)

try:
    print(xyz_an == xyz_traj)
except:
    pass
print (xyz_an)
print ('\n\nnext\n\n')
print (xyz_traj)


# In[ ]:


from MDAnalysis import *
import numpy as np
from mdtraj.geometry import _geometry
from mdtraj.utils import ensure_type
_ATOMIC_RADII = {'C': 0.15,  'F': 0.12,  'H': 0.04,
                 'N': 0.110, 'O': 0.105, 'S': 0.16,
                 'P': 0.16}


def shrake_rupley(uni, probe_radius=0.14, n_sphere_points=960): # mode = 'atom'
    
    xyz = uni.trajectory[-1].positions
    xyz = np.array([xyz])
    #xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='traj.xyz', shape=(None, None, 3), warn_on_cast=False)
    #  if (xyz.shape != (None, None, 3)):
    #      raise Exception("Shape of xyz is "+str(xyz.shape)+" ... should be (None, None, 3)")
    #if (xyz.dtype != 'float32'):
    #    xyz = xyz.astype('float32')
    
    #if mode == 'atom':
    dim1 = xyz.shape[1]
    atom_mapping = np.arange(dim1, dtype=np.int32)
    #elif mode == 'residue':
        #dim1 = traj.n_residues
        #atom_mapping = np.array(
        #    [a.residue.index for a in traj.top.atoms], dtype=np.int32)
        #if not np.all(np.unique(atom_mapping) ==
        #              np.arange(1 + np.max(atom_mapping))):
        #   raise ValueError('residues must have contiguous integer indices '
        #                     'starting from zero')
    #else
    #    raise ValueError('mode must be one of "residue", "atom". "%s" supplied' %
    #                    mode)

    out = np.zeros((xyz.shape[0], dim1), dtype=np.float32)
    atom_radii = [_ATOMIC_RADII[atom.type] for atom in u.atoms]
    radii = np.array(atom_radii, np.float32) + probe_radius

    _geometry._sasa(xyz, radii, int(n_sphere_points), atom_mapping, out)

    return out

    
#test = shrake_rupley(u)


# In[ ]:


def calc(uni):
    sasa = shrake_rupley(uni)
    
    print(sasa)
    print('sasa data shape', sasa.shape)
    
    total_sasa = sasa.sum(axis=1)
    print('sasa values per frame', total_sasa.shape)
    print('total sasa:',total_sasa)


# In[ ]:


u = Universe('./fs-peptide.pdb', './trajectory-1.xtc')
#xyz_an = u.trajectory[-1].positions
calc(u)

