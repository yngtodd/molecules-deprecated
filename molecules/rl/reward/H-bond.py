
# coding: utf-8

# In[3]:


get_ipython().magic(u'matplotlib inline')
from __future__ import print_function
import numpy as np
import mdtraj as md
t = md.load_frame('.\\trajectory-1.xtc', 9999, top='.\\fs-peptide.pdb')
print(t)


# In[7]:


hbonds = md.baker_hubbard(t, periodic=False)
label = lambda hbond : '%s -- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[2]))
for hbond in hbonds:
    print(label(hbond))
    
hbond_score = len(hbonds)
print('\n\nH-Bond score:', hbond_score)
# calculate heavy atom to atom contacts
# basically a number that looks like Nij/root(ninj)


# In[8]:


da_distances = md.compute_distances(t, hbonds[:, [0,2]], periodic=False)


# In[14]:


import itertools
import matplotlib.pyplot as plt
color = itertools.cycle(['r', 'b', 'gold'])
for i in [2, 3, 4]:
    plt.hist(da_distances[:, i], color=next(color), label=label(hbonds[i]), alpha=0.5)
plt.legend()
plt.ylabel('Freq');
plt.xlabel('Donor-acceptor distance [nm]')

