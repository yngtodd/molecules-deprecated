
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


X = np.loadtxt("./data/encoded_train_150.out"); # this is the latent space array
print "load data shape:", X.shape;


# In[3]:


#from sklearn.manifold import TSNE;
X1 = X[::10];
#X_embedded = TSNE(n_components=3).fit_transform(X1);
#print "after TSNE operation: embedded shape", X_embedded.shape;


# In[4]:


#np.save("./encoded_TSNE_3D.npy", X_embedded)


# In[5]:


#X_embedded = np.load("./encoded_TSNE_3D.npy")
X_embedded = X1
print X_embedded.shape


# In[6]:


label = np.load("./data/cont-mat_mod.npy");
sep_1 = len(label)*0.8
print sep_1
label = label[0:int(sep_1)]
y_train_c = label[::10,1];
y_train_c_frac = label[::10,2];
y_train_t = label[::10,0];
x_pred_encoded = X_embedded;


# In[7]:


label_2 = np.load("./output/RMSD.npy")
print sep_1
label_2 = label_2[0:int(sep_1)]
y_train_rmsd = label_2[::10,1];


# In[8]:


print y_train_c.shape
print y_train_c_frac.shape
print y_train_t.shape
print y_train_rmsd.shape


# In[18]:


# plot 1: 
import matplotlib.pyplot as plt;
plt.rcParams['axes.facecolor'] = 'black'

Dmax = y_train_c_frac;
[n,s] = np.histogram(Dmax, 11); 
d = np.digitize(Dmax, s);
#[n,s] = np.histogram(-np.log10(Dmax), 11); 
#d = np.digitize(-np.log10(Dmax), s);
from matplotlib import cm;
import matplotlib as mpl;
cmi = plt.get_cmap('jet');
cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax));
#cNorm = mpl.colors.Normalize(vmin=140, vmax=240);
scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);
import numpy as np;
from mpl_toolkits.mplot3d import Axes3D;
fig = plt.figure(figsize=(8.5,6));
ax = fig.add_subplot(111, projection='3d');
# scatter3D requires a 1D array for x, y, and z;
# ravel() converts the 100x100 array into a 1x10000 array;
p = ax.scatter3D(np.ravel(x_pred_encoded[:, 0]),
            np.ravel(x_pred_encoded[:, 1]),
           np.ravel(x_pred_encoded[:, 2]), 
            marker='.', c=scalarMap.to_rgba(Dmax));
ax.set_xlim3d(np.amin(np.ravel(x_pred_encoded[:, 0])), np.amax(np.ravel(x_pred_encoded[:, 0])));
ax.set_ylim3d(np.amin(np.ravel(x_pred_encoded[:, 1])), np.amax(np.ravel(x_pred_encoded[:, 1])));
ax.set_zlim3d(np.amin(np.ravel(x_pred_encoded[:, 2])), np.amax(np.ravel(x_pred_encoded[:, 2])));
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
#ax.xaxis.label.set_color('white')
#ax.yaxis.label.set_color('white')
#ax.zaxis.label.set_color('white')
#ax.set_xlabel('VAE 0');
#ax.set_ylabel('VAE 1');
#ax.set_zlabel('VAE 2');
scalarMap.set_array(Dmax);
fig.colorbar(scalarMap);
#plt.show()
plt.savefig('./fig/encoded_train_3D_frac.png', dpi=600);
plt.clf();


# In[19]:


# plot 1: 
import matplotlib.pyplot as plt;
plt.rcParams['axes.facecolor'] = 'black'

Dmax = y_train_rmsd;
[n,s] = np.histogram(Dmax, 11); 
d = np.digitize(Dmax, s);
#[n,s] = np.histogram(-np.log10(Dmax), 11); 
#d = np.digitize(-np.log10(Dmax), s);
from matplotlib import cm;
import matplotlib as mpl;
cmi = plt.get_cmap('jet');
cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax));
#cNorm = mpl.colors.Normalize(vmin=140, vmax=240);
scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);
import numpy as np;
from mpl_toolkits.mplot3d import Axes3D;
fig = plt.figure(figsize=(8.5,6));
ax = fig.add_subplot(111, projection='3d');
# scatter3D requires a 1D array for x, y, and z;
# ravel() converts the 100x100 array into a 1x10000 array;
p = ax.scatter3D(np.ravel(x_pred_encoded[:, 0]),
            np.ravel(x_pred_encoded[:, 1]),
           np.ravel(x_pred_encoded[:, 2]), 
            marker='.', c=scalarMap.to_rgba(Dmax));
ax.set_xlim3d(np.amin(np.ravel(x_pred_encoded[:, 0])), np.amax(np.ravel(x_pred_encoded[:, 0])));
ax.set_ylim3d(np.amin(np.ravel(x_pred_encoded[:, 1])), np.amax(np.ravel(x_pred_encoded[:, 1])));
ax.set_zlim3d(np.amin(np.ravel(x_pred_encoded[:, 2])), np.amax(np.ravel(x_pred_encoded[:, 2])));
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')
#ax.xaxis.label.set_color('white')
#ax.yaxis.label.set_color('white')
#ax.zaxis.label.set_color('white')
#ax.set_xlabel('VAE 0');
#ax.set_ylabel('VAE 1');
#ax.set_zlabel('VAE 2');
scalarMap.set_array(Dmax);
fig.colorbar(scalarMap);
#plt.show()
plt.savefig('./fig/encoded_train_3D_rmsd.png', dpi=600);
plt.clf();

