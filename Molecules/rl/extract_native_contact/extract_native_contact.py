# coding: utf-8
 
# In[36]:
 
#get_ipython().magic(u'matplotlib inline')
import gzip;
import os, sys;
import numpy as np;
import matplotlib.pyplot as plt;
plt.style.use('ggplot');
from math import sqrt 
import glob
def count_traj_files(path, extension):
    if not os.path.exists(path):
	raise Exception ("Path "+ str(path) + " does not exist.")
    """
    path : string
        Path of directory containing trajectory files.
    extension : string
        File extension type for trajectory files.
        EX) 'dcd', 'xtc', ...
    """
    return len(glob.glob1(path,"*."+extension)) 
# In[37]:
 
# specify path for trajectory+pdb files;
path_data = "/home/a05/Package_6_22/raw_MD_data/original/";
# create directories for results;
path_0 = "./results/";
path_1 = "./results/native-contact/";
path_2 = "./results/native-contact/raw/";
path_3 = "./results/native-contact/data/";
 
 
# In[38]:
 
# creating directories for results;
if not os.path.exists(path_0):
   os.mkdir(path_0, 0755);
if not os.path.exists(path_1):
   os.mkdir(path_1, 0755);
if not os.path.exists(path_2):
   os.mkdir(path_2, 0755);
if not os.path.exists(path_3):
   os.mkdir(path_3, 0755);
print "directories created or if already exists - then checked";
 
 
# In[39]:
 
# calculate native contacts & contact map;
import MDAnalysis as mdanal;
from MDAnalysis.analysis import contacts;
# define parameters;
# number of trajectories;
n = count_traj_files(path_data, 'xtc');

# number of frames per trajectories;
#f = int(0.01*10000);
# for file naming purpose;
k = 0;
# end define parameters
# calculate contact map over frames;
for i in range(1, (n+1)):    
    # specify path of structure & trajectory files;    
    print "Creating Universe"
    u0 =mdanal.Universe(path_data + '100-fs-peptide-400K.pdb', path_data + 'trajectory-%i.xtc' % i);
    f = len(u0.trajectory);
    print('trajectory no:'), i;
    print('number of frames'), f;
    # crude definition of salt bridges as contacts between CA atoms;
    #CA = "(name CA and resid 237-248 283-288 311-319 345-349 394-399)";
    #CA = "(name CA and resid 42:76)";
    CA = "(name CA and resid 1:24)";
    #CA = "(name CA and resid 42:76)";
    CA0 = u0.select_atoms(CA);
    print "Defining carbon alphas"
    #CA0 = u0.select_atoms(CA);
    # print progress;
#    print('read user defined atoms for frames:'), k;
    # calculate contact map over all frames; 
    for j in range(0, (f)):
        # calculating and saving native contact dat files per frame;     
        # set up analysis of native contacts ("salt bridges"); salt bridges have a distance <8 Angstrom;
        ca = contacts.ContactAnalysis1(u0, selection=(CA, CA), refgroup=(CA0, CA0), radius=8.0, 
                                       outfile= path_2 + 'cont-mat_%i.dat' % k)    
        ca.run(store=True, start=j, stop=j+1, step=1);
        # save ncontact figures per frame or function of residues;
        #ca.plot_qavg(filename="./fig_res/ncontact_res_%i.pdf" % k);
        # save ncontact over time;
        #ca.plot(filename="./fig_frame/ncontact_time_%i.pdf" % k);
        # read zipped native contact array files;
        inF_array = gzip.GzipFile(path_2 + "cont-mat_%i.array.gz" % k, 'rb');   
        s_array = inF_array.read();
 #       print s_array
	inF_array.close();
	#print(s_array)
	arr = s_array
	# Test
        arr = np.fromstring(s_array, dtype='float32', sep=' ')
#	print(arr)
        arr = np.reshape(arr, (int(sqrt(arr.shape[0])), int(sqrt(arr.shape[0]))))
	for i in range(0, arr.shape[0]):
    	    arr[i][i] = 0.
	    if i == arr.shape[0] - 1:
		break
	    else:
		arr[i][i+1] = 0.
		arr[i+1][i] = 0.
	temp = ''
	for ind in range(0, arr.shape[0]):
	    for inj in range(0, arr.shape[0]):
	        temp += str( arr[ind][inj])    
	   	temp += ' '
	    temp += '\n'
	s_array = temp
	#arr = np.reshape(arr,(arr.shape[0]**2)) 
	#s_array = np.array_str(arr)
	#s_array = s_array[1:-1]
	#s_array += ' '
	#s_array = s_array[1:int(sqrt(arr.shape[0]))]
	#s_array += '\n'
#	print('\n\n\n')
#	print(s_array)
	# END Test	
        # copy to another file;
        outF_array = file(path_2 + "cont-mat_%i.array" % k, 'wb');
        outF_array.write(s_array);
        outF_array.close(); 
        # remove zipped array file;
        os.remove(path_2 + "cont-mat_%i.array.gz" %k);
        # to next file name numerics;
        k += 1;
    print('read user defined atoms for frames:'), k;
 
# In[40]:
 
# create one contact map from all contact map files;
# for counting purpose;
l = 0;
for i in range(0, k):
    if i==10000*l:
        print "compressing frame:", i;
        l+= 1;
    fin = open(path_2 + "cont-mat_%i.array" % i, "r")
    data1 = fin.read()
    fin.close()
    fout = open(path_3 + "cont-mat.array", "a")
    fout.write(data1)
    fout.close() 
print "contact map file created";
# create one native contact from all native contact files;
# for counting purpose;
l = 0;
for i in range(0, k):
    if i==10000*l:
        print "compressing frame:", i;
        l+= 1;
    fin = open(path_2 + "cont-mat_%i.dat" % i, "r")
    data1 = fin.read()
    fin.close()
    fout = open(path_3 + "cont-mat.dat", "a")
    fout.write(data1)
    fout.close() 
print "native contact file created";
 
 
# In[43]:
 
# plot histogram of native contacts;    
#dat_check = np.loadtxt(path_3 + 'cont-mat.dat');
#[nhist, shist] = np.histogram(dat_check[ : ,1], 25);
#plt.semilogy(shist[1: ], nhist, 'r-');
#plt.savefig(path_1+'native-contact.png', dpi=600);
#plt.show();
#plt.clf();
 
 
# In[44]:
 
# check contact map shape;
map_check = np.loadtxt(path_3 + 'cont-mat.array');
print type(map_check)
print "contact map shape:", np.shape(map_check)
 
 
# In[ ]:
 
 
 
 
# In[ ]:
 
 
 
 
# In[ ]:
 
 
 
 
# In[ ]:
