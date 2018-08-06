from __future__ import print_function
import numpy as np
from sklearn import manifold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

class Isomap(object):

    def __init__(self, n_traj=10, f_traj=10000, sep_train=0.8, 
                 sep_test=0.9, sep_pred=1, choice=0, row=198, col=198):
    
        if(n_traj < 0):
            raise Exception("Invalid input: n_traj must be greater than 0!")
        if(f_traj < 0):
            raise Exception("Invalid input: f_traj must be greater than 0!")
        if(sep_train < 0 or sep_train > 1):
            raise Exception("Invalid input: sep_train must be in range 0 to 1!")
        if(sep_pred < 0):
            raise Exception("Invalid input: sep_pred must be greater than 0!")
        if(choice != 0 and choice != 1):
            raise Exception("Invalid input: choice must be 0 or 1!")
        if(row < 0):
            raise Exception("Invalid input: row must be greater than 0!")
        if(col < 0):
            raise Exception("Invalid input: col must be greater than 0!")

   	    # no of trajectory files and frames in each file
        self.n_traj = n_traj
        self.f_traj = f_traj
    	# fraction of train, test and pred data separation 
    	self.sep_train = sep_train
    	self.sep_test = sep_test
    	self.sep_pred = sep_pred
	    # choice to flatten data: "0" for NO & "1" for YES
	    self.choice = choice
	    # row and column dimension for each frame
	    self.row = row
	    self.col = col

    def build_array(self, path_data_array="./input_data/native-contact/data/cont-mat.array"):
        if (not os.path.exists(path_data_array)):
            raise Exception("Path " + str(path_data_array) + " does not exist!")
    	# read dat type large file line by line to save in array
    	nf = self.n_traj * self.f_traj
    	q = self.row * self.col
    	j_count = 0
    	k_count = 0
    	samples = (nf)
    	row_num = (nf) * self.row
    	column_num = (self.col)
    	array_f_int = np.zeros(shape=(row_num,column_num))
    	with open(path_data_array) as infile:
            for line in infile:
                array_f_string = line.split()
            	array_f_array = np.array(list(array_f_string), dtype='|S4')
            	array_f_float = array_f_array.astype(np.float)
            	array_f_int[j_count] = array_f_float
            	if j_count == k_count:
                    print("Frames read:", (j_count/row))
                    k_count = k_count + 10000*row
            	j_count = j_count + 1
                if j_count == (row_num):
                    break
        print("Initial matrix array dimension:", np.shape(array_f_int))
    	array_f = np.reshape(array_f_int, (samples, self.row, self.col))
    	print("Final matrix array dimension:", np.shape(array_f))
    	x_raw = array_f[0:80000]
    	print("Dataset dimension:", np.shape(x_raw))
    	print(np.shape(x_raw))
    	np.save("./output_data/protein_80000.npy", x_raw)

    def build_manifold(self):

    	x_train = np.load("./output_data/protein_80000.npy")
        print(np.shape(x_train))
        x_train_reshape = np.reshape((x_train), (len(x_train), (x_train.shape[1]*x_train.shape[2])))
        print(np.shape(x_train_reshape))
        x_train_reshape_mod = x_train_reshape[::20]
        print(np.shape(x_train_reshape_mod))
    	#iso = manifold.Isomap(n_neighbors=6, n_components=3)
    	#iso.fit(x_train_reshape)
    	#manifold_2D = iso.transform(x_train_reshape)

    	# Left with 2 dimensions
    	#print np.shape(manifold_2D)

    	iso = manifold.Isomap(n_neighbors=6, n_components=3)
    	iso.fit(x_train_reshape_mod)
    	manifold_2D_mod = iso.transform(x_train_reshape_mod)

    	# Left with 2 dimensions
    	print(np.shape(manifold_2D_mod))
    	np.save("./output_data/manifold_3Disomap_mod_4000.npy", manifold_2D_mod)

    def compute_TSNE(self):

        X = np.load("./output_data/manifold_3Disomap_mod_4000.npy")
        print(X.shape)

    	X1 = X#X[::10]
    	X_embedded = TSNE(n_components=2).fit_transform(X1)
    	print("After TSNE operation: embedded shape", X_embedded.shape)

    	np.save("./output_data/encoded_TSNE_2D_isomap_4000.npy", X_embedded)

    def build_training_array(self):
        X_embedded = np.load("./output_data/encoded_TSNE_2D_isomap_4000.npy")
    	print(X_embedded.shape)

    	#label = np.loadtxt("./cont-mat.dat")
    	label = np.load("./output_data/dist-mat.npy")
    	sep_1 = 80000#len(X)
    	print(sep_1)
    	label = label[0:int(sep_1)]
   	    self.y_train_c = label[::20,0]
    	#y_train_t = label[::10,0]
    	self.x_pred_encoded = X_embedded

    	print(self.y_train_c.shape)

    def plot1(self):
    	# plot 1: 
    	Dmax = self.y_train_c
    	[n,s] = np.histogram(Dmax, 11) 
    	d = np.digitize(Dmax, s)
    	#[n,s] = np.histogram(-np.log10(Dmax), 11)
    	#d = np.digitize(-np.log10(Dmax), s)
    	cmi = plt.get_cmap('jet')
   	    cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax))
   	    #cNorm = mpl.colors.Normalize(vmin=140, vmax=240)
   	    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi)
   	    fig = plt.figure()
   	    ax = fig.add_subplot(111, projection='3d')
   	    # scatter3D requires a 1D array for x, y, and z
   	    # ravel() converts the 100x100 array into a 1x10000 array
   	    p = ax.scatter3D(np.ravel(self.x_pred_encoded[:, 0]),
                     	  np.ravel(self.x_pred_encoded[:, 1]),
                     	  #np.ravel(x_pred_encoded[:, 2]), 
                     	  marker='o', c=scalarMap.to_rgba(Dmax))
    	ax.set_xlim3d(np.amin(np.ravel(self.x_pred_encoded[:, 0])), np.amax(np.ravel(self.x_pred_encoded[:, 0])))
    	ax.set_ylim3d(np.amin(np.ravel(self.x_pred_encoded[:, 1])), np.amax(np.ravel(self.x_pred_encoded[:, 1])))
    	#ax.set_zlim3d(np.amin(np.ravel(x_pred_encoded[:, 2])), np.amax(np.ravel(x_pred_encoded[:, 2])))
    	ax.set_xlabel('VAE 0')
    	ax.set_ylabel('VAE 1')
    	ax.set_zlabel('VAE 2')
    	scalarMap.set_array(Dmax)
    	fig.colorbar(scalarMap)
    	#plt.show()
    	plt.savefig('./images/encoded_train_2D_a_isomap_4000.png', dpi=600)
    	plt.clf()

    def plot2(self):

    	Dmax = self.y_train_c
    	[n,s] = np.histogram(Dmax, 11) 
    	d = np.digitize(Dmax, s)
    	cmi = plt.get_cmap('jet')
    	cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax))
    	#cNorm = mpl.colors.Normalize(vmin=140, vmax=240)
    	scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi)

    	x = self.x_pred_encoded[:,0]
    	y = self.x_pred_encoded[:,1]
    	#z = y_train_2
	
   	    plt.scatter(x, y, c=scalarMap.to_rgba(Dmax), marker="o", alpha=0.5)

    	scalarMap.set_array(Dmax)
    	plt.colorbar(scalarMap)
    	#plt.show()
    	plt.savefig('./images/encoded_train_2D_b_isomap_4000.png', dpi=600)
    	plt.clf()


