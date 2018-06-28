from __future__ import print_function
import numpy as np
from sklearn.manifold import TSNE;
import matplotlib.pyplot as plt
from matplotlib import cm;
import matplotlib as mpl;

# latent_array_path = "./encoded_train_80.out"
# label_path = "./../1FME-0_cont-mat.dat"

class FindIndex(object):
    def __init__(self, latent_array_path, label_path):
	
	    if(latent_array_path == None or label_path == None):
            raise ValueError("Must input latent_array_path and label_path as parameters.")
        if (not os.path.exists(latent_array_path)):
            raise Exception("Path " + str(latent_array_path) + " does not exist!")
        if (not os.path.exists(label_path)):
            raise Exception("Path " + str(label_path) + " does not exist!")

        self.latent_array_path = latent_array_path
        self.label_path = label_path

    def load_latent_array(self): 
    	self.X = np.loadtxt(self.latent_array_path); # this is the latent space array
    	print("Load data shape:", self.X.shape);

    def save_npy(self):
    	X1 = self.X[::10];
    	X_embedded = TSNE(n_components=2).fit_transform(X1);
    	print("After TSNE operation: embedded shape", X_embedded.shape);
    	np.save("./processed_data/encoded_TSNE_2D.npy", X_embedded)
	    print("Saved 'encoded_TSNE_2D.npy' in './processed_data' ")

    def get_data(self):
	    X_embedded = np.load("./processed_data/encoded_TSNE_2D.npy")
        print("Embedded shape:", X_embedded.shape)
    	label = np.loadtxt(self.label_path);
    	sep_1 = len(label[0:500000])*0.8
    	print(sep_1)
    	label = label[0:int(sep_1)]
    	self.y_train_c = label[::10,2];
    	y_train_t = label[::10,0];
    	self.x_pred_encoded = X_embedded;
    	print("y_train_c shape:", self.y_train_c.shape)

    def plot1():
    	Dmax = self.y_train_c;
    	[n,s] = np.histogram(Dmax, 11); 
    	d = np.digitize(Dmax, s);
    	cmi = plt.get_cmap('jet');
    	cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax));
    	#cNorm = mpl.colors.Normalize(vmin=140, vmax=240);
    	scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);

    	x = x_pred_encoded[:,0]
    	y = x_pred_encoded[:,1]
    	#z = y_train_2

    	plt.scatter(x, y, c=scalarMap.to_rgba(Dmax), marker=".", alpha=0.5)
	
    	scalarMap.set_array(Dmax);
    	plt.colorbar(scalarMap);
    	#plt.xlim([0, 5])
    	#plt.ylim([0, 5])
    	plt.show()

    	#plt.savefig('./encoded_train_2D_b1.png', dpi=600);
    	#plt.savefig('./encoded_train_2D_b2.png', dpi=600);
    	#plt.clf(); 


    def plot2():
    	Dmax = self.y_train_c;
    	[n,s] = np.histogram(Dmax, 11); 
    	d = np.digitize(Dmax, s)
    	cmi = plt.get_cmap('jet');
    	cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax));
    	#cNorm = mpl.colors.Normalize(vmin=140, vmax=240);
    	scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);

    	x = x_pred_encoded[:,0]
    	y = x_pred_encoded[:,1]
    	#z = y_train_2

    	plt.scatter(x, y, c=scalarMap.to_rgba(Dmax), marker=".", alpha=0.5)

    	scalarMap.set_array(Dmax);
    	plt.colorbar(scalarMap);
    	plt.xlim([-45, -35])
    	plt.ylim([-65, -60])
    	plt.show()

    	#plt.savefig('./encoded_train_2D_b1.png', dpi=600);
    	#plt.savefig('./encoded_train_2D_b2.png', dpi=600);
    	#plt.clf(); 

    def print_indices():
    	find = self.x_pred_encoded;
    	# print index of array satisfying condition;
    	print([np.where((find[:,0] > 10) & (find[:,0] < 15) & (find[:,1] < -40) & (find[:,1] > -45))])
    	# check;
    	print(find[475])

    	# print index of array satisfying condition;
    	print([np.where((find[:,0] > 71) & (find[:,0] < 75) & (find[:,1] < 15) & (find[:,1] > 5))])
    	# check;
    	print(find[17495])

    	# print index of array satisfying condition;
    	print([np.where((find[:,0] > 30) & (find[:,0] < 40) & (find[:,1] < 40) & (find[:,1] > 30))])
    	# check;
    	print(find[10269])
	
   	    # print index of array satisfying condition;
    	print([np.where((find[:,0] > -45) & (find[:,0] < -35) & (find[:,1] < -60) & (find[:,1] > -65))])
    	# check;
    	print(find[4099])