from __future__ import print_function
import numpy as np
from sklearn.manifold import TSNE;
import matplotlib.pyplot as plt;
from matplotlib import cm;
import matplotlib as mpl;
from mpl_toolkits.mplot3d import Axes3D;

class TSNE_3D(object):
    def __init__(self):
        pass

    # latent_array_path="./encoded_train_80.out"
    def load_latent_data(self, latent_array_path=None):
        """
         latent_array_path : string
            - path of the .out file. Should be located in ./output_data
        """
        if(latent_array_path == None):
            raise ValueError("Must input latent_array_path as parameter.")
        if (not os.path.exists(latent_array_path)):
            raise Exception("Path " + str(latent_array_path) + " does not exist!")

        self.X = np.loadtxt(latent_array_path); # this is the latent space array
        print("Load data shape:", self.X.shape);
   
        X1 = self.X[::10];
        X_embedded = TSNE(n_components=3).fit_transform(X1);
        print("After TSNE operation: embedded shape", X_embedded.shape)
        np.save("./output_data/encoded_TSNE_3D.npy", X_embedded)
        #X_embedded = np.load("./encoded_TSNE_3D.npy")
        self.x_pred_encoded = X_embedded;
        print(X_embedded.shape)

    # cont_mat_path="./../1FME-0_cont-mat.dat"
    def load_label_data(self, cont_mat_path=None):
        """
         cont_mat_path : string
            - path of the cont-mat.dat file. Should be located in ./input_data
        """
        if(latent_array_path == None):
            raise ValueError("Must input cont_mat_path as parameter.")
        if (not os.path.exists(cont_mat_path)):
            raise Exception("Path " + str(cont_mat_path) + " does not exist!")

        label = np.loadtxt(cont_mat_path);
        sep_1 = len(self.X)
        print sep_1
        label = label[0:int(sep_1)]
        self.y_train_c = label[::10,2];
        y_train_t = label[::10,0];
        print(self.y_train_c.shape)
 

    def plot(self):
        # plot 1: 
        Dmax = self.y_train_c;
        [n,s] = np.histogram(Dmax, 11); 
        d = np.digitize(Dmax, s);
        #[n,s] = np.histogram(-np.log10(Dmax), 11); 
        #d = np.digitize(-np.log10(Dmax), s);
        cmi = plt.get_cmap('jet');
        cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax));
        #cNorm = mpl.colors.Normalize(vmin=140, vmax=240);
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);
        fig = plt.figure();
        ax = fig.add_subplot(111, projection='3d');
        # scatter3D requires a 1D array for x, y, and z;
        # ravel() converts the 100x100 array into a 1x10000 array;
        p = ax.scatter3D(np.ravel(self.x_pred_encoded[:, 0]),
                         np.ravel(self.x_pred_encoded[:, 1]),
                         np.ravel(self.x_pred_encoded[:, 2]), 
                         marker='o', c=scalarMap.to_rgba(Dmax))
        ax.set_xlim3d(np.amin(np.ravel(self.x_pred_encoded[:, 0])), np.amax(np.ravel(self.x_pred_encoded[:, 0])));
        ax.set_ylim3d(np.amin(np.ravel(self.x_pred_encoded[:, 1])), np.amax(np.ravel(self.x_pred_encoded[:, 1])));
        #ax.set_zlim3d(np.amin(np.ravel(self.x_pred_encoded[:, 2])), np.amax(np.ravel(self.x_pred_encoded[:, 2])));
        ax.set_xlabel('VAE 0');
        ax.set_ylabel('VAE 1');
        ax.set_zlabel('VAE 2');
        scalarMap.set_array(Dmax);
        fig.colorbar(scalarMap);
        #plt.show()
        plt.savefig('./images/encoded_train_3D.png', dpi=600);
        plt.clf();

