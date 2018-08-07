from __future__  import print_function
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.kde import gaussian_kde


class TSNE_2D(object):
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

        self.X = np.loadtxt(latent_array_path) # this is the latent space array
        print("Load data shape:", self.X.shape)

        X1 = self.X[::10]
        X_embedded = TSNE(n_components=2).fit_transform(X1)
        print("After TSNE operation: embedded shape", X_embedded.shape)
        np.save("./output_data/encoded_TSNE_2D.npy", X_embedded)
        #X_embedded = np.load("./encoded_TSNE_2D.npy")
        self.x_pred_encoded = X_embedded
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

        label = np.loadtxt(cont_mat_path)
        sep_1 = len(self.X)
        print(sep_1)
        label = label[0:int(sep_1)]
        self.y_train_c = label[::10,2]
        y_train_t = label[::10,0]
        print(self.y_train_c.shape)
         
    def plot_a(self):
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
    
        plt.savefig('./images/encoded_train_2D_a.png', dpi=600)
        plt.clf()


    def plot_b(self):
        x = self.x_pred_encoded[:,0]
        y = self.x_pred_encoded[:,1]
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
     
        fig = plt.figure(figsize=(7,8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
   
        # alpha=0.5 will make the plots semitransparent
        ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1)
        ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=1)
 
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(y.min(), y.max())
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(y.min(), y.max())
    
        #plt.show()
        plt.savefig('./images/encoded_train_2D_b.png', dpi=600)
        # overlay with another figure (jpg)
        im = plt.imread('./images/encoded_train_2D_b.png')
        ax1.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
        ax2.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

