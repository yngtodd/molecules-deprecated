import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import grab_file
#file_to_load_1 = './layer_output_conv1_th.out'
#file_to_load_2 = './layer_output_conv1_tf.out'
#file_to_load_3 = './layer_output_conv1_pt.out'

class GenerateImages(object):
    def __init__(self):
        pass
    def generate_image(file_to_load= None, n=10, pick=0, row_dim=24, col_dim=24):
    
        """
        file_to_load : str
            .out file compatable with numpy.loadtxt() function
        n : int
            n = 10  # how many digits we will display
        pick : int
           pick = 0  # what image to pick 
       row_dim : int 
       col_dim : int 
       """

        if(file_to_load == None):
            raise ValueError("Must input file_to_load.")
        if (not os.path.exists(file_to_load)):
            raise Exception("Path " + str(file_to_load) + " does not exist!")
        if(n < 0):
            raise Exception("Invalid input: n must be greater than 0!")
        if(pick < 0):
            pass # TODO: implement exception
            # raise Exception("Invalid input: pick must be greater than 0!")
        if(row_dim < 0):
            raise Exception("Invalid input: row_dim must be greater than 0!")
        if(col_dim < 0):
            raise Exception("Invalid input: col_dim must be greater than 0!")

        load_file = np.loadtxt(file_to_load)
    
        if(file_to_load[-6:-4] == 'pt'):
            load_file = load_file/100
    
        print("Load file shape:", np.shape(load_file))

        cmi = plt.get_cmap('jet');
        cNorm = mpl.colors.Normalize(vmin=0, vmax=4);
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi);

        plt.figure(figsize=(20, 4))
        for i in range (n):
            # display reconstruction
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(load_file[i+pick].reshape(row_dim, col_dim))
            #np.savetxt('./imgs/decoded_imgs_%i.out' % i, (conv_full[i+pick].reshape(row_dim, col_dim)))
            scalarMap.set_array(load_file);
            plt.colorbar(scalarMap);
            #plt.colorbar(orientation='vertical')
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
       #plt.show()
        save_file = grab_file(file_to_load)
        plt.savefig("./images/" + save_file + "png", dpi=600)
        plt.clf()

