import numpy as np;


class History(object):
    def __init__(self):
        pass
    def hist(self, nb_start=0, nb_end=400, backend='tf', model = 'cn', data = 'cm', acc = False):
 
        """
        nb_start = TODO
        nb_end = TODO
        backend = Either tensorflow 'tf' or theano 'th'
        model = Either CNN 'cn' or VAE 'v'
    	data = Either contact matrix 'cm' or coordinates 'c'
    	acc = If true, then acc and val_acc files will be added to the data array and saved 
    	"""
    
   	    # Exception handling
    	if nb_start < 0:
            raise Exception("nb_start must be greater than 0!")
    	if nb_end < nb_start:
	        raise Exception("nb_end must be greater than nb_start!")
    	if (backend != 'tf' and backend != 'th'):
	        raise Exception("backend must be 'tf' or 'th'!")
    	if (model != 'cn' and model != 'v'):
	        raise Exception("model must be 'cn' or 'v'!")
     	if (data != 'cm' and data != 'c'):
	        raise Exception("data must be 'cm' or 'c'!")

    	# Parameters
    	mod = "convnet"
    	dat = "cont-mat"
    	back = "tensorflow"
    	dim = 3
    	if model == 'v':
	        mod = "vae"
    	if data == 'c':
	        dat = "coor"
    	if backend == 'th':
	        back = "theano"
    	if acc == True:
	        dim = 5
        loss_path = "./history/%s_%s/hist_%s/history."% (mod, dat, back)

    	# Matrix creation
    	hist = np.zeros(((nb_end-nb_start), dim))
    	for i in range((nb_start+1), (nb_end+1)):
            hist_loss = np.loadtxt(loss_path+"losses_%i" %i)
	        hist_val_loss = np.loadtxt(loss_path+"val_losses_%i" %i)
	        if acc == True:
                hist_acc = np.loadtxt(loss_path+"acc_%i" %i)
            	hist_val_acc = np.loadtxt(loss_path+"val_acc_%i" %i)
            	tmp = np.array([i, hist_loss, hist_val_loss, hist_acc, hist_val_acc]);
            else:
	    	    tmp = np.array([i, hist_loss, hist_val_loss])

            hist[i-1] = tmp;
    	np.savetxt("history_%s_%s_%s.dat"% (mod, dat, backend), hist, delimiter = ' ')


    def calculate_all(self):
        hist(nb_start=0, nb_end=400, backend='tf', model = 'cn', data = 'cm', acc = True)
	    hist(nb_start=0, nb_end=400, backend='tf', model = 'cn', data = 'c', acc = True)
	    hist(nb_start=0, nb_end=325, backend='th', model = 'cn', data = 'cm', acc = True)
	    hist(nb_start=0, nb_end=400, backend='th', model = 'cn', data = 'c', acc = True)
	    hist(nb_start=0, nb_end=80, backend='tf', model = 'v', data = 'cm', acc = False)
	    hist(nb_start=0, nb_end=80, backend='tf', model = 'v', data = 'c', acc = False)
	    hist(nb_start=0, nb_end=80, backend='th', model = 'v', data = 'cm', acc = False)
	    hist(nb_start=0, nb_end=80, backend='th', model = 'v', data = 'c', acc = False)