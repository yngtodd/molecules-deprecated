# import all autoencoder versions

class ConvEncoder(object):
    def __init__(self, type, latent_size):
       
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        output_size : int
            Output dimension for the data. Should equal input_dimension of AE.
        kernel* : int, defualt=4
            Convolutional filter size for layer *.
        stride* : int, default=2
            Stride length for convolutional filter at layer *.
        """
        self.latent_size = latent_size


class LinEncoder(object):
    def __init__(self, type, latent_size, output_size, num_layers, activation):
       
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        output_size : int
            Output dimension for the data. Should equal input_dimension of AE.
        kernel* : int, defualt=4
            Convolutional filter size for layer *.
        stride* : int, default=2
            Stride length for convolutional filter at layer *.
        """
        self.latent_size = latent_size
        self.output_size = output_size

        self.num_layers =  num_layers
        self.activation = activation

class LinDecoder(nn.Module):
    def __init__(self, type, latent_size, output_size, num_layers, activation):
       
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        output_size : int
            Output dimension for the data. Should equal input_dimension of AE.
        kernel* : int, defualt=4
            Convolutional filter size for layer *.
        stride* : int, default=2
            Stride length for convolutional filter at layer *.
        """
        self.latent_size = latent_size
        self.output_size = output_size

        self.num_layers =  num_layers
        self.activation = activation


class AutoEncoder(object):
    def __init__(self, encoder, decoder, mode, backend=None):
          """
        Parameters:
        ----------
        encoder : LinEncoder or ConvEncoder
            encoder options for the autoencoder.
        decoder : LinDecoder or ConvDecoder
            decoder options of the autoencoder.
        mode : str
            Decides backend. Either 'pytorch' or 'keras'.
        backend : str
            Required iff backend is 'keras'. Selects keras backend as 'tf' or 'th'.
        """
        if (decoder.latent_size != encoder.latent_size):
            raise Exception("Encoder and decoder must have the same latent dimension.")

        if (mode != 'pytorch' and mode != 'keras'):
            raise Exception("mode must be 'pytorch' or 'keras'")

        if (mode != 'keras' and backend != None):
            raise Exception("Only specify backend if mode is 'keras'")

        self.mode = mode

        if (self.mode == 'keras'):
            if (backend != 'tf' and backend != 'th' and backend != 'cntk'):
                raise Exception("keras selected, must also specify backend 'tf', 'th', or 'cntk'")

        self.backend = backend

        self.model = buildAutoEncoder()
        
            
    def buildAutoEncoder(self):
        AE_model = None
        
        if (self.mode == 'pytorch'):
            AE_model = buildPytorchVAE()
        elif (self.mode == 'keras'):
            AE_model = buildKerasVAE(self.backend)
        
        if (self.mode == 'pytorch'):
            AE_model = buildPytorchCVAE()
        elif (self.mode == 'keras'):
            AE_model = buildKerasCVAE(self.backend)

        return AE_model


    def train(x_train, y_train=None):
        if y_train == True:
            pass
        if mode is keras:
            do keras


    def predict(x_test, y_test=None):
        pass
