from abc import ABC

# import all autoencoder versions
from models.unsupervised import linear_vae as lv_py
from models.unsupervised import vae_conv-train_load as cv_k

 
class Encoder(ABC):
 
    def __init__(self, latent_size):
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        """
        if (latent_size < 0):
            raise Exception("latent_size must be greater than 0!")

        self.latent_size = latent_size
        super().__init__()

class ConvEncoder(Encoder):
    def __init__(self, latent_size):
       
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        """
        super().__init__(latent_size)
        self.type = 'conv'

class LinEncoder(Encoder):
    def __init__(self, latent_size, input_size, num_layers, activation):
       
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        input_size : int
            Input dimension for the data. Should equal the output dimension of the decoder.
        num_layers : int
            Number of linear dense layers to add to the encoder.
        activation : str
            Type of activation function EX) 'sigmoid', 'relu'
        
        """

        super().__init__(latent_size)

        if (input_size < 0):
            raise Exception("input_size must be greater than 0!")
        if (num_layers < 0):
            raise Exception("num_layers must be greater than 0!")

        # TODO: Add exception handling for activation

        self.latent_size = latent_size
        self.output_size = output_size
        self.num_layers =  num_layers
        self.activation = activation
        self.type = 'lin'

class Decoder(ABC):
     def __init__(self, latent_size):
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        """
        if (latent_size < 0):
            raise Exception("latent_size must be greater than 0!")

        self.latent_size = latent_size
        super().__init__()

class ConvDecoder(Decoder):
    def __init__(self, latent_size):
       
        """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        """
        super().__init__(latent_size)
        self.type = 'conv'
      
class LinDecoder(Decoder):
    def __init__(self, latent_size, output_size, num_layers, activation):
       
         """
        Parameters:
        ----------
        latent_size : int
            latent dimension size of the autoencoder.
        output_size : int
            Output dimension for the data. Should equal the input dimension of the encoder.
        num_layers : int
            Number of linear dense layers to add to the decoder.
        activation : str
            Type of activation function EX) 'sigmoid', 'relu'
        
        """

        super().__init__(latent_size)

        if (output_size < 0):
            raise Exception("output_size must be greater than 0!")
        if (num_layers < 0):
            raise Exception("num_layers must be greater than 0!")

        # TODO: Add exception handling for activation

        self.latent_size = latent_size
        self.output_size = output_size
        self.num_layers =  num_layers
        self.activation = activation
        self.type = 'lin'

class AutoEncoder(object):
    def __init__(self, encoder, decoder, mode, shape, backend=None):
          """
        Parameters:
        ----------
        encoder : Encoder
            encoder options for the autoencoder.
        decoder : Decoder
            decoder options of the autoencoder.
        mode : str
            Decides backend. Either 'pytorch' or 'keras'.
        shape : tuple
            Shape of X_train EX) shape=X_train.shape
        backend : str
            Required iff backend is 'keras'. Selects keras backend as 'tf' or 'th' or 'cntk'.
        """
        if (encoder.type != decoder.type):
            raise Exception("Encoder and decoder must have the same type. Either both linear or both convolutional.")

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

        # TODO: Add checks to see that conv_e == conv_d and lin respecitevly.

        self.type = encoder.type
        self.backend = backend
        self.encoder = encoder
        self.decoder = decoder
        self.shape = shape

        self.model = self.buildAutoEncoder()
                   
    def buildAutoEncoder(self):
        AE_model = None
        if (self.type == 'lin'):
            if (self.mode == 'pytorch'):
                AE_model = self.buildPytorchLinModel()
            elif (self.mode == 'keras'):
                AE_model = buildKerasLinModel()
        elif (self.type == 'conv'):
            if (self.mode == 'pytorch'):
                AE_model = buildPytorchConvModel()
            elif (self.mode == 'keras'):
                AE_model = buildKerasConvModel()

        if (AE_model == None):
            raise Exception("Internal Error: AE_model is uninitialized!")
        return AE_model

    def buildPytorchLinModel(self):
        encoder = lv.Encoder(self.encoder.input_size,
                             self.encoder.latent_size)
        decoder = lv.Decoder(self.decoder.latent_size,
                             self.decoder.output_size)

        return lv.VAE(encoder, decoder)

    def buildKerasLinModel(self):
        pass

    def buildPytorchConvModel(self):
        pass

    def buildKerasConvModel(self):
        # define parameters for variational autoencoder - convolutional
        image_size = self.shape[-2:];
        channels = 1
        conv_layers = 3
        feature_maps = [128,128,128,128]
        filter_shapes = [(3,3),(3,3),(3,3),(3,3)]
        strides = [(1,1),(2,2),(1,1),(1,1)]
        dense_layers = 1
        dense_neurons = [128]
        dense_dropouts = [0]
        latent_dim = 3
       
        feature_maps = feature_maps[0:conv_layers]
        filter_shapes = filter_shapes[0:conv_layers]
        strides = strides[0:conv_layers]

        return cv_k.conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
                                            filter_shapes,strides,dense_layers,dense_neurons,
                                            dense_dropouts,latent_dim) 



    def train(x_train, y_train=None):
        if y_train == True:
            pass
        if (self.type == 'lin'):
            if (self.mode == 'pytorch'):
                self.model.buildPytorchLinModel()
            elif (self.mode == 'keras'):
                self.model.buildKerasLinModel()
        elif (self.type == 'conv'):
            if (self.mode == 'pytorch'):
                self.model.buildPytorchConvModel()
            elif (self.mode == 'keras'):
                self.model.buildKerasConvModel()


    def predict(x_test, y_test=None):
        pass
