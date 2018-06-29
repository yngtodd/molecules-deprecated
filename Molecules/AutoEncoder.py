from abc import ABC

# import all autoencoder versions
from models.unsupervised import linear_vae as lv

 
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
      
class LinDecoder(nn.Module):
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

class AutoEncoder(object):
    def __init__(self, encoder, decoder, mode, backend=None):
          """
        Parameters:
        ----------
        encoder : Encoder
            encoder options for the autoencoder.
        decoder : Decoder
            decoder options of the autoencoder.
        mode : str
            Decides backend. Either 'pytorch' or 'keras'.
        backend : str
            Required iff backend is 'keras'. Selects keras backend as 'tf' or 'th' or 'cntk'.
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
        self.encoder = encoder
        self.decoder = decoder


        self.model = buildAutoEncoder()
        
            
    def buildAutoEncoder(self):
        AE_model = None
        
        if (self.mode == 'pytorch'):
            AE_model = buildPytorchModel()
        elif (self.mode == 'keras'):
            AE_model = buildKerasModel(self.backend)
        
        return AE_model




    def buildPytorchModel(self):
        e = lv.Encoder(self.encoder.input_size,
                       self.encoder.latent_size)
        d = lv.Decoder(self.decoder.latent_size,
                       self.decoder.output_size)








    def buildKerasModel(self):
        pass


    def train(x_train, y_train=None):
        if y_train == True:
            pass
        if mode is keras:
            do keras


    def predict(x_test, y_test=None):
        pass
