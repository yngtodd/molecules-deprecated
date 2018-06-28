# import all autoencoder versions

class AutoEncoder(object):
    def __init__(self, type, mode, backend=None):

        if (type != 'linear' and type != 'conv'):
            raise Exception("type must be 'linear' or 'conv'")

        self.type = type

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
        if (self.type == 'linear'):
            if (self.mode == 'pytorch'):
                AE_model = buildPytorchVAE()
            elif (self.mode == 'keras'):
                AE_model = buildKerasVAE(self.backend)
        elif (self.type == 'conv'):
            if (self.mode == 'pytorch'):
                AE_model = buildPytorchCVAE()
            elif (self.mode == 'keras'):
                AE_model = buildKerasCVAE(self.backend)

        return AE_model