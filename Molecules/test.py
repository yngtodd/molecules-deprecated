import Molecules as m
from m.models.unsupervised import *
from m.utils import Extract as ex

batch_size = 32
epochs = 2

structure_path = "./protein.pdb"
trajectory_path = "./cont-mat.array"
native_contact = ex.ExtractNativeContact(structure_path, trajectory_path)
native_contact.extract_native_contact()
x_train, x_test = native_contact.load_native_contact(split=0.8)

input_shape = x_train.shape[1]*x_train.shape[1]

encoder = LinEncoder(latent_size = 3,
                     input_size = input_shape,
                     num_layers = 5,
                     activation = 'relu')

decoder = LinDecoder(latent_size = 3,
                     output_size = input_shape,
                     num_layers = 5,
                     activation = 'relu')

autoEncoder = AutoEncoder(encoder, decoder, mode='pytorch')

autoEncoder.summary()

autoEncoder.fit(X_train)

autoEncoder.predict(x_test)
