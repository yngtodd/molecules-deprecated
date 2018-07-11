from __future__ import print_function
import Molecules as m
from m.models.unsupervised import *
from m.utils import Extract as ex


class AutoEncoderTestSuite(object):
    def __init__(self):
        pass

    def regression_test(self):
        print("Regression testing ...\n")
        self.test_pytorch_vae()
        self.test_pytorch_cvae()
        self.test_keras_vae()
        self.test_keras_cvae()
        print("Finished regression testing\n")

    def test_pytorch_vae(self):
        print("Testing pytorch VAE\n")
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
        print ("Finished testing pytorch VAE\n")

    def test_pytorch_cvae(self):
        print("Testing pytorch CVAE\n")
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

        print ("Finished testing pytorch CVAE\n")

    def test_keras_vae(self):
        print("Testing keras VAE\n")
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

        print ("Finished testing keras VAE\n")

    def test_keras_cvae(self):
        print("Testing keras CVAE\n")
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

        print ("Finished testing keras CVAE\n")
