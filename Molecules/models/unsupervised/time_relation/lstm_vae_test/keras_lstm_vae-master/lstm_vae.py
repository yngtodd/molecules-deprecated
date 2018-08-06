import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from lstm_vae import create_lstm_vae

def get_data():
    # read data from file
    data = np.load("./encoded_train_150_tf.npy")
    timesteps = 100
    dataX = []
    for i in range(len(data[179200:]) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)


if __name__ == "__main__":
    x = get_data()
    input_dim = x.shape[-1]
    timesteps = x.shape[1]
    batch_size = 1

    vae, enc, gen = create_lstm_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=batch_size, 
        intermediate_dim=32,
        latent_dim=3,
        epsilon_std=1.)

    # fit, save & load weights 
    #vae.fit(x, x, epochs=30)
    #vae.save_weights('./lstm_vae_30')
    vae.load_weights('./try_01/lstm_vae_30')

    # predict & encode
    preds = vae.predict(x[:], batch_size=batch_size)
    np.save("./pred.npy", preds)
    encodes = enc.predict(x[:], batch_size=batch_size)
    np.save("./enc.npy", encodes)

    # pick a column to plot.
    print("[plotting...]")
    print("x: %s, preds: %s" % (x.shape, preds.shape))
    plt.plot(x[:1000,0,2], label='data')
    plt.plot(preds[:1000,0,2], label='predict')
    plt.legend()
    plt.show()


