from tensorflow.keras.layers import Input, Dropout, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import set_random_seed
import numpy as np
import utilityFunctions as uf
import os

import multiprocessing
import time
from pathlib import Path


def seedy(s):
    np.random.seed(s)
    set_random_seed(s)
    
    
def load_encoder(folder):
    return load_model(r'/data/weights/' + str(folder) + '/encoder_weights.h5')


def load_decoder(folder):
    return load_model(r'/data/weights/' + str(folder) + '/decoder_weights.h5')

            
def data_generator(filename, n_pack):
    # n_pack => 100 samples of matrix
    # n_pack = batch_size
    f = np.load(filename)
    files = f.files
    
    counter = 0
    while True:
        rand_num = np.random.randint(len(files))
        x = f[files[rand_num]]
        yield (x, x)
        
        if counter >= n_pack:
            counter = 0


class AutoEncoder:

    def __init__(self, encoding_dim=3, data=None):
        self.encoding_dim = encoding_dim
        if data is None:
            # generate dummy data
            r = lambda: np.random.randint(1, 3)
            self.x = np.array([[r(), r(), r()] for _ in range(1000)])
        else:
            self.x = data
            
        self.x_dim = self.x.shape[1]
        self.path = r'/data/weights/' + str(np.int(np.sqrt(self.x_dim)))
        print(self.x)
        print(self.x.shape)
        print(self.x_dim)
        print(self.path)

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(self.x_dim)(inputs)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size=100, epochs=100):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = '/data/logs/'
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)]
#             EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')]

        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks
                      )
    
    def fit_generator(self, filename, n_packs=2, epochs=100):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = '/data/logs/'
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True),
            EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
        ]

        self.model.fit_generator(data_generator(filename, n_packs), steps_per_epoch=n_packs, epochs=epochs, callbacks=callbacks)

    def save(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.encoder.save(self.path + '/encoder_weights.h5')
        self.decoder.save(self.path + '/decoder_weights.h5')
        self.model.save(self.path + '/ae_weights.h5')
        
    def load_encoder(self):
        return load_model(self.path + '/encoder_weights.h5')

    def load_decoder(self):
        return load_model(self.path + '/decoder_weights.h5')


# Example how to call Autoencoder class
# if __name__ == '__main__':
    # print('-------Build model-------')
    # seedy(42)
    # ae = AutoEncoder(encoding_dim=2, data=data)
    # ae.encoder_decoder()
    # ae.fit(batch_size=50, epochs=300)
    # ae.save()
    #
    # print()
    # print('-------Predict data-------')
    # encoder = load_encoder()
    # decoder = load_decoder()
    #
    # inputs = [data]
    # x = encoder.predict(inputs)
    # y = decoder.predict(y)
    #
    # print('Inputs: {}'.format(inputs))
    # print('Encoded: {}'.format(x))
    # print('Decoded: {}'.format(y))

    
# Example - how to run Autoencoder
# x, y = data.shape
# data=data.reshape(1, x * y)
# ae = AutoEncoder(encoding_dim=20, data=data)
# ae.encoder_decoder()
# # ae.fit(batch_size=250, epochs=100)
# # fn = '/mag/483_data.npz'
# ae.fit_generator(fn, n_packs=10, epochs=200)
# ae.save()

# encoder = ae.load_encoder()
# decoder = ae.load_decoder()

# # test_data = np.asarray([data[0].flatten()])
# test_data = np.asarray([data.flatten()])
# # print(test_data)
# # np.random.shuffle(test_data[0])
# # print(test_data)
# print(test_data.shape)

# x = encoder.predict(test_data)
# y = decoder.predict(x)

# mse = mean_squared_error(test_data, y)
# print('MSE: ' + str(mse))