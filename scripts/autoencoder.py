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


def load_encoder():
    return load_model(r'/mag/weights/encoder_weights.h5')


def load_decoder():
    return load_model(r'/mag/weights/decoder_weights.h5')


def mp_worker(arr):
#     new_data = uf.sample_generator3(data, num_of_samples=100, density=0.7)
    new_data = uf.sample_generator3(arr[0], num_of_samples=arr[1], density=arr[2])
    return new_data


def data_generator(data, n_samples, batch_size, density):
#     p = multiprocessing.Pool(8)
#     gen_samples = np.empty((0, data.shape[0] * data.shape[1]))
#     iterations = int(np.round(n_samples/100)) if  n_samples > 100 else 1
    
#     params = [[data, 100, density] for x in range(5)]
#     for result in p.imap(mp_worker, params):
#         print(result.shape)
#         gen_samples = np.r_[gen_samples, result]
    
#     return gen_samples
    samples_per_epoch = n_samples
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    
    while True:
        x = uf.sample_generator3(data, num_of_samples=n_samples, density=0.7)
        yield (x, x)
        
        if counter >= number_of_batches:
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
        print(self.x)
        print(self.x.shape)
        print(self.x_dim)

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
        log_dir = '/mag/logs/'
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)]
#             EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')]

        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks
                      )
    
    def fit_generator(self, n_samples=100, density=0.8, batch_size=20, epochs=100):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = '/mag/logs/'
        callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)]
#             EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')]

        self.model.fit_generator(data_generator(self.x, n_samples, batch_size, density), steps_per_epoch=int((n_samples - 1)/batch_size) + 1, epochs=epochs)


    def save(self):
        if not os.path.exists(r'/mag/weights'):
            os.mkdir(r'/mag/weights')

        self.encoder.save(r'/mag/weights/encoder_weights.h5')
        self.decoder.save(r'/mag/weights/decoder_weights.h5')
        self.model.save(r'/mag/weights/ae_weights.h5')


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
