from tensorflow.keras.layers import Input, Dropout, Dense, Masking
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import set_random_seed
import numpy as np
import os
import imageio
import scipy.misc


def seedy(s):
    np.random.seed(s)
    set_random_seed(s)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def duplicate_data(data, num_of_samples=10, sample_discard=.2):
    x, y = data.shape
    removed_elements = round(sample_discard * y)

    samples = np.zeros((x * num_of_samples, y))

    for i in range(x):
        for j in range(num_of_samples):
            vector = np.copy(data[i])
            random_vector = np.random.choice(y, removed_elements, replace=False)
            vector[random_vector] = 0

            samples[i * num_of_samples + j] = vector

    return samples


def read_data():
    arr = imageio.imread('../data/testImage4k.jpg')

    return arr


class AutoEncoder:

    def __init__(self, encoding_dim=3, num_of_samples=10, sample_discard=0):
        self.encoding_dim = encoding_dim
        data = read_data()
        #         data = rgb2gray(data)
        self.x = duplicate_data(data, num_of_samples, sample_discard)
        self.y = data
        print(self.x.shape)

    def _encoder(self):
        print('encoder')
        inputs = Input(shape=(self.x[0].shape))

        encoded = inputs
        if self.layers < 2:
            #             inputs = Masking(mask_value=0.0)(inputs)
            encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        else:
            layer_dim = self.x[0].shape[0]

            for i in range(self.layers):
                layer_dim -= self.step
                #                 encoded = Masking(mask_value=0.0)(encoded)
                encoded = Dense(layer_dim, activation='relu')(encoded)
                print('Layer dim: ' + str(layer_dim))

            if layer_dim > self.encoding_dim:
                encoded = Dense(self.encoding_dim, activation='relu')(encoded)
                print('Layer dim: ' + str(self.encoding_dim))

        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        print('decoder')
        inputs = Input(shape=(self.encoding_dim,))

        decoded = inputs;
        if self.layers < 2:
            decoded = Dense(self.x[0].shape[0])(inputs)
        else:
            residue = self.x[0].shape[0] - (self.layers * self.step) - self.encoding_dim
            layer_dim = 0
            for i in range(self.layers):
                layer_dim = self.encoding_dim + residue + self.step * i
                decoded = Dense(layer_dim, activation='relu')(decoded)
                print('Layer dim: ' + str(layer_dim))

            if layer_dim < self.x[0].shape[0]:
                decoded = Dense(self.x[0].shape[0], activation='sigmoid')(decoded)
                print(self.x[0].shape[0])

        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self, num_of_layers=3):
        self.layers = num_of_layers
        self.step = int((self.x[0].shape[0] - self.encoding_dim) / self.layers)
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
        log_dir = '/mag/logs/'
        #         callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)]
        callbacks = [
            EarlyStopping(monitor='val_loss',
                          min_delta=0.0,
                          patience=3,
                          verbose=0,
                          mode='auto'),
            TensorBoard(log_dir=log_dir)
        ]

        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(self.y, self.y),
                       callbacks=callbacks
                       )

    def save(self, folder):
        if not os.path.exists(r'/mag/weigths/' + folder):
            os.mkdir(r'/mag/weigths/' + folder)

        self.encoder.save(r'/mag/weigths/' + folder + '/encoder_weigths.h5')
        self.decoder.save(r'/mag/weigths/' + folder + '/decoder_weigths.h5')
        self.model.save(r'/mag/weigths/' + folder + '/ae_weigths.h5')
        imageio.imwrite('/mag/weigths/' + folder + '/input.png', self.x)


if __name__ == '__main__':
    discard = .2
    seedy(42)
    ae = AutoEncoder(encoding_dim=300, num_of_samples=10, sample_discard=discard)
    ae.encoder_decoder(4)
    ae.fit(batch_size=500, epochs=200)
    ae.save(str(discard) + '_1')
