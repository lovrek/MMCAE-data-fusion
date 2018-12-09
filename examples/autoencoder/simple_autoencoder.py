from tensorflow.keras.layers import Input, Dropout, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import set_random_seed
import numpy as np
import os


def seedy(s):
    np.random.seed(s)
    set_random_seed(s)


class AutoEncoder:

    def __init__(self, encoding_dim=3):
        self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)
        self.x = np.array([[r(), r(), r()] for _ in range(1000)])
        print(self.x)

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(3)(inputs)
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

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = '/mag/logs/'
        callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)]

        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks)

    def save(self):
        if not os.path.exists(r'/mag/weigths'):
            os.mkdir(r'/mag/weigths')

        self.encoder.save(r'/mag/weigths/encoder_weigths.h5')
        self.decoder.save(r'/mag/weigths/decoder_weigths.h5')
        self.model.save(r'/mag/weigths/ae_weigths.h5')


if __name__ == '__main__':
    seedy(2)
    ae = AutoEncoder(encoding_dim=2)
    ae.encoder_decoder()
    ae.fit(batch_size=50, epochs=300)
    ae.save()
