from tensorflow.keras.layers import Input, Dense, Layer, Reshape, UpSampling2D, Flatten, Masking, Dropout, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Lambda
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import set_random_seed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error

import numpy as np
import utilityFunctions as uf

epsilon_std = .1
batch_size = 1
latent_dim = 20

def data_generator(filenames, n_pack):
    # n_pack => 100 samples of matrix
    # n_pack = batch_size
    
    files = []              # different files
    num_packs = []          # subpacked inside of file
    for filename in filenames:
        f = np.load(filename)
        files.append(f)
        num_packs.append(f.files)
    
    counter = 0
    while True:
        x = []
        for i in range(len(files)):
            rand_num = np.random.randint(len(num_packs[i]))
            f = files[i]
            pack = num_packs[i]
            x.append(f[pack[rand_num]])
            
        yield (x, x)
        
        if counter >= n_pack:
            counter = 0
    
    
def shuffle_data(org_data):
    shufle_data = []
    for data in org_data:
        tmp_data = data
        _, row, col, _ = tmp_data.shape
        tmp_data = tmp_data.flatten()
#         tmp_data[tmp_data > 0] = 1;   # set nonzero values to 1
#         tmp_data = tmp_data * np.random.rand(len(tmp_data))  # multiply with random vecotor
        np.random.shuffle(tmp_data)  # shuffle org data
        shufle_data.append(np.array(tmp_data).reshape(1, row, col, 1))

    return shufle_data

    
def order_inputs(model, org_data):
    new_order_data = []
    for inp in model.inputs:
        _, x, y, _ = inp.shape
        for data in org_data:
            row, col = data.shape
            if x == row and y == col:
                new_order_data.append(np.array(data).reshape(1,row,col,1))
                break

    return new_order_data
            
        
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
            
    
def vae_loss(input_img, output):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(output - input_img))
    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
    # return the average loss over all images in batch
    total_loss = K.mean(reconstruction_loss + kl_loss)
    return total_loss
        
    
class MultiModal:
    
    def __init__(self, graph=None, path=''):
        self.input_visibles = []
        self.input_layers = []
        self.outputs_layers = []
        self.input_data_size = []
        self.filenames = []
        self.inputs = None
        self.model = None
        self.path = path
        self.org_data = []
        self.latent_dim = 100
        self.intermediate_dim = 600
        self.epsilon_std = 0.1
        self.batch_size = 1
        
        if graph is not None:
            self._read_graph(graph)
        self._callbacks()
                    
    def _read_graph(self, graph):
        already = set()
        for obj in graph.objects.values():        
            for relation in obj.relation_x:
                if relation.name not in already:
                    self._set_params(relation)
                    already.add(relation.name)

            for relation in obj.relation_y:
                if relation.name not in already:
                    self._set_params(relation)
                    already.add(relation.name)
                    
    def _vae_loss(self, _input, _output):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.sum(K.square(_output - _input))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.square(K.exp(self.z_log_var)), axis=-1)
        # return the average loss over all images in batch
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss
    
    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
            
    
                        
    def _set_params(self, relation):
        print(relation.name + '\t' + str(relation.matrix.shape))
        self._input_layer(relation.matrix.shape)
        
        row, col = relation.matrix.shape
        self.org_data.append(np.array(relation.matrix).reshape(1,row,col,1))
        self.input_data_size.append(relation.matrix.shape)
        self.filenames.append(self.path + relation.name + '.npz')
        
    def _input_layer(self, input_size):
        row, col = input_size
        visible = Input(shape=(row, col, 1))
        
#         Conv2D filtered by columns
        layer1 = Conv2D(64, (1, col-7), activation='relu', padding='valid')(visible)
        layer1 = MaxPooling2D((2, 2))(layer1)
        layer1 = Conv2D(32, (3, 3), activation='relu')(layer1)
        layer1 = MaxPooling2D((2, 2))(layer1)
        layer1 = Conv2D(1, (1, 1), activation='relu')(layer1)
#         layer1 = MaxPooling2D((2, 2))(layer1)
#         layer1 = Conv2D(8, (1, 3), activation='relu')(layer1)
        layer1 = Flatten()(layer1)
        
        # Conv2D filtered by rows
        layer2 = Conv2D(64, (row-7, 1), activation='relu', padding='valid')(visible)
        layer2 = MaxPooling2D((2, 2))(layer2)
        layer2 = Conv2D(32, (3, 3), activation='relu')(layer2)
        layer2 = MaxPooling2D((2, 2))(layer2)
        layer2 = Conv2D(1, (1, 1), activation='relu')(layer2)
#         layer2 = MaxPooling2D((2, 2))(layer2)
#         layer2 = Conv2D(8, (3, 1), activation='relu', padding='same')(layer2)
        layer2 = Flatten()(layer2)
        
        # Merge both layer
        merge = concatenate([layer1, layer2])
        layer = Dense(int((row + col)/2), activation="relu")(merge);
        
        # Encoder
        
        layer = Dense(1000, activation="relu")(layer)
        layer = Dense(100, activation="relu")(layer)
        layer = Dense(20, activation="relu")(layer) 
        
        self.input_layers.append(layer)
        self.input_visibles.append(visible)
    
    def _output_layer(self, input_size):
        row, col = input_size

        layer = Dense(row, activation='relu')(self.inputs)
        layer = Reshape((row, 1, 1))(layer)
        layer = UpSampling2D((1, col))(layer)
        layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer)

        self.outputs_layers.append(layer)
        
    def _vae_encoder(self):
        # Encoding Layers
        h = Dense(self.intermediate_dim,activation="elu")(self.inputs)
        h = Dropout(0.7)(h)
        h = Dense(self.intermediate_dim, activation='elu')(h)
        h = BatchNormalization()(h)
        h = Dense(self.intermediate_dim, activation='elu')(h)
        
        # Latent Layers
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)
        self.z = Lambda(self._sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])
        
    def _vae_decoder(self):
        decoder_1= Dense(self.intermediate_dim, activation='elu')
        decoder_2=Dense(self.intermediate_dim, activation='elu')
        decoder_2d=Dropout(0.7)
        decoder_3=Dense(self.intermediate_dim, activation='elu')
        decoder_out=Dense(1000, activation='sigmoid')
        x_decoded_mean = decoder_out(decoder_3(decoder_2d(decoder_2(decoder_1(self.z)))))
        self.inputs = x_decoded_mean
        
    def _encoder(self):
        x = Dense(1000, activation="relu")(self.inputs)
        x = Dense(100, activation="relu")(x)
        x = Dense(20, activation="relu")(x) 
        self.inputs = x
        
    def _decoder(self):
        x = Dense(100, activation="relu")(self.inputs)
        x = Dense(1000, activation="relu")(x)
        self.inputs = x
        
    def _callbacks(self):
        log_dir = '/data/logs/'
        self.callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True),
            EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
        ]
        
    def build_model(self, optimizer='sgd', loss='mse'):
        self.inputs = concatenate(self.input_layers)
        self.inputs = Dense(20, activation="relu")(self.inputs)
        self._encoder()
        self._decoder()
#         self.inputs = Dense(800, activation="relu")(self.inputs)
#         self._vae_encoder()
#         self._vae_decoder()

        [self._output_layer(data_size) for data_size in self.input_data_size]
        
        self.model = Model(inputs=self.input_visibles, outputs=self.outputs_layers)
        self.model.compile(optimizer=optimizer, loss=loss)
#         self.model.compile(optimizer=optimizer, loss='kullback_leibler_divergence')
# optimizer="adam"

#         self.model.compile(optimizer="sgd", loss=self._vae_loss
#                            ,metrics=[
#             "categorical_accuracy",
#         #     "fmeasure",
#         #     "top_k_categorical_accuracy",
#             "mean_squared_error"
#         ]
#                           )
        self.model.summary()
        
    def fit(self, batch_size, epochs):
        self.batch_size = batch_size;
        self.model.fit_generator(
            data_generator(self.filenames, batch_size), 
            steps_per_epoch=batch_size, 
            epochs=epochs,
            callbacks=self.callbacks
        )
        
    def save(self, path, version='1'):
        self.model.save(path + 'mm_weight_v' + version + '.h5')

    def load_model(self, path):
        self.model = load_model(path)
        
    def predict(self, data=None):
        if data is not None:
            self.org_data = order_inputs(self.model, data)
            
        shuffled_data = shuffle_data(self.org_data)
        predict_data = self.model.predict(self.org_data)
        base_line = self.model.predict(shuffled_data)
        print('Data \t\t\tDensity \tPredict \tBaseLine \tMean')
        for i in range(len(self.org_data)):
            _, row, col, _ = self.org_data[i].shape 
            mse = mean_squared_error(self.org_data[i].flatten(), predict_data[i].flatten())
            mse_base_line = mean_squared_error(shuffled_data[i].flatten(), base_line[i].flatten())
            
            non_zeros = self.org_data[i].flatten()
            non_zeros[self.org_data[i].flatten() > 0] = 1
            org_mean = round(np.mean(non_zeros * self.org_data[i].flatten()), 5)
            predict_mean = round(np.mean(non_zeros * predict_data[i].flatten()), 5)
            
            print('(' + str(row) + ',' + str(col) + ') ' + '\t\t' + str(round((np.count_nonzero(self.org_data[i])/(row * col)) * 100, 2)) + '% \t\t' + str(round(mse * 100, 5)) + '% \t' + str(round(mse_base_line * 100, 5)) + '%'+ '\t' + str(org_mean) + ' - ' + str(predict_mean))
            
       