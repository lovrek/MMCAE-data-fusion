from relationGraph import Relation, RelationGraph, MatrixOfRelationGraph
from autoencoder import seedy, AutoEncoder
import utilityFunctions as uf
from main import test_build_relation_graph_with_symertic_data, test_convert_graph_to_2D_matrix, test_get_matix_for_autoencoder, test_autoencoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from base import load_source
from os.path import join
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Masking
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import autoencoder as ae

#-------------------------------------------------------

# num = 162   # most densely filled  0.1
# num = 323   # most densely filled  0.2
num = 483   # most densely filled  0.3
# num = 645   # most densely filled  0.4
# num = 807   # most densely filled  0.5

fn = '/data/samples/' + str(num) + '_org_data.npz'
f = np.load(fn)
data = f[f.files[0]]

fn = '/data/samples/' + str(num) + '_data.npz'
        
x,y = data.shape
data=data.reshape(1, x * y)
input_dim = data.shape[1]
epochs = 200
encoding_dim = 20
n_packs = 50

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(input_dim, )))
model.add(Dense(encoding_dim, input_shape=(input_dim, ), activation='relu'))
model.add(Dense(input_dim))
model.compile(loss='mse', optimizer='sgd')

# model = Sequential()
# model.add(Masking(mask_value=0, input_shape=(input_dim, )))
# model.add(Dense(int(input_dim / 2), activation='relu'))
# model.add(Dense(int(input_dim / 4), activation='relu'))
# model.add(Dense(encoding_dim, activation='relu'))
# model.add(Dense(int(input_dim / 4), activation='relu'))
# model.add(Dense(int(input_dim / 2), activation='relu'))
# model.add(Dense(input_dim))
# model.compile(loss='mse', optimizer='sgd')
# model.summary()


log_dir = '/data/logs/'
callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True),
            EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
        ]

model.fit_generator(ae.data_generator(fn, n_packs), steps_per_epoch=n_packs, epochs=epochs, callbacks=callbacks)

# model.save('/data/sequential/weights/' + str(data.shape[0]) + 'model.h5')
model.save('/data/sequential/weights/' + str(num) + '_model.h5')

decoded_imgs = model.predict(data)

mse = mean_squared_error(data, decoded_imgs)
print('MSE: ' + str(mse))

#-----------------------------------------------------------------

# num = 162   # most densely filled  0.1
num = 323   # most densely filled  0.2
# num = 483   # most densely filled  0.3
# num = 645   # most densely filled  0.4
# num = 807   # most densely filled  0.5

f = np.load('/data/samples/' + str(num) + '_org_data.npz')
test_data = np.asarray([f[f.files[0]].flatten()])
# test_data = np.asarray([data.flatten()])

# prediction with normal data
model = load_model('/data/sequential/weights/' + str(num) + '_model.h5')
y = model.predict(test_data)
mse = mean_squared_error(test_data, y)
print(test_data[0])
print(y[0])
print()
print(test_data.shape)
print(y.shape)
print()
print('MSE org data: ' + str(mse))


# prediction with shuffled data
np.random.shuffle(test_data[0])
y = model.predict(test_data)
mse = mean_squared_error(test_data, y)
print('MSE shuffled data: ' + str(mse))
print()
# print('Mean predict data: ' + str(np.mean(y[0])))
# print(test_data[0])
# print(y[0])
print()
print('Min org data:' + str(np.min(test_data)))
print('Max org data:' + str(np.max(test_data)))
print('Mean org data: ' + str(np.mean(test_data)))
print()
print('Min predict:' + str(np.min(y)))
print('Max predict:' + str(np.max(y)))
print('Mean predict: ' + str(np.mean(y)))
print()