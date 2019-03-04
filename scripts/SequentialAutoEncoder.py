from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Masking
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import autoencoder as ae
#-----------------------------------------
from relationGraph import Relation, RelationGraph, MatrixOfRelationGraph
from autoencoder import seedy, AutoEncoder
import utilityFunctions as uf
from main import test_build_relation_graph_with_symertic_data, test_convert_graph_to_2D_matrix, test_get_matix_for_autoencoder, test_autoencoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from base import load_source
from os.path import join
import numpy as np
#-------------------------------------------------
gene = 'Gene'
go_term = 'GO term'
exprc = 'Experimental condition'

data, rn, cn = load_source(join('dicty', 'dicty.gene_annnotations.csv.gz'))
ann = Relation(data=data, x_name=gene, y_name=go_term, name='ann',
               x_metadata=rn, y_metadata=cn)

data, rn, cn = load_source(join('dicty', 'dicty.gene_expression.csv.gz'))
expr = Relation(data=data, x_name=gene, y_name=exprc, name='expr',
                x_metadata=rn, y_metadata=cn)
expr.matrix = np.log(np.maximum(expr.matrix, np.finfo(np.float).eps))

data, rn, cn = load_source(join('dicty', 'dicty.ppi.csv.gz'))
ppi = Relation(data=data, x_name=gene, y_name=gene, name='ppi',
               x_metadata=rn, y_metadata=cn)

ann_t = ann.transpose()
expr_t = expr.transpose()

relationGraph = RelationGraph()
relationGraph.add_relations([ann, expr, ppi, ann_t, expr_t])
relationGraph.display_objects()
graph = relationGraph

#----------------------------------------------------
mrg = MatrixOfRelationGraph(graph=graph)
mrg.convert_to_2D_matrix()
mrg.display_metadata_2D_matrix()
data = mrg.density_data(.2)
print(data.shape)
fn = '/data/samples/' + str(data.shape[0]) + '_data.npz'
# fn = '/data/samples/' + str(data.shape[0]) + '_ord_data.npz'    // original data for prediciton
print(fn)
print(data.shape)

# f = np.load('/data/samples/org_data.npz')
# data = f[f.files[0]]

print(data.shape)

#---------------------------------------------



def load_data (filename, n_packs):
    f = np.load(filename)
    files = f.files
    
    while n_packs > 0:
        rand_num = np.random.randint(len(files))
        x = f[files[rand_num]]
        print(x.shape)
        
        n_packs -= 1

        
x,y = data.shape
data=data.reshape(1, x * y)
input_dim = data.shape[1]
epochs = 200
encoding_dim = 20
n_packs = 250

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(input_dim, )))
model.add(Dense(encoding_dim, input_shape=(input_dim, ), activation='relu'))
model.add(Dense(input_dim))
model.compile(loss='mse', optimizer='sgd')

log_dir = '/data/logs/'
callbacks = [
            TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True),
            EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
        ]

model.fit_generator(ae.data_generator(fn, n_packs), steps_per_epoch=n_packs, epochs=epochs, callbacks=callbacks)

model.save('/data/sequential/weights/model.h5')

# decoded_imgs = model.predict(data)

# mse = mean_squared_error(data, decoded_imgs)
# print('MSE: ' + str(mse))
#-------------------------------------------
f = np.load(filename)
test_data = f[f.files[0]]

# prediction with normal data
model = load_model.save('/data/sequential/weights/model.h5')
y = model.predict(test_data)
mse = mean_squared_error(data, decoded_imgs)
print('MSE: ' + str(mse))
print()

# prediction with shuffled data
np.random.shuffle(test_data)
y = model.predict(test_data)
mse = mean_squared_error(data, decoded_imgs)
print('MSE: ' + str(mse))