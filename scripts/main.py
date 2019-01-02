from relationGraph import Relation, RelationGraph, MatrixOfRelationGraph
from autoencoder import seedy, load_encoder, load_decoder, AutoEncoder
from base import load_source
from sklearn.metrics import mean_squared_error
from os.path import join
import utilityFunctions as uf
import numpy as np


def test_build_relation_graph_with_symertic_data():
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

    return relationGraph


def test_build_relation_graph():
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

    relationGraph = RelationGraph()
    relationGraph.add_relations([ann, expr, ppi])
    relationGraph.display_objects()

    return relationGraph


def test_convert_graph_to_2D_matrix(graph):
    mrg = MatrixOfRelationGraph(graph=graph)
    mrg.convert_to_2D_matrix()
    mrg.display_metadata_2D_matrix()
    mrg.display_density_data()
    print()
    print('2D matrix: ' + str(mrg.matrix_2D.shape))


def test_get_matix_for_autoencoder(graph):
    mrg = MatrixOfRelationGraph(graph=graph)
    mrg.convert_to_2D_matrix()
    data = mrg.density_data()
    print(data.shape)
    print(data.flatten())
    return data


def test_autoencoder(data):
    seedy(42)
    ae = AutoEncoder(encoding_dim=2, data=data.flatten())
    ae.encoder_decoder()
    ae.fit(batch_size=50, epochs=300)
    ae.save()

    encoder = load_encoder()
    decoder = load_decoder()

    inputs = [data.reshape(26244, 1)]
    x = encoder.predict(inputs)
    y = decoder.predict(x)

    mse = mean_squared_error(inputs, y)
    print('MSE: ' + str(mse))


def test_generation_samples_by_rows(data=None):
    data = uf.random_generate_sample(np.ones((10, 10)), 5, 8)
    assert data.shape == (5, 8)


def test_generate_samples_by_cells(data=None):
    data = uf.sample_generator2(np.ones((10, 10)))
    print(data)


if __name__ == '__main__':
    if False:
        graph = test_build_relation_graph()
        test_convert_graph_to_2D_matrix(graph)

    if False:
        graph = test_build_relation_graph_with_symertic_data()
        test_convert_graph_to_2D_matrix(graph)

    if False:
        graph = test_build_relation_graph_with_symertic_data()
        data = test_get_matix_for_autoencoder(graph)
        test_autoencoder(data)
    if False:
        test_generation_samples_by_rows()
        test_generate_samples_by_cells()

