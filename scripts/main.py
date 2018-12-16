from relationGraph import Relation, RelationGraph, MatrixOfRelationGraph
from datasets.base import load_source
from os.path import join
import numpy as np

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


if __name__ == '__main__':
    graph = test_build_relation_graph()
    test_convert_graph_to_2D_matrix(graph)

