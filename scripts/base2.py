"""
Base code for handling data sets.
"""
import gzip
import csv
from collections import defaultdict
from os.path import dirname
from os.path import join
import utilityFunctions as uf
from relationGraph import Relation, RelationGraph, MatrixOfRelationGraph

import numpy as np

def resize_rows_and_columns(relation, _exp):
    data = relation.matrix
    if _exp is None:
        return relation
    
    row, col = data.shape
    new_row = 1
    new_col = 1

    for i in range(1,1000):
        exp = np.power(_exp,i)
        if row < exp and new_row == 1:
            if (exp - row)/row > 0.5:
                new_row = np.power(_exp,i-1)
            else:
                new_row = exp
                
        if col < exp and new_col == 1:
            if (exp - col)/col > 0.5:
                new_col = np.power(_exp, i-1)
            else:
                new_col = exp

        if new_row != 1 and new_col != 1:
            break

    if row > new_row:
        data = data[:new_row]
        relation.x = dict(zip(relation.get_x_list()[:new_row], range(new_row)))
    elif row < new_row:
        data = np.r_[data, np.zeros((new_row - row, col))]
        new_x = dict(zip([relation.name + '_empty_' + str(x) for x in range(new_row-row)], range(row, new_row)))
        relation.x = uf.merge_two_dicts(relation.x, new_x)

    if col > new_col:
        data = data[:, :new_col]
        relation.y = dict(zip(relation.get_y_list()[:new_col], range(new_row)))
    elif col < new_col:
        data = np.c_[data, np.zeros((data.shape[0], new_col - col))]
        new_y = dict(zip([relation.name + '_empty_' + str(x) for x in range(new_col-col)], range(col, new_col))) 
        relation.y = uf.merge_two_dicts(relation.y, new_y)
    
    relation.matrix = data
    
    return relation


def resize_rows_and_columns_data(data, _exp):
    if _exp is None:
        return relation
    
    row, col = data.shape
    new_row = 1
    new_col = 1

    for i in range(1,1000):
        exp = np.power(_exp,i)
        if row < exp and new_row == 1:
            if (exp - row)/row > 0.5:
                new_row = np.power(_exp,i-1)
            else:
                new_row = exp
                
        if col < exp and new_col == 1:
            if (exp - col)/col > 0.5:
                new_col = np.power(_exp, i-1)
            else:
                new_col = exp

        if new_row != 1 and new_col != 1:
            break

    if row > new_row:
        data = data[:new_row]
    elif row < new_row:
        data = np.r_[data, np.zeros((new_row - row, col))]

    if col > new_col:
        data = data[:, :new_col]
    elif col < new_col:
        data = np.c_[data, np.zeros((data.shape[0], new_col - col))]
    
    return data


def load_source(source_path, delimiter=',', filling_value='0'):
    """Load and return a data source.

    Parameters
    ----------
    delimiter : str, optional (default=',')
        The string used to separate values. By default, comma acts as delimiter.

    filling_value : variable, optional (default='0')
        The value to be used as default when the data are missing.

    Returns
    -------
    data : DataSource
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'obj1_names', the meaning of row objects,
        'obj2_names', the meaning of column objects.
    """
    module_path = dirname(__file__)
    data_file = gzip.open(join(module_path, 'data', source_path))
    row_names = np.array(next(data_file).decode('utf-8').strip().replace(', ', ' ').replace('.,', '.').split(delimiter))
    col_names = np.array(next(data_file).decode('utf-8').strip().replace(', ', ' ').replace('.,', '.').split(delimiter))
    data = np.genfromtxt(data_file, delimiter=delimiter, missing_values=[''],
                         filling_values=filling_value)
    return data, row_names, col_names

def sorted_data(data, sort_alg=None):
    sort_algs = ['clustering']
    if sort_alg is not None:
        if sort_alg in sort_algs:
            return uf.order_by_clustering(data, 'single')
        else:
            raise Exception('Parameter \'sort\' must be one of the following: ' + str(sort_algs))
    return data

def sorted_relation_data(relation, sort_alg=None):
    sort_algs = ['clustering']
    if sort_alg is not None:
        if sort_alg in sort_algs:
            data = relation.matrix
            data, res_order = uf.clustering(data, 'single')
        
            x_list = relation.get_x_list()
            y_list = relation.get_y_list()
            
            new_x = {}
            new_y = {}
            
            for i, val in enumerate(res_order):
                new_x[x_list[i]] = val
                
            data, res_order = uf.clustering(data.T, 'single')
            for i, val in enumerate(res_order):
                new_y[y_list[i]] = val
                
            relation.matrix = data.T
            relation.x = new_x
            relation.y = new_y
        else:
            raise Exception('Parameter \'sort\' must be one of the following: ' + str(sort_algs))
    return relation
           

def load_dicty(sort=None, exp=None):
    gene = 'Gene'
    go_term = 'GO term'
    exprc = 'Experimental condition'

    data, rn, cn = load_source(join('dicty', 'dicty.gene_annnotations.csv.gz'))
#     data = resize_rows_and_columns(data, exp)
    data = uf.normalization(data)
    ann = Relation(data=data, x_name=gene, y_name=go_term, name='ann',
                   x_metadata=rn, y_metadata=cn)
    ann = resize_rows_and_columns(ann, exp)
    ann = sorted_relation_data(ann, sort);
    print(np.min(data))
    print(np.max(data))
    print()

    data, rn, cn = load_source(join('dicty', 'dicty.gene_expression.csv.gz'))
    expr = Relation(data=data, x_name=gene, y_name=exprc, name='expr',
                    x_metadata=rn, y_metadata=cn)
    expr.matrix = np.log(np.maximum(expr.matrix, np.finfo(np.float).eps))
#     expr.matrix = resize_rows_and_columns(expr.matrix, exp)
    expr.matrix = uf.normalization(expr.matrix)
    expr = resize_rows_and_columns(expr, exp)
    expr = sorted_relation_data(expr, sort);
    print(np.min(expr.matrix))
    print(np.max(expr.matrix))
    print()

    data, rn, cn = load_source(join('dicty', 'dicty.ppi.csv.gz'))
#     data = resize_rows_and_columns(data, exp)
    data = uf.normalization(data)
    ppi = Relation(data=data, x_name=gene, y_name=gene, name='ppi',
                   x_metadata=rn, y_metadata=cn)
    ppi = resize_rows_and_columns(ppi, exp)
    ppi = sorted_relation_data(ppi, sort);
    print(np.min(data))
    print(np.max(data))

    ann_t = ann.transpose()
    expr_t = expr.transpose()

    relationGraph = RelationGraph()
    # relationGraph.add_relations([ann, expr, ppi, ann_t, expr_t])
    relationGraph.add_relations([ann, expr, ppi])
    relationGraph.display_objects()
    return relationGraph

def load_pharma(sort=None):
    action='Action'
    pmid='PMID'
    depositor='Depositor'
    fingerprint='Fingerprint'
    depo_cat='Depositor category'
    chemical='Chemical'

    data, rn, cn = load_source(join('pharma', 'pharma.actions.csv.gz'))
    data = sorted_data(data, sort);
    actions = Relation(data=data, x_name=chemical, y_name=action, name='actions',
                       x_metadata=rn, y_metadata=cn)
    print(np.min(data))
    print(np.max(data))
    print()
    a = cn
#     data, rn, cn = load_source(join('pharma', 'pharma.pubmed.csv.gz'))
#     cn = replace_duplicate(set(rn) & set(cn), cn, '--')
#     data = sorted_data(data, sort);
#     pubmed = Relation(data=data, x_name=chemical, y_name=pmid, name='pudmed',
#                       x_metadata=rn, y_metadata=cn)
#     p = cn
#     print(np.min(data))
#     print(np.max(data))
#     print()

    data, rn, cn = load_source(join('pharma', 'pharma.depositors.csv.gz'))
    data = sorted_data(data, sort);
    depositors = Relation(data=data, x_name=chemical, y_name=depositor, name='depositors',
                          x_metadata=rn, y_metadata=cn)
    d = cn
    print(np.min(data))
    print(np.max(data))
    print()
    data, rn, cn = load_source(join('pharma', 'pharma.fingerprints.csv.gz'))
    cn = replace_duplicate(set(rn) & set(cn), cn, '**')
#     cn = replace_duplicate(set(p) & set(cn), cn, '**')
    data = sorted_data(data, sort);
    fingerprints = Relation(data=data, x_name=chemical, y_name=fingerprint, name='fingerprints',
                            x_metadata=rn, y_metadata=cn)
    f = cn
    print(np.min(data))
    print(np.max(data))
    print()
    data, rn, cn = load_source(join('pharma', 'pharma.depo_cats.csv.gz'))
    data = sorted_data(data, sort);
    depo_cats = Relation(data=data, x_name=depositor, y_name=depo_cat, name='depo_cats',
                         x_metadata=rn, y_metadata=cn)
    dc = cn
    print(np.min(data))
    print(np.max(data))
    print()
    data, rn, cn = load_source(join('pharma', 'pharma.tanimoto.csv.gz'))
    data = uf.normalization(data)
    data = sorted_data(data, sort);
    tanimoto = Relation(data=data, x_name=chemical, y_name=chemical, name='tanimoto',
                        x_metadata=rn, y_metadata=cn)
    c = cn
    print(np.min(data))
    print(np.max(data))
    print()
    actions_t = actions.transpose()
#     pubmed_t = pubmed.transpose()
    depositors_t = depositors.transpose()
    fingerprints_t = fingerprints.transpose()
    depo_cats_t = depo_cats.transpose()

    relationGraph = RelationGraph()
    relationGraph.add_relations([
        actions,
#         pubmed,
        depositors,
        fingerprints,
        depo_cats,
        tanimoto
    ])
    relationGraph.display_objects()
    return relationGraph

    
def replace_duplicate(duplicates, arr, term='--'):
    arr = np.array(arr, np.dtype('<U10'))
#     print(duplicates)
    for element in duplicates:
        idx = np.where(arr == element)
        arr[idx[0][0]] = str(element) + term
#         print(element)
#         print(idx)
#         print(arr[idx[0][0]])
#         print()
    return arr