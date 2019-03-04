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

from skfusion.fusion import ObjectType, Relation, FusionGraph


__all__ = ['load_dicty', 'load_pharma', 'load_movielens']


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


def load_dicty():
    """Construct fusion graph from molecular biology of Dictyostelium."""
    gene = ObjectType('Gene', 50)
    go_term = ObjectType('GO term', 15)
    exprc = ObjectType('Experimental condition', 5)

    data, rn, cn = load_source(join('dicty', 'dicty.gene_annnotations.csv.gz'))
    ann = Relation(data=data, row_type=gene, col_type=go_term, name='ann',
                   row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('dicty', 'dicty.gene_expression.csv.gz'))
    expr = Relation(data=data, row_type=gene, col_type=exprc, name='expr',
                    row_names=rn, col_names=cn)
    expr.data = np.log(np.maximum(expr.data, np.finfo(np.float).eps))
    data, rn, cn = load_source(join('dicty', 'dicty.ppi.csv.gz'))
    ppi = Relation(data=data, row_type=gene, col_type=gene, name='ppi',
                   row_names=rn, col_names=cn)
    return FusionGraph([ann, expr, ppi])


def load_pharma():
    """Construct fusion graph from the pharmacology domain."""
    action=ObjectType('Action', 5)
    pmid=ObjectType('PMID', 5)
    depositor=ObjectType('Depositor', 5)
    fingerprint=ObjectType('Fingerprint', 20)
    depo_cat=ObjectType('Depositor category', 5)
    chemical=ObjectType('Chemical', 10)

    data, rn, cn = load_source(join('pharma', 'pharma.actions.csv.gz'))
    actions = Relation(data=data, row_type=chemical, col_type=action,
                       row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.pubmed.csv.gz'))
    pubmed = Relation(data=data, row_type=chemical, col_type=pmid,
                      row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.depositors.csv.gz'))
    depositors = Relation(data=data, row_type=chemical, col_type=depositor,
                          row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.fingerprints.csv.gz'))
    fingerprints = Relation(data=data, row_type=chemical, col_type=fingerprint,
                            row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.depo_cats.csv.gz'))
    depo_cats = Relation(data=data, row_type=depositor, col_type=depo_cat,
                         row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.tanimoto.csv.gz'))
    tanimoto = Relation(data=data, row_type=chemical, col_type=chemical,
                        row_names=rn, col_names=cn)
    return FusionGraph([actions, pubmed, depositors, fingerprints, depo_cats, tanimoto])


def load_movielens(ratings=True, movie_genres=True, movie_actors=True):
    module_path = join(dirname(__file__), 'data', 'movielens')
    if ratings:
        ratings_data = defaultdict(dict)
        with gzip.open(join(module_path, 'ratings.csv.gz'), 'rt', encoding='utf-8') as f:
            f.readline()
            for line in f:
                line = line.strip().split(',')
                ratings_data[int(line[0])][int(line[1])] = float(line[2])
    else:
        ratings_data = None

    if movie_genres:
        movie_genres_data = {}
        with gzip.open(join(module_path, 'movies.csv.gz'), 'rt', encoding='utf-8') as f:
            f.readline()
            lines = csv.reader(f)
            for line in lines:
                movie_genres_data[int(line[0])] = line[2].split('|')
    else:
        movie_genres_data = None

    if movie_actors:
        movie_actors_data = {}
        with gzip.open(join(module_path, 'actors.csv.gz'), 'rt', encoding='utf-8') as f:
            f.readline()
            lines = csv.reader(f)
            for line in lines:
                movie_actors_data[int(line[0])] = line[2].split('|')
    else:
        movie_actors_data = None
    return ratings_data, movie_genres_data, movie_actors_data


def show_data_dicty():
    data1, rn1, cn1 = load_source(join('dicty', 'dicty.gene_annnotations.csv.gz'))
    data2, rn2, cn2 = load_source(join('dicty', 'dicty.gene_expression.csv.gz'))
    data3, rn3, cn3 = load_source(join('dicty', 'dicty.ppi.csv.gz'))

    print('Gene - GO term')
    print('Data1: ' + str(data1.shape))
    print('Gene - Experiment conditions')
    print('Data2: ' + str(data2.shape))
    print('Gene - Gene')
    print('Data3: ' + str(data3.shape))
    print()

    print('Subset (data1 - data2): ' + str(len(set(rn1) & set(rn2))))
    print('Subset (data1 - data3): ' + str(len(set(rn1) & set(rn3))))
    print('Subset (data2 - data3): ' + str(len(set(rn2) & set(rn3))))
    print('Subset (data1 - data2 - data3): ' + str(len(set(rn1) & set(rn2) & set(rn3))))


def show_data_pharma():
    data1, rn1, cn1 = load_source(join('pharma', 'pharma.actions.csv.gz'))
    data2, rn2, cn2 = load_source(join('pharma', 'pharma.pubmed.csv.gz'))
    data3, rn3, cn3 = load_source(join('pharma', 'pharma.depositors.csv.gz'))
    data4, rn4, cn4 = load_source(join('pharma', 'pharma.fingerprints.csv.gz'))
    data5, rn5, cn5 = load_source(join('pharma', 'pharma.depo_cats.csv.gz'))
    data6, rn6, cn6 = load_source(join('pharma', 'pharma.tanimoto.csv.gz'))

    print("Chemical - Action")
    print('Data1: ' + str(data1.shape))
    print('Chemical - PMID')
    print('Data2: ' + str(data2.shape))
    print('Chemical - Depositor')
    print('Data3: ' + str(data3.shape))
    print('Chemical - Fingerprint')
    print('Data4: ' + str(data4.shape))
    print('Depositor - Depositor cateoory')
    print('Data5: ' + str(data5.shape))
    print('Chemical - chemical')
    print('Data6: ' + str(data6.shape))
    print()

    print('Subset(data1 - data2): ' + str(len(set(rn1) & set(rn2))))
    print('Subset(data1 - data3): ' + str(len(set(rn1) & set(rn3))))
    print('Subset(data1 - data4): ' + str(len(set(rn1) & set(rn4))))
    print('Subset(data1 - data6): ' + str(len(set(rn1) & set(rn6))))
    print('Subset(data3 - data5): ' + str(len(set(cn3) & set(rn5))))