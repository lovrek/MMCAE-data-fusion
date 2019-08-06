import numpy as np
import random as r
import relationGraph as rg
import tempfile
import multiprocessing
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

def swap_column(matrix, frm, to):
    # swap columns
    copy = matrix[:, frm].copy()
    matrix[:, frm] = matrix[:, to]
    matrix[:, to] = copy
    return matrix


def swap_row(matrix, frm, to):
    # swap rows
    copy = matrix[frm, :].copy()
    matrix[frm, :] = matrix[to, :]
    matrix[to, :] = copy
    return matrix


def convert_dict_to_list(dict):
    return sorted(dict, key=dict.get)


def find_new_values(target, source):
    target = np.asarray(target)
    source = np.asarray(source)
    mask = np.in1d(target, source, invert=True)
    return target[mask]


def merge_two_dicts(old, new):
    tmp = old.copy()
    tmp.update(new)
    return tmp


def write_3D_matrix_to_file(file_name, data=None):
    # Generate some test data
    if data is None:
        data = np.arange(200).reshape((4, 5, 10))

    with open('./../data/' + file_name, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            # np.savetxt(outfile, data_slice, fmt='%-7.2f')
            np.savetxt(outfile, data_slice)

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')


def read_3D_matrix_from_file(file_name):
    with open('./../data/' + file_name,  'r') as file:
        line = file.readline()
        start = line.index('(')
        end = line.index(')')
        reshape_index = line[start+1:end].split(',')

    # Read the array from disk
    new_data = np.loadtxt('./../data/' + file_name)

    # Note that this returned a 2D array!
    # print(new_data.shape)

    # However, going back to 3D is easy if we know the
    # original shape of the array
    new_data = new_data.reshape((int(reshape_index[0].strip()),
                                 int(reshape_index[1].strip()),
                                 int(reshape_index[2].strip())))

    # Just to check that they're the same...
    # assert np.all(new_data == data)
    return new_data


def get_non_zeros_indices(matrix):
    x, y = np.where(matrix != 0)

    return np.c_[x, y]


def sample_generator(matrix, dismiss_elements=[], num_of_samples=1):
    # too slower method O(n^2)
    # zastarelo, za enkrat se ne uporablja
    non_zeros = np.count_nonzero(matrix)
    sample_density = 0.8             # 0.8 -> 80%
    x, y = matrix.shape
    indices = get_non_zeros_indices(matrix)
    elements_in_sample = round(non_zeros*sample_density)
    samples = np.zeros((num_of_samples, x, y))

    print('Number of elements in sample: ' + str(round(non_zeros*sample_density)))
    counter = 1
    while num_of_samples > 0:
        inserted = 0
        already_used = set() | set(dismiss_elements)
        while elements_in_sample > inserted:
            random = int(r.random()*non_zeros)
            if random not in already_used:
                random_x, random_y = indices[random]
                samples[num_of_samples-1, random_x, random_y] = matrix[random_x, random_y]
                inserted += 1
                already_used.add(random)

        if counter % 100 == 0:
            print('Number of generated samples: ' + str(counter))
            print()
        counter += 1
        num_of_samples -= 1

    print('Total samples: ' + str(samples.shape[0]))

    return samples


def sample_generator2(matrix, num_of_samples=1, density=0.8):
    # vhod v NN je vrsica v matriki - opustiom, porebno bi bilo urejanje po sosednostih
    x_size, y_size = matrix.shape
    data = np.empty((0, y_size))

    i = 1
    for row in matrix:
        counter = 0
        while counter < num_of_samples:
            tmp_row = np.copy(row)
            indices = np.random.choice(y_size, round(y_size * density), replace=False)
            mask = np.isin(np.arange(y_size), indices, invert=True)
            tmp_row[mask] = 0
            data = np.r_[data, tmp_row.reshape(1, y_size)]
        
            if i % 100 == 0:
                print(i)
            i += 1
            counter += 1

    return data


def sample_generator3(matrix, num_of_samples=1, density=0.8):
    # vhod v NN je celotna matrika
    x_size, y_size = matrix.shape
    data = np.empty((0, x_size * y_size))
    print('Random seed: '  + str(np.random.randint(y_size)))
    
    while num_of_samples > 0:
        tmp_matrix = np.copy(matrix)
        for row in tmp_matrix:
            indices = np.random.choice(y_size, round(y_size * density), replace=False)
            mask = np.isin(np.arange(y_size), indices, invert=True)
            row[mask] = 0
            
        data = np.r_[data, tmp_matrix.reshape(1, x_size * y_size)]                
        num_of_samples -= 1

    return data


def set_test_matrix(matrix, num_of_elements=10):
    x, y = matrix.shape
    test_matrix = np.zeros((1, x, y))
    indices = get_non_zeros_indices(matrix)
    already_used = set()

    while num_of_elements > 0:
        random = int(r.random()*indices.shape[0])
        if random not in already_used:
            random_x, random_y = indices[random]
            # save cell for test
            test_matrix[0, random_x, random_y] = matrix[random_x, random_y]
            # remove cel from original matrix - this cells not use for input to NN
            # self.input_matrix[random_x, random_y] = 0
            already_used.add(random)
            num_of_elements -= 1

    return test_matrix


def get_related_data(list_entities, num_of_relation=1):
        if not check_object_type(list_entities, 'list'):
            raise ValueError('Parameter must be type \'list\'!')

        for r in list_entities:
            if not check_object_type(r, 'relation'):
                raise ValueError('Element does not type \'Relation\'!')

        already_used = set()
        all_elements = {}
        for e in list_entities:

            if e.x_object == e.y_object or e.name in already_used:
                continue

            print(e.matrix.shape)
            already_used.add(e.name)
            new_element = np.unique(np.append([*e.x], [*e.y]))

            for element in (set([*all_elements]) & set(new_element)):
                all_elements[element] += 1

            if len(all_elements) == 0:
                all_elements = dict.fromkeys(new_element, 0)
            elif len(new_element) > 0:
                all_elements = merge_two_dicts(all_elements, dict.fromkeys(set(new_element) - set([*all_elements]), 0))

        # print(len([k for k, v in all_elements.items() if v > 0]))
        # print([(k, v) for k, v in all_elements.items() if v > 0])
        # print()
        #
        print('Related data: ')
        print(len([k for k, v in all_elements.items() if v >= num_of_relation]))
        print([(k, v) for k, v in all_elements.items() if v >= num_of_relation])
        print()
        # print(len([k for k, v in all_elements.items() if v > 2]))
        #
        # print()
        # print('All elements: ' + str(count))

        return np.asarray([k for k, v in all_elements.items() if v >= num_of_relation])


def check_object_type(object, type_name):
    if type_name == 'relation' and isinstance(object, (type, rg.Relation)):
        return True

    if type_name == 'object' and isinstance(object, (type, rg.Object)):
        return True

    if type_name == 'RelationGraph' and isinstance(object, (type, rg.RelationGraph)):
        return True

    if type_name == 'list' and isinstance(object, (type, list)):
        return True

    return False


def random_generate_sample(matrix, x_size, y_size):
    # randomly remove some columns and rows
    x, y = matrix.shape
    sample = matrix

    if x_size < x:
        x_indices = np.random.choice(x, x_size, replace=False)
        mask = np.isin(np.arange(x), x_indices, invert=True)
        remove_x = np.arange(x)[mask]
        sample = np.delete(sample, remove_x, axis=0)

    if x_size < y:
        y_indices = np.random.choice(y, y_size, replace=False)
        mask = np.isin(np.arange(y), y_indices, invert=True)
        remove_y = np.arange(y)[mask]
        sample = np.delete(sample, remove_y, axis=1)

    return sample


def assessment_results(old_matrix, new_matrix, eps=0.1):
    # old_matrix = np.power(old_matrix, 2)
    # new_matrix = np.power(new_matrix, 2)
    tmp_res = np.subtract(new_matrix, old_matrix)

    # print(np.subtract(new_matrix, old_matrix))

    tmp_res[tmp_res < eps] = 1
    tmp_res[tmp_res >= eps] = 0

    x, y = old_matrix.shape

    print('Number of successed: ' + str(np.count_nonzero(tmp_res)))
    print('Number of faild: ' + str((x * y) - np.count_nonzero(tmp_res)))
    print('All elements: ' + str(x * y))


def n_high_values(d, num=10):
    if len(d)-1 < num:
        return d

    l = sorted(d.items(), key=lambda x: x[1])
    l.reverse()
    return dict(l[:num])

class my_savez(object):
    def __init__(self, file):
        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        import zipfile
        # Import deferred for startup time improvement
        import tempfile
        import os

        if isinstance(file, str):
            if not file.endswith('.npz'):
                file = file + '.npz'

        compression = zipfile.ZIP_STORED

        zip = self.zipfile_factory(file, mode="w", compression=compression)

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
        os.close(fd)

        self.tmpfile = tmpfile
        self.zip = zip
        self.i = 0

    def zipfile_factory(self, *args, **kwargs):
        import zipfile
        import sys
        if sys.version_info >= (2, 5):
            kwargs['allowZip64'] = True
        return zipfile.ZipFile(*args, **kwargs)

    def savez(self, *args, **kwds):
        import os
        import numpy.lib.format as format

        namedict = kwds
        for val in args:
            key = 'arr_%d' % self.i
            if key in namedict.keys():
                raise ValueError("Cannot use un-named variables and keyword %s" % key)
            namedict[key] = val
            self.i += 1

        try:
            for key, val in namedict.items():
                fname = key + '.npy'
                fid = open(self.tmpfile, 'wb')
                try:
                    format.write_array(fid, np.asanyarray(val))
                    fid.close()
                    fid = None
                    self.zip.write(self.tmpfile, arcname=fname)
                finally:
                    if fid:
                        fid.close()
        finally:
            os.remove(self.tmpfile)

    def close(self):
        self.zip.close()

# tmp = '/mag/test.npz'
# f = my_savez(tmp)
# for i in range(10):
#   array = np.zeros(10)
#   f.savez(array)
# f.close()

# # tmp.seek(0)

# tmp_read = np.load(tmp)
# print (tmp_read.files)
# for k, v in tmp_read.iteritems():
#      print (k, v)

#------------------------------------
# Multiprocessing - generate samples
def mp_worker(arr):
#     new_data = uf.sample_generator3(data, num_of_samples=100, density=0.7, 1)
    np.random.seed(arr[3])
    new_data = sample_generator3(arr[0], num_of_samples=arr[1], density=arr[2])
    return new_data


def data_generator(data, n_samples, pools, density, filename):
    batch_size = 100
    p = multiprocessing.Pool(pools)
    gen_samples = np.empty((0, data.shape[0] * data.shape[1]))
    iterations = int(np.round((n_samples -1)/batch_size)) + 1
    
    params = [[data, batch_size, density, i] for i in range(iterations)]
    
    i = 1
    f = my_savez(filename)
    for result in p.imap(mp_worker, params):
        print('samples: ' + str(i * batch_size))
        i+=1
        f.savez(result)
    f.close()

    
def normalization(data, _min=0, _max=1):
    if _min >= _max:
        raise ValueError('Attribute \'min\' must be lower than \'min\'.')
    if _min > 0 or _max < 0:
        raise ValueError('This operation is not supported!')
    
    min_val = np.min(data)    
    if min_val < 0:
        data = data + np.abs(min_val)
        data[np.where(data == np.abs(min_val))] = 0
        
    max_val = np.max(data)
    if max_val > 1:
        data = data / max_val
    elif max_val < 1:
        factor = 1/max_val
        data = data * factor
        
    return data

# list of matrices, only data
def get_org_data(graph):
    org_data = []
    already = set()
    for obj in graph.objects.values():        
        for relation in obj.relation_x:
            if relation.name not in already:
                a,b = relation.matrix.shape
                org_data.append(np.array(relation.matrix).reshape(1,a,b,1))
                already.add(relation.name)

        for relation in obj.relation_y:
            if relation.name not in already:
                a,b = relation.matrix.shape
                org_data.append(np.array(relation.matrix).reshape(1,a,b,1))
                already.add(relation.name)
    
    return org_data

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="single"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    methods = ["ward","single","average","complete"]
    if method not in methods:
        raise Exception('Parameter \'method\' must be one of the following: ' + str(methods))
    
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

def order_by_clustering(data, method, direction='all'):
    if direction == 'row':
        return clustering(data, method)
    elif direction == 'col':
        return clustering(data.T, method)[0].T
    elif direction == 'all':
        tmp_data = clustering(data, method)
        return clustering(tmp_data.T, method)[0].T
    else:
        raise Exception('Parameter \'direction\' must have value row, col or all')
        
def clustering(data, method):    
    row, col = data.shape
    dist_mat = squareform(pdist(data))
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat,method)

    new_data = np.empty((row, col))    
    for i, val in enumerate(res_order):
        new_data[i] = data[val]
    
    return new_data, res_order

# latest version for generated sameples
def sample_generator4(matrix, num_of_samples=1, dismissed=0.2):
    # input  => sample_generator4([1219, 116], 1000, 0.8)
    # output => [1000, 1219, 116, 1]
    x_size, y_size = matrix.shape
    data = np.zeros((0, x_size, y_size, 1))
    print('Random seed: '  + str(np.random.randint(y_size)))
    
    while num_of_samples > 0:
        tmp_matrix = np.copy(matrix)
        
        tmp_matrix = tmp_matrix.flatten()
        nonzeros = np.nonzero(tmp_matrix)[0]
        dismiss = np.ceil(len(nonzeros) * dismissed)
        
        if dismissed < 1 and np.ceil(len(nonzeros) * dismissed) == len(nonzeros):
            dismiss = np.ceil(len(nonzeros) * dismissed) - 1
        
        idx = np.random.choice(len(nonzeros), int(dismiss), replace=False)
        mask = np.isin(nonzeros, idx)
        remove = nonzeros[mask]
        tmp_matrix[remove] = 0 
            
        data = np.r_[data, tmp_matrix.reshape(1, x_size, y_size, 1)]                
        num_of_samples -= 1

    return data
