import numpy as np
import random as r
import relationGraph as rg


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
    counter = 1
    
    while num_of_samples > 0:
        tmp_matrix = np.copy(matrix)
        for row in tmp_matrix:
            indices = np.random.choice(y_size, round(y_size * density), replace=False)
            mask = np.isin(np.arange(y_size), indices, invert=True)
            row[mask] = 0
            
        data = np.r_[data, tmp_matrix.reshape(1, x_size * y_size)]                
        
        if counter % 100 == 0:
            print(counter)
        num_of_samples -= 1
        counter += 1

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
