import numpy as np
import copy
import utilityFunctions as uf


class MatrixOfRelationGraph:

    def __init__(self, graph=None):
        self.graph = graph
        self.matrix_2D = np.zeros((0, 0))
        self.metadata_object_index = {}
        self.metadata_object_matrix = np.zeros((0, 0))
        self.metadata = []
        self.matrix_3D = np.zeros((1, 0, 0))
        self.x_index = {}
        self.y_index = {}

    def convert_to_2D_matrix(self):
        self.metadata_object_matrix = np.zeros((len(self.graph.objects), len(self.graph.objects)))
        self.metadata = [x[:] for x in [[None] * len(self.graph.objects)] * len(self.graph.objects)]

        elements = 0
        for index, obj in enumerate(self.graph.objects.values()):
            self.metadata_object_index[obj.name] = (index, (elements, elements + len(obj.elements)-1))
            elements += len(obj.elements)
            index += 1

        already_used = set()

        for obj in self.graph.objects.values():
            for relation in obj.get_all_relations():
                if relation.name in already_used:
                    continue

                already_used.add(relation.name)

                x, x_coordinate = self.metadata_object_index[relation.x_object]
                y, y_coordinate = self.metadata_object_index[relation.y_object]

                self.metadata_object_matrix[x][y] = 1
                self.metadata[x][y] = Metadata(relation.name,
                                               start=(x_coordinate[0], y_coordinate[0]),
                                               end=(x_coordinate[1], y_coordinate[1]))

                self.build_2D_matrix(relation)
                
                print(self.matrix_2D.shape)
                print()

    def build_2D_matrix(self, relation):
        print('-----------' + relation.name + ' ' + str(relation.matrix.shape) + '-----------')
        x_size, y_size = self.matrix_2D.shape

        if self.matrix_2D.shape == (0, 0):
            self.x_index = relation.x
            self.y_index = relation.y
            self.matrix_2D = relation.matrix
            return

        new_x = uf.find_new_values(uf.convert_dict_to_list(relation.x), uf.convert_dict_to_list(self.x_index))
        new_y = uf.find_new_values(uf.convert_dict_to_list(relation.y), uf.convert_dict_to_list(self.y_index))

        if len(new_x) == 0 and len(new_y) == 0:
            print('x == y')
            # inset data to matrix, find start and stop index
            # TODO en bug je, testiraj z simetricnimi matrikami
            relation = self.sort_rows(relation)
            relation = self.sort_columns(relation)

            list_x = uf.convert_dict_to_list(relation.x)
            list_y = uf.convert_dict_to_list(relation.y)

            first_x = relation.first_x()
            first_y = relation.first_y()
            last_x = relation.last_x()
            last_y = relation.last_y()

            if self.x_index[last_x] - self.x_index[first_x] + 1 != len(relation.x):
                raise ValueError('Dimenzija X se ne ujema: ' + str(
                    self.x_index[last_x] - self.x_index[first_x] + 1) + ' vs ' + str(len(relation.x)))

            if self.y_index[last_y] - self.y_index[first_y] + 1 != len(relation.y):
                raise ValueError('Dimenzija Y se ne ujema: ' + str(
                    self.y_index[last_y] - self.y_index[first_y] + 1) + ' vs ' + str(len(relation.y)))

#             if first_x != list_x[self.x_index[first_x]] or last_x != list_x[self.x_index[last_x]] or \
#                     first_y != list_y[self.y_index[first_y]] or last_y != list_y[self.y_index[last_y]]:
#                 raise ValueError('Indexi se ne ujemajo!')

            self.matrix_2D += relation.matrix

        elif len(new_x) == 0:
            print('x == 0')
            # resize matrix to y direction (add columns)
            relation = self.sort_rows(relation)
            y_list = relation.get_y_list()

            new_y = dict(zip(y_list, range(y_size, y_size + len(y_list))))

            self.y_index = uf.merge_two_dicts(self.y_index, new_y)
            self.matrix_2D = np.c_[self.matrix_2D, relation.matrix]

        elif len(new_y) == 0:
            print('y == 0')
            # resize matrix to x direction (add rows)
            relation = self.sort_columns(relation)
            x_list = relation.get_x_list()

            new_x = dict(zip(x_list, range(x_size, x_size + len(x_list))))

            self.x_index = uf.merge_two_dicts(self.x_index, new_x)
            self.matrix_2D = np.r_[self.matrix_2D, relation.matrix]

        else:
            print('x != y')
            # resize to both direction (add columns and rows)
            x_list = relation.get_x_list()
            y_list = relation.get_y_list()

            new_x = dict(zip(x_list, range(x_size, x_size + len(x_list))))
            new_y = dict(zip(y_list, range(y_size, y_size + len(y_list))))

            self.x_index = uf.merge_two_dicts(self.x_index, new_x)
            self.y_index = uf.merge_two_dicts(self.y_index, new_y)

            matrix_1 = np.c_[self.matrix_2D, np.zeros((x_size, len(new_y)))]
            matrix_2 = np.c_[np.zeros((len(new_x), y_size)), relation.matrix]

            self.matrix_2D = np.r_[matrix_1, matrix_2]

    def sort_rows(self, relation):
        tmp_relation = copy.copy(relation)

        subtracted = len(self.x_index) - len(tmp_relation.x)
        shift = 0
        if subtracted > 0:
            list_x = uf.convert_dict_to_list(self.x_index)[: subtracted + 1]

            for i in range(len(list_x)):
                if list_x[i] in tmp_relation.x.keys():
                    shift = i
                    break
                    
        already_swap = set()
        reversed_dict = dict((v, k) for k, v in tmp_relation.x.items())

        for key, value in tmp_relation.x.items():
            if value != self.x_index[key] and self.x_index[key] not in already_swap:
                # zamenjaj vrednosti v matriki
                tmp_relation.matrix = uf.swap_row(tmp_relation.matrix, value, self.x_index[key] - shift)

                # zamenjaj vrednosti v metedata
                old_key = reversed_dict[self.x_index[key] - shift]
                old_value = value
                new_key = reversed_dict[value]
                new_value = self.x_index[key] - shift

                tmp_relation.x[new_key] = new_value
                tmp_relation.x[old_key] = old_value

                reversed_dict[new_value] = new_key
                reversed_dict[old_value] = old_key

                already_swap.add(self.x_index[key])

        # resize output matrix and shift in right
        if subtracted > 0:
            if shift > 0:
                new_matrix = np.r_[np.zeros((shift, tmp_relation.matrix.shape[1])), tmp_relation.matrix]
                new_matrix = np.r_[new_matrix, np.zeros((subtracted - shift, tmp_relation.matrix.shape[1]))]
            else:
                new_matrix = np.r_[tmp_relation.matrix, np.zeros((subtracted, tmp_relation.matrix.shape[1]))]

            tmp_relation.matrix = new_matrix
            tmp_relation.x = dict((key, value + shift) for key, value in tmp_relation.x.items())

        return tmp_relation

    def sort_columns(self, relation):
        tmp_relation = copy.copy(relation)

        subtracted = len(self.y_index) - tmp_relation.matrix.shape[1]
        shift = 0
        if subtracted > 0:
            list_y = uf.convert_dict_to_list(self.y_index)[: subtracted + 1]

            for i in range(len(list_y)):
                if list_y[i] in tmp_relation.y.keys():
                    shift = i
                    break

        already_swap = set()
        reversed_dict = dict((v, k) for k, v in tmp_relation.y.items())
        
        for key, value in tmp_relation.y.items():
            if value != self.y_index[key] and self.y_index[key] not in already_swap:
                # zamenjaj vrednosti v matriki
                tmp_relation.matrix = uf.swap_column(tmp_relation.matrix, value, self.y_index[key] - shift)
                # zamenjaj vrednosti v metedata
                old_key = reversed_dict[self.y_index[key] - shift]
                old_value = value
                new_key = reversed_dict[value]
                new_value = self.y_index[key] - shift

                tmp_relation.y[new_key] = new_value
                tmp_relation.y[old_key] = old_value

                reversed_dict[new_value] = new_key
                reversed_dict[old_value] = old_key

                already_swap.add(self.y_index[key])

        # resize output matrix and shift in right
        if subtracted > 0:
            if shift > 0:
                new_matrix = np.c_[np.zeros((tmp_relation.matrix.shape[0], shift)), tmp_relation.matrix]
                new_matrix = np.c_[new_matrix, np.zeros((tmp_relation.matrix.shape[0], subtracted - shift))]
            else:
                new_matrix = np.c_[tmp_relation.matrix, np.zeros((tmp_relation.matrix.shape[0], subtracted))]

            tmp_relation.matrix = new_matrix
            tmp_relation.y = dict((key, value + shift) for key, value in tmp_relation.y.items())

        return tmp_relation

    def display_metadata_2D_matrix(self):
        print('-------------2D Matrix-------------')
        print('Objects: ' + ', '.join([key + ': ' + str(value) for key, value in self.metadata_object_index.items()]))

        for row in self.metadata_object_matrix:
            print(row)
        print()

    def density_data(self, density=0.1):
        rows_density = np.count_nonzero(self.matrix_2D, axis=0)

        idx = np.array([])
        for key, value in self.metadata_object_index.items():
            elements = dict(zip(range(value[1][0], value[1][1]), rows_density[value[1][0]:value[1][1]]))  # kljuc je pozicija gena v 2D matriki, vrednost je stevilo ne praznih vrstic
            top_highest = round(len(elements) * density)

            filtered_idx = uf.convert_dict_to_list(uf.n_high_values(elements, top_highest))
            idx = np.append(idx, filtered_idx)

            print(key + ': ' + str(len(filtered_idx)))

        mask = np.in1d(range(len(self.x_index)), idx, invert=True)
        delete_idx = np.array(range(len(self.x_index)))[mask]
        matrix = np.delete(self.matrix_2D, delete_idx, axis=1)
        matrix = np.delete(matrix, delete_idx, axis=0)

        return matrix

    def display_density_data(self):
        density = 0.3
        rows_density = np.count_nonzero(self.matrix_2D, axis=0)

        objects = {}
        for key, value in self.metadata_object_index.items():
            objects[key] = dict(zip(range(value[1][0], value[1][1]), rows_density[value[1][0]:value[1][1]]))   # kljuc je pozicija gena v 2D matriki, vrednost je stevilo ne praznih vrstic

        for key, value in objects.items():
            # objects[key] = uf.n_high_values(objects[key], round(len(objects[key]) * 0.1))
            print(len(objects[key]))
            print(round(len(objects[key]) * density))
            tmp = uf.n_high_values(objects[key], round(len(objects[key]) * density))
            l = np.asarray([x for x in tmp.values()])
            mask = np.where(l >= len(objects[key]))
            print('---------Object ' + key + '-------')
            print(tmp)
            print('Number of elements: ' + str(len(tmp)))
            print('Avg value: ' + str(np.mean(l)))
            print('Full rows: ' + str(len(l[mask[0]])))
            print()       



class Metadata:

    def __init__(self, name, start, end):
        self.object = name
        self.start = start
        self.end = end

    def contains_index(self, index):
        if index[0] >= self.start[0] and index[0] <= self.end[0] \
                and index[1] >= self.start[1] and index[1] <= self.end[1]:
            return True

        return False

    def show(self):
        print('-------------Metadata-------------')
        print('Object: ' + self.object)
        print('First element: ' + str(self.start))
        print('Last element: ' + str(self.end))
        print()


class RelationGraph:

    def __init__(self):
        self.objects = {}

    def display_objects(self):
        print('-------------RelationGraph-------------')
        for obj in self.objects.values():
            print(obj.name + '\t' + str(len(obj.elements)))
            print(str(len(obj.relation_x)) + '\t' + ', '.join(
                [x.name + '-' + str(x.matrix.shape) for x in obj.relation_x]))
            print(str(len(obj.relation_y)) + '\t' + ', '.join(
                [y.name + '-' + str(y.matrix.shape) for y in obj.relation_y]))
        print()
        
    def get_data(self):
        already = set()
        list_of_data =[]
        for obj in self.objects.values():        
            for relation in obj.relation_x:
                if relation.name not in already:
                    list_of_data.append(relation.matrix)
                    already.add(relation.name)

            for relation in obj.relation_y:
                if relation.name not in already:
                    list_of_data.append(relation.matrix)
                    already.add(relation.name)
                    
        return list_of_data
                    
                    
    def add_relation(self, relation):
        tmp_relation = copy.copy(relation)

        if relation.x_object == relation.y_object:
            if relation.x_object not in self.objects:
                tmp_x = uf.convert_dict_to_list(relation.x)
                tmp_y = uf.convert_dict_to_list(relation.y)
                new_elements = np.unique(np.append(tmp_x, tmp_y))
                obj = Object(relation.x_object, new_elements)
                self.objects[relation.x_object] = obj

                # update x, first need sort
                new_elements = uf.find_new_values(obj.elements, tmp_x)
                new_x = dict(zip(new_elements, range(len(tmp_x), len(tmp_x) + len(new_elements))))
                tmp_relation.x = uf.merge_two_dicts(relation.x, new_x)

                # update y, first need sort
                new_elements = uf.find_new_values(obj.elements, tmp_y)
                new_y = dict(zip(new_elements, range(len(tmp_y), len(tmp_y) + len(new_elements))))
                tmp_relation.y = uf.merge_two_dicts(relation.y, new_y)

                # update matrix
                tmp_relation.matrix = np.r_[tmp_relation.matrix, np.zeros((len(new_x), relation.matrix.shape[1]))]
                tmp_relation.matrix = np.c_[tmp_relation.matrix, np.zeros((relation.matrix.shape[0], len(new_y)))]

                obj.add_relation_x(tmp_relation)
                # obj.add_relation_y(entity)
                return

            else:
                obj = self.objects[relation.x_object]
                tmp_x = np.asarray(uf.convert_dict_to_list(relation.x))
                tmp_y = np.asarray(uf.convert_dict_to_list(relation.y))

                new_x = uf.find_new_values(tmp_x, obj.elements)
                new_y = uf.find_new_values(tmp_y, obj.elements)

                obj.elements = np.append(obj.elements, np.unique(np.append(new_x, new_y)))
                new_elements = np.unique(np.append(new_x, new_y))

                obj.resize_relations_x(new_elements)
                obj.resize_relations_y(new_elements)

                # update x
                new_elements = uf.find_new_values(obj.elements, tmp_x)
                new_x = dict(
                    zip(new_elements, range(relation.matrix.shape[0], len(new_elements) + relation.matrix.shape[0])))
                tmp_relation.x = uf.merge_two_dicts(relation.x, new_x)

                # update y
                new_elements = uf.find_new_values(obj.elements, tmp_y)
                new_y = dict(
                    zip(new_elements, range(relation.matrix.shape[1], len(new_elements) + relation.matrix.shape[1])))
                tmp_relation.y = uf.merge_two_dicts(tmp_relation.y, new_y)

                # update matrix
                tmp_relation.matrix = np.r_[tmp_relation.matrix, np.zeros((len(new_x), tmp_relation.matrix.shape[1]))]
                tmp_relation.matrix = np.c_[tmp_relation.matrix, np.zeros((tmp_relation.matrix.shape[0], len(new_y)))]

                obj.relation_x.append(tmp_relation)
                # obj.relation_y.append(entity)
                return

        if relation.x_object not in self.objects:
            obj = Object(relation.x_object, np.asarray(uf.convert_dict_to_list(relation.x)))
            obj.add_relation_x(tmp_relation)
            self.objects[relation.x_object] = obj
        else:
            obj = self.objects[relation.x_object]
            tmp_x = np.asarray(uf.convert_dict_to_list(relation.x))

            new_elements = uf.find_new_values(tmp_x, obj.elements)
            obj.elements = np.append(obj.elements, new_elements)

            obj.resize_relations_x(new_elements)
            obj.resize_relations_y(new_elements)

            # update x
            new_elements = uf.find_new_values(obj.elements, tmp_x)
            new_x = dict(zip(new_elements, range(len(tmp_x), len(tmp_x) + len(new_elements))))
            tmp_relation.x = uf.merge_two_dicts(tmp_relation.x, new_x)

            tmp_relation.matrix = np.r_[tmp_relation.matrix, np.zeros((len(new_x), tmp_relation.matrix.shape[1]))]
            obj.relation_x.append(tmp_relation)

        if relation.y_object not in self.objects:
            obj = Object(tmp_relation.y_object, np.asarray(uf.convert_dict_to_list(relation.y)))
            obj.add_relation_y(tmp_relation)
            self.objects[relation.y_object] = obj
        else:
            obj = self.objects[relation.y_object]
            tmp_y = np.asarray(uf.convert_dict_to_list(relation.y))

            new_elements = uf.find_new_values(tmp_y, obj.elements)
            obj.elements = np.append(obj.elements, new_elements)

            obj.resize_relations_x(new_elements)
            obj.resize_relations_y(new_elements)

            # update y
            new_elements = uf.find_new_values(obj.elements, tmp_y)
            new_y = dict(zip(new_elements, range(len(tmp_y), len(tmp_y) + len(new_elements))))
            tmp_relation.y = uf.merge_two_dicts(tmp_relation.y, new_y)

            tmp_relation.matrix = np.c_[tmp_relation.matrix, np.zeros((relation.matrix.shape[0], len(new_y)))]
            obj.relation_y.append(tmp_relation)

    def add_relations(self, list_of_relations):
        if not uf.check_object_type(list_of_relations, 'list'):
            raise ValueError('Parameter must be type \'list\'!')

        for r in list_of_relations:
            if not uf.check_object_type(r, 'relation'):
                raise ValueError('Element does not type \'Relation\'!')

            self.add_relation(r)


class Object:

    def __init__(self, name, elements=None):
        self.name = name
        self.relation_x = []
        self.relation_y = []
        self.elements = []

        if elements is not None:
            self.elements = elements

    def add_relation_x(self, relation):
        if not uf.check_object_type(relation, 'relation'):
            return

        self.relation_x.append(relation)

    def add_relation_y(self, relation):
        if not uf.check_object_type(relation, 'relation'):
            return

        self.relation_y.append(relation)

    def resize_relations_x(self, new_elements):
        for relation in self.relation_x:
            # find new elements
            new_x = dict(zip(new_elements, range(len(relation.x), len(relation.x) + len(new_elements))))
            relation.x = uf.merge_two_dicts(relation.x, new_x)

            # resize matrix to X dimension
            relation.matrix = np.r_[relation.matrix, np.zeros((len(new_elements), relation.matrix.shape[1]))]

            if relation.x_object == relation.y_object:
                # resize matrix to Y dimension with same new elements
                new_y = dict(zip(new_elements, range(len(relation.y), len(relation.y) + len(new_elements))))
                relation.y = uf.merge_two_dicts(relation.y, new_y)

                relation.matrix = np.c_[relation.matrix, np.zeros((relation.matrix.shape[0], len(new_y)))]

    def resize_relations_y(self, new_elements):
        for relation in self.relation_y:
            # find new elements
            new_y = dict(zip(new_elements, range(len(relation.y), len(relation.y) + len(new_elements))))
            relation.y = uf.merge_two_dicts(relation.y, new_y)

            # resize matrix to Y dimension
            relation.matrix = np.c_[relation.matrix, np.zeros((relation.matrix.shape[0], len(new_y)))]

            if relation.x_object == relation.y_object:
                # resize matrix to X dimension with same new elements
                new_x = dict(zip(new_elements, range(len(relation.x), len(relation.x) + len(new_elements))))
                relation.x = uf.merge_two_dicts(relation.x, new_x)

                relation.matrix = np.r_[relation.matrix, np.zeros((len(new_elements), relation.matrix.shape[1]))]

    def display_releations(self, param=''):
        if param.lower() == 'x':
            print('Relations X: ' + str(len(self.relation_x)))
            for r in self.relation_x:
                r.show()

        elif param.lower() == 'y':
            print('Relations Y: ' + str(len(self.relation_y)))
            for r in self.relation_y:
                r.show()

        else:
            self.display_relations('x')
            self.display_releations('y')

    def get_relation_by_name(self, name=None):
        for relation in self.relation_x:
            if relation.name.lower() == name.lower():
                return relation

        for relation in self.relation_y:
            if relation.name.lower() == name.lower():
                return relation

    def get_all_relations(self):
        if len(self.relation_x) > 0 and len(self.relation_y) > 0:
            return self.relation_x + self.relation_y
        elif len(self.relation_x) > 0:
            return self.relation_x
        else:
            return self.relation_y


class Relation:

    def __init__(self, name=None, data=np.array((0, 0)), x_metadata={}, y_metadata={}, x_name=None, y_name=None):
        self.name = name
        self.matrix = data
        self.x_object = x_name
        self.y_object = y_name
        self.x = dict(zip(x_metadata, range(len(x_metadata))))
        self.y = dict(zip(y_metadata, range(len(y_metadata))))
        self.matrix = data

    def merge(self, new_relation, new_name):
        if not uf.check_object_type(new_relation, 'relation'):
            return

        if self.x_object == self.y_object:
            new_relation = new_relation.transpose()

        new_x = list(set(new_relation.x.keys()) - set(self.x.keys()))
        new_y = list(set(new_relation.y.keys()) - set(self.y.keys()))

        # update metadata
        self.x = self.add_element_to_dict(self.x, new_x)
        self.y = self.add_element_to_dict(self.y, new_y)

        # update matrix
        self.matrix = self.init_matrix()
        self.matrix = self.update_matrix_values(new_relation)
        self.name = new_name

    def transpose(self):
        t_relation = Relation(self.name + '_T', None, self.y_object, self.x_object)
        t_relation.y = self.x
        t_relation.x = self.y
        t_relation.x_object = self.y_object
        t_relation.y_object = self.x_object
        t_relation.matrix = self.matrix.T
        # relation doesn't used

        return t_relation

    def first_x(self):
        return min(self.x, key=self.x.get)

    def first_y(self):
        return min(self.y, key=self.y.get)

    def last_x(self):
        return max(self.x, key=self.x.get)

    def last_y(self):
        return max(self.y, key=self.y.get)

    def get_x_list(self):
        # return list keys, sorted by value
        return sorted(self.x, key=self.x.get)

    def get_y_list(self):
        # return list keys, sorted by value
        return sorted(self.y, key=self.y.get)

    def get_non_zero_pair(self):
        indices = uf.get_non_zeros_indices(self.matrix)
        x = self.get_x_list()
        y = self.get_y_list()

        return np.asarray([[x[indices[i, 0]], y[indices[i, 1]]] for i in range(indices.shape[0])])

    def show(self):
        print(self.name + '\t' + str(self.matrix.shape) + '\t' + '(' + self.x_object + ', ' + self.y_object + ')')

    def display_data(self):
        print('Name: ' + self.name)
        print('Dimensions: ' + str(self.matrix.shape))
        print()

