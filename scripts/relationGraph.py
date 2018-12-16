import numpy as np
import copy
import utilityFunctions as uf


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
        t_entity = Relation(self.name, None, self.y_object, self.x_object)
        t_entity.y = self.x
        t_entity.x = self.y
        t_entity.matrix = self.matrix.T
        # relation doesn't used

        return t_entity

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

