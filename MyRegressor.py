import json
import numpy as np


class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param max_depth, type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        if min_samples_split < 2:
            print("min_samples_split has to be at least 2!")
            exit(1)
        self.min_samples_split = min_samples_split
        self.root = {}

    def calculate_loss(self, g1, g2):
        group1, y1 = g1
        group1_sum = 0.0
        for index in range(len(group1)):
            group1_sum += y1[index]
        group1_c = group1_sum / len(group1)

        group2, y2 = g2
        group2_sum = 0.0
        for index in range(len(group2)):
            group2_sum += y2[index]
        group2_c = group2_sum / len(group2)

        result = 0.0
        for index in range(len(group1)):
            result += (y1[index] - group1_c) ** 2
        for index in range(len(group2)):
            result += (y2[index] - group2_c) ** 2

        return result

    def split_on_value(self, data, label, feature_index, threshold):
        left_index = []
        right_index = []
        for i in range(len(data)):
            if data[i][feature_index] <= threshold:
                left_index.append(i)
            else:
                right_index.append(i)
        return [data[left_index, :], label[left_index]], [data[right_index, :], label[right_index]]

    def find_best_split(self, train_data, ground_truth):
        # place holder for recording best result
        current_best_variable = None
        current_lowest_loss = np.inf

        # go through every feature variable
        num_feature = len(train_data[0])
        for splitting_variable in range(num_feature):
            # go through every possible splitting value
            value_set = sorted(set(train_data[:, splitting_variable]))[:-1]
            for splitting_threshold in value_set:
                # split the data based on threshold
                tmp_left_items, tmp_right_items = self.split_on_value(train_data, ground_truth, splitting_variable, splitting_threshold)

                # calculate loss value if no group are empty
                loss = self.calculate_loss(tmp_left_items, tmp_right_items)
                # update best variable and loss
                if loss < current_lowest_loss:
                    current_lowest_loss = loss
                    current_best_variable = [splitting_variable, splitting_threshold]

        # get the best result here and split data based on it
        splitting_variable, splitting_threshold = current_best_variable
        left_items, right_items = self.split_on_value(train_data, ground_truth, splitting_variable, splitting_threshold)

        return splitting_variable, splitting_threshold, left_items, right_items

    def split_node(self, current_node, current_depth):
        current_depth += 1
        do_split = False

        # read in the items at node
        try:
            [train_data, label] = current_node["items"]
            del(current_node["items"])
            do_split = True
        except KeyError:
            do_split = False

        if do_split:
            # find the best split for the data
            splitting_variable, splitting_threshold, left_items, right_items = self.find_best_split(train_data, label)

            # record into tree
            current_node["splitting_variable"] = splitting_variable
            current_node["splitting_threshold"] = splitting_threshold

            # go left
            if len(left_items[0]) < self.min_samples_split or current_depth == self.max_depth:
                current_node["left"] = left_items[1].mean()
            else:
                current_node["left"] = {}
                current_node["left"]["items"] = left_items
                # continue to split
                self.split_node(current_node["left"], current_depth)

            # go right
            if len(right_items[0]) < self.min_samples_split or current_depth == self.max_depth:
                current_node["right"] = right_items[1].mean()
            else:
                current_node["right"] = {}
                current_node["right"]["items"] = right_items
                # continue to split
                self.split_node(current_node["right"], current_depth)

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''

        # define the un-split root node, only contains training samples
        self.root["items"] = [X, y]

        # call split_node, which will recursively call itself to split every node until
        self.split_node(self.root, 0)

    def tree_predict(self, node, data):
        # traverse tree recursively and predict
        if data[node["splitting_variable"]] <= node["splitting_threshold"]:
            # go left
            if isinstance(node["left"], dict):
                # go further down
                return self.tree_predict(node["left"], data)
            else:
                # reached leaf
                return node["left"]
        else:
            # go right
            if isinstance(node["right"], dict):
                # go further down
                return self.tree_predict(node["right"], data)
            else:
                # reached leaf
                return node["right"]

    def predict(self, X):
        '''
        :param X: Feature data, python list or 2d array or numpy array (N, num_features)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        X = np.array(X)
        if X.ndim == 1:
            return self.tree_predict(self.root, X)        
        else:
            result = []
            for data in X:
                result.append(self.tree_predict(self.root, data))
            return np.array(result)

    def get_model_dict(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)
