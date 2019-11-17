import numpy as np
import json
from MyRegressor import *


class MyGradientBoostingRegressor():
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param learning_rate, type:float
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        int (default=100)
        :param n_estimators, type: integer
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        :param max_depth, type: integer
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node

        estimators: the regression estimators
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = np.empty((self.n_estimators,), dtype=np.object)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.initial_prediction = 0

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.estimators in this function
        '''

        # the initial prediction is the mean of labels
        current_prediction = np.array([y.mean()] * y.shape[0])
        self.initial_prediction = current_prediction

        # create trees to fit the residual
        for m in range(self.n_estimators):
            current_residual = y - current_prediction
            self.estimators[m] = MyDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            self.estimators[m].fit(X, current_residual)

            # pass the original data into the residual-fitting tree
            assigned_residual = self.estimators[m].predict(X)
            current_prediction += self.learning_rate * assigned_residual

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        results = []
        for data in X:
            result = 0
            for m in range(self.n_estimators):
                result += self.learning_rate * self.estimators[m].predict(data)
            results.append(result)
        return np.array(results)

    def get_model_dict(self):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})

        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)

