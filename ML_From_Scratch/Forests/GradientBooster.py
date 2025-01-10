import numpy as np
from CART import *
from scipy import stats as st

class GradientBoostingRegressor:
    def __init__(self, n_trees = 10, min_samples = 2, max_depth = 3, epsilon = 0.1):
        self.n_trees = n_trees
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.feature_frac = 1
        self.epsilon = epsilon
        self.residual_trees = []
    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        #Create initial tree
        self.initial_tree = Tree(min_samples = self.min_samples, max_depth = self.max_depth)
        self.initial_tree.train(X, y)
        curr_prediction = self.initial_tree.predict(X)
        
        #Construct other trees
        self.residual_trees = []
        for i in range(0, self.n_trees - 1):
            pseudo_residuals = curr_prediction - y
            next_tree = Tree(min_samples = self.min_samples, max_depth = self.max_depth)
            next_tree.train(X, pseudo_residuals)
            curr_prediction += self.epsilon * next_tree.predict(X)
            self.residual_trees.append(next_tree)
    def predict(self, X_predict):
        """Go through each tree"""
        predictions = self.initial_tree.predict(X_predict)
        for tree in self.residual_trees:
            predictions += self.epsilon * tree.predict(X_predict)
        return predictions