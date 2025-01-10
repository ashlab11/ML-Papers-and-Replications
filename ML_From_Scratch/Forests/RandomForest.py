import numpy as np
from CART import *
from scipy import stats as st


class RandomForestClassifier:
    def __init__(self, n_trees = 10, min_samples = 2, max_depth = 3, feature_frac = 0.5):
        self.n_trees = n_trees
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.feature_frac = feature_frac
        self.trees = []
    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        """Basic random forest, we want to make it more complicated eventually"""
        for i in range(self.n_trees):
            self.trees.append(Tree(min_samples = self.min_samples, max_depth = self.max_depth))
            bootstrapped_idx = np.random.choice(len(X), len(X), replace = True)
            X_bootstrapped = X[bootstrapped_idx, :]
            y_bootstrapped = y[bootstrapped_idx]
            self.trees[i].train(X_bootstrapped, y_bootstrapped)
    def predict(self, X_predict):
        """We want to go through each tree"""
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X_predict))
        return st.mode(predictions)[0]
    
class RandomForestRegressor:
    pass
        