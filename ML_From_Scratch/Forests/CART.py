import numpy as np
from scipy import stats as st


class Leaf:
    """Leaf node that predicts the mode of the data"""
    def __init__(self, y):
        self.y = y
        self.left = None
        self.right = None
    def split(self):
        return None
    def train(self):
        return None
    def predict(self, X_predict):
        return st.mode(self.y)[0]

class Node:
    def __init__(self, X, y, min_samples = 2, max_depth = 3, col_frac = 1, min_impurity_decrease = 0):
        self.X = np.array(X)
        self.y = np.array(y)
        self.left = None
        self.right = None
        self.splitting_col = None
        self.splitting_val = None
        self.splitting_criteria = None
        self.col_frac = col_frac
        self.min_samples = min_samples
        self.min_impurity_decrease = min_impurity_decrease
        if min_samples < 1 or type(min_samples) != int:
            raise("Min samples must be a positive integer")
        self.max_depth = max_depth
        if max_depth < 0 or max_depth == None or type(max_depth) != int:
            raise("Max depth must be a non-negative integer")
    def gini_impurity(self, vals):
        counts = np.unique(vals, return_counts = True)[1]
        return (1 - np.sum((counts / sum(counts))**2))
    def best_col_split(self, col_idx):
        """Split the data based on the column index"""
        best_idx = 0
        best_impurity = 1
        
        #Going through each index
        col_vals = self.X[:, col_idx]
        sorted_indices = np.argsort(col_vals)
        for sorted_idx in sorted_indices:
            left_sort = self.y[col_vals < col_vals[sorted_idx]]
            left_impurity = self.gini_impurity(left_sort)
            
            right_sort = self.y[col_vals >= col_vals[sorted_idx]]
            right_impurity = self.gini_impurity(right_sort)
            #print(sum(sorted_indices))
            impurity = (right_impurity * len(right_sort) + left_impurity * len(left_sort)) / (len(sorted_indices))
            if impurity < best_impurity:
                best_impurity = impurity
                best_idx = sorted_idx
        return (best_idx, best_impurity)
    def split(self):
        """Go through each column of the data and find the best split"""
        current_impurity = self.gini_impurity(self.y)
        best_col = 0
        best_idx = 0
        best_impurity = 1
        num_features = self.X.shape[1] # How many features X contains
        col_sample_size = round(self.col_frac * num_features)

        #Choosing a random fraction of col values
        for col in np.random.choice(num_features, col_sample_size, replace=False):
            (best_idx, best_impurity) = self.best_col_split(col)
            if best_impurity < best_impurity:
                best_col = col
                best_idx = best_idx
                best_impurity = best_impurity
        
        if (best_impurity == current_impurity):
            print("No impurity decrease. Created new leaf.")
            self = Leaf(self.y)
            return
        
        #We have the best column and index to split on, so we can now split the data!
        splitting_criteria = lambda x: x[best_col] < self.X[:, best_col][best_idx]
        self.splitting_col = best_col
        self.splitting_val = self.X[:, best_col][best_idx]
        self.splitting_criteria = splitting_criteria
        
        left_split = self.X[:, best_col] < self.X[:, best_col][best_idx]
        right_split = self.X[:, best_col] >= self.X[:, best_col][best_idx]
        
        print(left_split) 
        print(right_split)
        print("NEW")       
        if len(np.unique(left_split)) == 1 or len(left_split) <= self.min_samples or self.max_depth == 0:
            self.left = Leaf(self.y[left_split])
        else:
            self.left = Node(self.X[left_split], self.y[left_split], max_depth= self.max_depth - 1)
        if len(np.unique(right_split)) == 1 or len(right_split) <= self.min_samples or self.max_depth == 0:
            self.right = Leaf(self.y[right_split])
        else:
            self.right = Node(self.X[right_split], self.y[right_split], max_depth= self.max_depth - 1)
            
        return (self.left, self.right)
    def train(self):
        self.split()
        if self.left != None:
            self.left.train()
        if self.right != None:
            self.right.train()
    def predict(self, X_predict):
        """"We want to go through nodes until we reach a leaf node"""        
        if self.splitting_criteria(X_predict):
            return self.left.predict(X_predict)
        else:
            return self.right.predict(X_predict)
        
    
class Tree:
    def __init__(self, min_samples = 2, max_depth = 3):
        self.min_samples = min_samples
        self.max_depth = max_depth
    def train(self, X, y):
        self.root = Node(X, y, self.min_samples, self.max_depth - 1)
        self.root.train()        
    def predict(self, X_predict):
        if len(X_predict.shape) > 1: # Array is multidimensional
            predictions = np.apply_along_axis(self.root.predict, axis=1, arr=np.array(X_predict))
        else:
            predictions = self.root.predict(np.array(X_predict))
        return predictions
    