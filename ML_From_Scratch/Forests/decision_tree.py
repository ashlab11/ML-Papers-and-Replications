import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
from CART import *
from RandomForest import *

# Create the dataset
data = {
    "Age": [25, 45, 34, 29, 52, 40, 33, 38, 28, 60],
    "Income": [30000, 80000, 45000, 60000, 120000, 75000, 40000, 50000, 70000, 95000],
    "Gender_Male": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    "Tenure": [2, 8, 5, 3, 15, 7, 4, 6, 2, 20],
    "ProductType_Basic": [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    "ProductType_Premium": [0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
    "Churn": [0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
}

dataframe = pd.DataFrame(data)
X = dataframe.drop(columns = "Churn")
y = dataframe['Churn']

tree = Tree(min_samples = 2, max_depth = 3)
tree.train(X, y)

forest = RandomForestClassifier()
forest.train(X, y)
forest.predict(X)