import numpy as np
import matplotlib.pyplot as plt
from Layer_Dense import *
from Activations import *
from ErrorFunctions import *
from Network import Network
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

np.random.seed(0)

def create_regression_data(n_points, n_features):
    X = np.random.rand(n_points, n_features)
    y = X[:, 0] * 5 + np.sign(X[:, 1]) + np.sin(X[:, 2]) + 1 + np.random.randn(n_points)
    return X, y


#We're using row terminology -- each layer is a row
X, y = create_regression_data(1000, 3)
X_test, y_test = create_regression_data(100, 3)

#Run xgboost on this as well
param_dist = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [1.0, 1.5, 2.0]
}

search = RandomizedSearchCV(xgb.XGBRegressor(), param_dist, n_iter = 50, n_jobs = -1, cv = 3, verbose = 1, random_state = 0)
search.fit(X, y)
predictions = search.predict(X_test)

print(np.mean((predictions - y_test) ** 2))

#What we want the network to look like
"""network = Network(layers = [
    Layer_Input(3), #Input layer
    Layer_Dense(3, 10, Activation_ReLU()),
    Layer_Dense(10, 10, Activation_ReLU()),
    Layer_Dense(10, 1, Activation_None()), #Output layer
], error_func = MSE())

network.train(X, y, epochs=1000)
predictions = network.predict(X_test)
print(np.mean((predictions - y_test) ** 2))"""