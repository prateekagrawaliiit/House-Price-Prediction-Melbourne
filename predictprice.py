# DEPENDENCIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('melb_data.csv')
# print(df.columns)
# print(df.shape)

# DataFrames have two axes (axis=0 corresponds to row amd axis=1 corresponds to columns)
df = df.dropna(axis=0)

X = np.asanyarray(
    df[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']])
y = np.asanyarray(df['Price'])

tree = DecisionTreeRegressor(random_state=1)
tree.fit(X, y)

# print("Making predictions for the following houses :")
# print(df.head())
# print("The predictions are :")
# print(tree.predict(X)[:5])


def get_mae(leaf_nodes, X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mean_error = mean_absolute_error(y_test, prediction)
    return mean_error


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# model = DecisionTreeRegressor()
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# print(prediction[:5])
# print(y_test[:5])
mae_array = {}
for nodes in [5, 100, 500, 1000]:
    mae = get_mae(nodes, X_train, y_train, X_test, y_test)
    mae_array[nodes] = mae

key_min = min(mae_array, key=(lambda k: mae_array[k]))

print("Minimum mean absolute error is ",
      mae_array[key_min], ' for max leaf nodes = ', key_min)


""" Can also be done like  
    scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
    best_tree_size = min(scores, key=scores.get)

"""
