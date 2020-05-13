#DEPENDENCIES 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('melb_data.csv')
# print(df.columns)
# print(df.shape)

df = df.dropna(axis=0) #DataFrames have two axes (axis=0 corresponds to row amd axis=1 corresponds to columns)

X = np.asanyarray(df[['Rooms','Bathroom','Landsize','Lattitude','Longtitude']])
y = np.asanyarray(df['Price'])

tree = DecisionTreeRegressor(random_state=1)
tree.fit(X,y)

print("Making predictions for the following houses :")
print(df.head())
print("The predictions are :")
print(tree.predict(X)[:5])