# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 20:50:48 2018

@author: vball
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor



training_table = pd.read_csv('C:/Users/vball/Downloads/nba_hackathon/nba-hackathon-2018/Business Analytics/training_table.csv')
training_table = training_table.iloc[:,1:]
training_table.fillna(0, inplace = True)
y_table = training_table[['label']]
x_table = training_table.drop('label', axis = 1)



def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



train_x, test_x, train_y, test_y = train_test_split(x_table, y_table, train_size = 0.75)

param_ridge = {'alpha': np.logspace(-3, 3, 13)}
'''
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('reg', Ridge()),])
'''
param_ridge = {'alpha': np.logspace(-3, 3, 13)}
grid_ridge = GridSearchCV(Ridge(), param_ridge, cv=10)
grid_ridge.fit(train_x, train_y)
print(grid_ridge.best_params_)
print(grid_ridge.best_score_)


param_lasso  = {'alpha': np.logspace(-3, 0, 5)}
grid_lasso = GridSearchCV(Lasso(max_iter=10000), param_lasso, cv=5)
grid_lasso.fit(train_x,train_y)
print(grid_lasso.best_params_)
print(grid_lasso.best_score_)


grid_gbr = GridSearchCV(GradientBoostingRegressor(), param_grid = {'n_estimators':[30,50,60,80]}, cv=5)
grid_gbr.fit(train_x,train_y)
print(grid_gbr.best_params_)
print(grid_gbr.best_score_)


grid_rf = GridSearchCV(RandomForestRegressor(), param_grid = {'n_estimators':[10,50,100]}, cv=5)
grid_rf.fit(train_x,train_y)
print(grid_rf.best_params_)
print(grid_rf.best_score_)



