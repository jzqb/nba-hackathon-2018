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
from sklearn.preprocessing import PolynomialFeatures



training_table = pd.read_csv('C:/Users/vball/Downloads/nba_hackathon/nba1/nba-hackathon-2018/Business Analytics/training_table.csv')
training_table = training_table.iloc[:,1:]
training_table.fillna(0, inplace = True)
y_table = training_table[['label']]
x_table = training_table.drop('label', axis = 1)

x_table['Start Date'] = x_table['Start Date'].astype(int)
x_table['Christmas'] = x_table['Christmas'].astype(int)

x_poly = x_table.iloc[:,150:]
poly_features = PolynomialFeatures()
x_poly = poly_features.fit_transform(x_poly)
x_poly = pd.DataFrame(x_poly)
x_poly_column_names = []
for i in range(len(x_poly.columns)):
    x_poly_column_names.append(str(i)+'_')
x_poly.columns = x_poly_column_names
x_table = pd.concat([x_table.iloc[:,:150],x_poly],axis=1)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



train_x, test_x, train_y, test_y = train_test_split(x_table, y_table, train_size = 0.75)

scaler = StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

#poly_features = PolynomialFeatures()
#train_x = poly_features.fit_transform(train_x)
#test_x = poly_features.transform(test_x)

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

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

xgb_model = XGBRegressor()

params={
    'max_depth': [2,3], #[3,4,5,6,7,8,9], # 5 is good but takes too long in kaggle env
    'subsample': [0.6,.7,.8], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [.6,.75,.9], #[0.5,0.6,0.7,0.8],
    'n_estimators': [1000], #[1000,2000,3000]
    'reg_alpha': [.001,.01,.1] #[0.01, 0.02, 0.03, 0.04]
    }

grid_xgb = GridSearchCV(xgb_model, param_grid = params, cv=7, refit=True)
grid_xgb.fit(train_x,train_y)
print(grid_xgb.best_params_)
print(grid_xgb.best_score_)
predicted_y = grid_xgb.predict(test_x)




pred_y = grid_ridge.predict(test_x)

mape = mean_absolute_percentage_error(pred_y, test_y)



##### Train final model and apply to test data

scaler2 = StandardScaler().fit(x_table)
train_x_final = scaler2.transform(x_table)

from sklearn.linear_model import Ridge
ridge_final = Ridge(alpha=100).fit(train_x_final,y_table)



test_table = pd.read_csv('C:/Users/vball/Downloads/nba_hackathon/nba1/nba-hackathon-2018/Business Analytics/test_table.csv')
x_table_test = test_table.iloc[:,1:]
x_table_test.fillna(0, inplace = True)
x_table_test['Start Date'] = x_table_test['Start Date'].astype(int)
x_table_test['Christmas'] = x_table_test['Christmas'].astype(int)

x_poly_test = x_table_test.iloc[:,150:]
x_poly_test = poly_features.transform(x_poly_test)
x_poly_test = pd.DataFrame(x_poly_test)
x_poly_column_names_test = []
for i in range(len(x_poly_test.columns)):
    x_poly_column_names_test.append(str(i)+'_')
x_poly.columns = x_poly_column_names_test
x_table_test = pd.concat([x_table_test.iloc[:,:150],x_poly_test],axis=1)
x_table_test.fillna(0, inplace = True)
x_table_test = scaler2.transform(x_table_test)

the_prediction = ridge_final.predict(x_table_test)
the_prediction = pd.DataFrame(the_prediction, columns = ['Total_Viewers'])

test_set = pd.read_csv('C:/Users/vball/Downloads/nba_hackathon/nba1/nba-hackathon-2018/Business Analytics/test_set.csv')
test_set = test_set.iloc[:,:5]
test_set = pd.concat([test_set,the_prediction],axis=1)
test_set.to_csv('C:/Users/vball/Downloads/nba_hackathon/nba1/nba-hackathon-2018/Business Analytics/test_set.csv')





