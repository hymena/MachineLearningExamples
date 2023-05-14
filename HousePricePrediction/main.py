# -*- coding: utf-8 -*-
"""
Created on Mon May 15 00:41:45 2023

@author: Omer
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# importing dataset 

house_price_dataset = sklearn.datasets.load_boston();
print(house_price_dataset);

# Loading the dataset to a pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

# add the target column to dataframe
house_price_dataframe['price'] = house_price_dataset.target
print(house_price_dataframe.head())
print(house_price_dataframe.shape)
# if null values exist, we need to drop 
print(house_price_dataframe.isnull().sum())
# examining the data statistics
print(house_price_dataframe.describe())
# understanding the correlation between various features in the dataset 
# positive correlation : if x increases and y is also increases 
# negative correlation : if x increases but y is decreases 

correlation = house_price_dataframe.corr()

# constracting the heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar= True, square= True, fmt='.1f',annot= True, annot_kws={'size':8},cmap='Blues')

# splitting the data and the target
x = house_price_dataframe.drop(['price'], axis=1)
y = house_price_dataframe['price']

# splitting the data into train and test
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape,x_test.shape)

# Model training. XGBoost Regressor is decision tree base enssemble model
model = XGBRegressor()
# training the model
model.fit(x_train,y_train) 
# evaluation of the model
# prediction on training data 
training_data_predition = model.predict(x_train)
# R squared error 
score_1 = metrics.r2_score(y_train,training_data_predition)
# Mean abs error
score_2 = metrics.mean_absolute_error(y_train, training_data_predition)
# visualizing the actual prices and predictions
plt.scatter(y_train, training_data_predition)
plt.xlabel("Actual preices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs predicted prices")
# prediction on testing data
test_data_predition = model.predict(x_test)
score_3 = metrics.r2_score(y_test,test_data_predition)
score_4 = metrics.mean_absolute_error(y_test,test_data_predition)
print("r squared error: ",score_3)
print("mean squared error: ",score_4)
plt.scatter(y_test, test_data_predition)
plt.xlabel("Actual preices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs predicted prices")
