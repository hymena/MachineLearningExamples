# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:24:33 2023

@author: Omer Faruk Uysal
"""

# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


# data reading and analyzes 
wine_dataset = pd.read_csv("D:\MachineLearningExamples\WineQualityPrediction\data.csv")
print(wine_dataset.shape)
print(wine_dataset.head())
print(wine_dataset.describe())
# number of values for each quality
sns.catplot(x='quality',data=wine_dataset,kind='count')
#volatile acidity vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=wine_dataset)
#citric acid vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=wine_dataset)
#residual sugar vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='residual sugar',data=wine_dataset)
#chlorides vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='chlorides',data=wine_dataset)
#density vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='density',data=wine_dataset)
#free sulfur dioxide vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='free sulfur dioxide',data=wine_dataset)

# correlation 
correlation = wine_dataset.corr()
# constructing heatmap to understand the correlation between columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar = True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Reds')

# data processing 
x = wine_dataset.drop(columns='quality', axis=1)

# binarizing the labels
y = wine_dataset['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
print(y)

# split the data into train and test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=3, stratify=y)
print(x_train.shape,x_test.shape)

# model training 
model = RandomForestClassifier()
model.fit(x_train,y_train)

# model evaluation
# accuracy on test data
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# predictive system
input_data = (6.7,0.58,0.08,1.8,0.09699999999999999,15.0,65.0,0.9959,3.28,0.54,9.2)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
prediction = model.predict(input_data)
if(prediction == 1):
    print("good quality")
else:
    print("bad quality")