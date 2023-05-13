# -*- coding: utf-8 -*-
"""
Created on Sat May 13 13:19:24 2023

@author: Omer Faruk Uysal
"""

# Importing the dependencies 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data reading and processing
diabet_data = pd.read_csv("D:/MachineLearningExamples/DiabetesPrediction/diabetes.csv")
diabet_data.head()
diabet_data.shape

# getting the statistical measures of data 
diabet_data.describe()
diabet_data['Outcome'].value_counts()
diabet_data.groupby("Outcome").mean()

# seperating data and labels
x = diabet_data.drop(columns="Outcome" , axis = 1)
y = diabet_data['Outcome']

# Data standardization. Since the value ranges are varying for the columns, it would be harder to fit these values for machine learning models
# So standardization in a particular range helps for better predictions

scaler = StandardScaler()
scaler.fit(x) 
standardized_data = scaler.transform(x)
print(standardized_data)

# Splitting data 
x_train,x_test,y_train,y_test = train_test_split(standardized_data,y,test_size=0.2,stratify=y , random_state=2)
print(x.shape,x_train.shape,x_test.shape)

# training data 
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train) 

# Model evaluation
# accuracy on training data 
x_train_prediction = classifier.predict(x_train)
x_train_accuracy = accuracy_score(y_train, x_train_prediction)

# accuracy on test data 
x_test_prediction = classifier.predict(x_test)
x_test_accuracy = accuracy_score(y_test, x_test_prediction)

# Making a predictive system
input_data = (4,110,92,0,0,37.6,0.191,30)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
input_data = scaler.transform(input_data)
prediction = classifier.predict(input_data)
print(prediction)

