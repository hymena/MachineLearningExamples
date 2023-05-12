# -*- coding: utf-8 -*-
"""
Created on Fri May 12 23:46:46 2023

@author: Faruk
"""

# Importing the libraries
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data reading and processing
# Loading the dataset as the pandas dataframe
sonar_data = pd.read_csv("/sonar data.csv", header = None)
# see data information and stats for beter understanding
sonar_data.head()
sonar_data.shape
sonar_data.describe() # statistical measures of data

sonar_data[60].value_counts() # see the number of instances for each label
sonar_data.groupby(60).mean()

# seperating data and labels

x = sonar_data.drop(columns=60, axis = 1)
y = sonar_data[60]

# split the data into training and test data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, stratify = y, random_state=1)

print(f"Shape of x = {x.shape}, x_train = {x_train.shape}, x_test = {x_test.shape}")

# Model training
model = LogisticRegression()
model.fit(x_train, y_train)

# Model evaluation
# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy on training data : " ,training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy on test data : " ,test_data_accuracy)

# Making a predictive system
# put some data
input_data = (0.0094,0.0333,0.0306,0.0376,0.1296,0.1795,0.1909,0.1692,0.1870,0.1725,0.2228,0.3106,0.4144,0.5157,0.5369,0.5107,0.6441,0.7326,0.8164,0.8856,0.9891,1.0000,0.8750,0.8631,0.9074,0.8674,0.7750,0.6600,0.5615,0.4016,0.2331,0.1164,0.1095,0.0431,0.0619,0.1956,0.2120,0.3242,0.4102,0.2939,0.1911,0.1702,0.1010,0.1512,0.1427,0.1097,0.1173,0.0972,0.0703,0.0281,0.0216,0.0153,0.0112,0.0241,0.0164,0.0055,0.0078,0.0055,0.0091,0.0067)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)

prediction = model.predict(input_data)