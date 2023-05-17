# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:23:03 2023

@author: Omer Faruk Uysal
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data processing
loan_dataset = pd.read_csv("D:/MachineLearningExamples/LoanStatusPrediction/data.csv")
print(loan_dataset.head())
print(loan_dataset.shape)

# statistic measures
print(loan_dataset.describe())

print(loan_dataset.isnull().sum())
# dropping missing values
loan_dataset = loan_dataset.dropna()
# label encoding
loan_dataset.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)
# Dependent column values
print(loan_dataset['Dependents'].value_counts())
# replacing the value of 3+ to 4
loan_dataset.replace({'Dependents':{'3+':4}},inplace=True)
# Data visualization
# education and loan status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)
# marital status and loan status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)
# gender status and loan status
sns.countplot(x='Gender',hue='Loan_Status',data=loan_dataset)
# self employed status and loan status
sns.countplot(x='Self_Employed',hue='Loan_Status',data=loan_dataset)
# convert categorical columns to numberical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
print(loan_dataset.head())

# separating the data and label
x = loan_dataset.drop(columns=['Loan_Status','Loan_ID'],axis=1)
y = loan_dataset['Loan_Status']

# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2) 
print(x.shape,x_train.shape,x_test.shape)

# training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)
# model evaluation
x_train_prediction = classifier.predict(x_train)
acc_on_train = accuracy_score(y_train, x_train_prediction)

x_test_prediction = classifier.predict(x_test)
acc_on_test = accuracy_score(y_test, x_test_prediction)