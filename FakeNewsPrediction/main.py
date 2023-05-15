# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:33:17 2023

@author: Omer Faruk Uysal
"""

# Importing the dependencies
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# downloading the stopwords. These are the words does not add much value to the dataset
nltk.download('stopwords')
print(stopwords.words('english'))
#testing something


# data preprocessing 
# loading the dataset 
news_dataset = pd.read_csv('D:/MachineLearningExamples/FakeNewsPrediction/train.csv')
print(news_dataset.shape)
# countng the number of missing values in the dataset
print(news_dataset.isnull().sum())
# replacing the null values with empty string
news_dataset = news_dataset.fillna('')
# merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
print(news_dataset['content'])


# stemming: the process of reducing a word to its root word
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # removing all the non alphabet characters and replacing them with space
    stemmed_content = stemmed_content.lower() # converting them to lowercase
    stemmed_content = stemmed_content.split() # splitting the words 
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # applying stem funtion to each non-stopword
    stemmed_content = ' '.join(stemmed_content) # combining them 
    
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

# separating the data and label
x = news_dataset['content'].values
y = news_dataset['label'].values

# converting textual data to numerical data
vectorizer = TfidfVectorizer() # vectorizing is based on the frequencies of the words and their affects to the output
vectorizer.fit(x)

x = vectorizer.transform(x)
print(x)

# splitting the dataset to training and test

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

# training the model
model = LogisticRegression()
model.fit(x_train, y_train)

# evaluating the model
# accuracy score on the training data
x_train_prediction = model.predict(x_train)
training_score = accuracy_score(x_train_prediction, y_train)
# accuracy score on the test data
x_test_predictions = model.predict(x_test)
test_score = accuracy_score(x_test_predictions, y_test)
print(test_score)