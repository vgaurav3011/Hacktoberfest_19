#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading Data

import sklearn.datasets

import numpy as np

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target
print(X)
print(Y)

type(X)

print(X.shape, Y.shape)

import pandas as pd

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class'] = breast_cancer.target

data.head()

type(X)

print(data['class'].value_counts())

print(breast_cancer.target_names)

data.groupby('class').mean()

type(X)

#Train test Split Data

type(X)

from sklearn.model_selection import train_test_split

X = data.drop('class', axis = 1)
Y = data['class']

type(X)

typ

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)

type(X_train)

print(X.shape, X_train.shape, X_test.shape
     )

print(Y.shape, Y_train.shape, Y_test.shape)
     

print(Y.mean(), Y_train.mean(), Y_test.mean())

print(X.mean(), X_train.mean(), X_test.mean())

type(X_train)

import matplotlib.pyplot as plt

plt.plot(X_train.T, '*')
plt.xticks(rotation = 'vertical')
plt.show()

X_binarised_3_train = X_train['mean area'].map(lambda x: 0 if x<1000 else 1)

plt.plot(X_binarised_3_train, '*')

X_binarised_train = X_train.apply(pd.cut, bins = 2, labels = [1, 0])

plt.plot(X_binarised_train.T, '*')
plt.xticks(rotation = 'vertical')
plt.show()

X_binarised_test = X_test.apply(pd.cut, bins = 2, labels = [1,0])

plt.plot(X_binarised_test.T, '*')
plt.xticks(rotation = 'vertical')
plt.show()

type(X_binarised_test)

X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values

type(Y_train)

type(X_binarised_test)

#NP Neuron Model

b = 3
i = 100
if (np.sum(X_binarised_train[100,:]) > b):
  print('MP Neuron inference is malignent')
else:
  print('MP Neuron inference is benign')
if (Y_train[i] == 1):
  print('Ground truth is Malignent')
else:
  print('Ground truth is benign')
  
 


from random import randint
b = 3
i = randint(0, X_binarised_train.shape[0])
print('For row', i)
if (np.sum(X_binarised_train[100,:]) > b):
  print('MP Neuron inference is malignent')
else:
  print('MP Neuron inference is benign')
if (Y_train[i] == 1):
  print('Ground truth is Malignent')
else:
  print('Ground truth is benign')
  
 



b = 4
Y_pred_train = []
accurate_rows = 0
for x, y in zip(X_binarised_train, Y_train):
  Y_pred = (np.sum(x) >= b)
  Y_pred_train.append(Y_train)
  accurate_rows += (y == Y_pred)
print(accurate_rows, accurate_rows/X_binarised_train.shape[0])
  

type(Y_pred)

for b in range(X_binarised_train.shape[1] + 1):
  Y_pred_train = []
  accurate_rows = 0
  for x, y in zip(X_binarised_train, Y_train):
    Y_pred = (np.sum(x) >= b)
    Y_pred_train.append(Y_train)
    accurate_rows += (y == Y_pred)
  print(b, accurate_rows/X_binarised_train.shape[0])


#Test Data

from sklearn.metrics import accuracy_score
b = 28
Y_pred_test = []
for x in X_binarised_test:
  Y_pred = (np.sum(x) >= b)
  Y_pred_test.append(Y_pred)
accuracy = accuracy_score(Y_pred_test, Y_test)
print(b, accuracy)




#MP Neuron Class

class MPNeuron:
  def __init__(self):
    self.b = None
  def model(self, x):
    return(sum(x) >= self.b)
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
      return np.array(Y)
  def fit(self, X, Y):
    accuracy = {}
    for b in range(X.shape[1], +1):
      self.b = b
      Y_pred = self.predict(X)
      accuracy [b] = accuracy_score(Y_pred, Y)
    best_b = max(accuracy, key = accuracy.get)
    self.b = best_b
    print('Optimal value of b is', best_b)
    print('Highest accuracy is', accuracy[best_b])
    

mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train, Y_train)

#Perceptron Class

class Perceptron:
  def __init__(self):
    self.w = None
    self.b = None
  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  def fit(self, X, Y):
    self.w = np.ones(X.shape[1])
    self.b = 0
    for x, y in zip(X, Y):
      y_pred = self.model(x)
      if y == 1 and y_pred == 0:
        self.w = self.w + x
        self.b = self.b + 1
      elif y == 0 and y_pred == 1:
        self.w = self.w - x
        self.b = self.b - 1
      

perceptron = Perceptron()

X_train = X_train.values
X_test = X_test.values

perceptron.fit(X_train, Y_train)

plt.plot(perceptron.w)
plt.show()

Y_pred_train = perceptron.predict(X_train)
print(accuracy_score(Y_pred_train, Y_train))

Y_pred_test = perceptron.predict(X_test)
print(accuracy_score(Y_pred_test, Y_test))

