#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from self_functions import compute_accuracy

#%% load the data
true_static_1 = sio.loadmat('./features/true_static_label.mat')['good_features1']
true_static_2 = sio.loadmat('./features/true_static_label.mat')['good_features2']
true_moving = sio.loadmat('./features/True_Moving.mat')['moving_good_features']

false_static_1 = sio.loadmat('./features/false_static_label.mat')['bad_features1']
false_static_2 = sio.loadmat('./features/false_static_label.mat')['bad_features2']
false_moving = sio.loadmat('./features/false_moving.mat')['false_moving']

true = np.concatenate((true_static_1, true_static_2, true_moving), axis=1)
false = np.concatenate((false_static_1, false_static_2, false_moving), axis=1)
true_rows, true_cols = true.shape
false_rows, false_cols = false.shape

train_data = np.zeros([true_rows+1, true_cols+false_cols])
train_data[:-1, :true_cols] = true
train_data[-1, :true_cols] = 1
train_data[:-1, true_cols:] = false

sheng_true_static = sio.loadmat('./features/sheng_true.mat')['sheng_static_good']
sheng_true_moving = sio.loadmat('./features/sheng_true.mat')['sheng_moving_good']
sheng_false_static = sio.loadmat('./features/sheng_false.mat')['sheng_static_false']
sheng_false_moving = sio.loadmat('./features/sheng_false.mat')['sheng_moving_false']

sheng_true = np.concatenate((sheng_true_static, sheng_true_moving), axis=1)
sheng_false = np.concatenate((sheng_false_static, sheng_false_moving), axis=1)
sheng_true_rows, sheng_true_cols = sheng_true.shape
sheng_false_rows, sheng_false_cols = sheng_false.shape

test_data = np.zeros([sheng_true_rows+1, sheng_true_cols+sheng_false_cols])
test_data[:-1, :sheng_true_cols] = sheng_true
test_data[-1, :sheng_true_cols] = 1
test_data[:-1, sheng_true_cols:] = sheng_false

train = train_data[:-1, :]
train_label = train_data[-1, :]
test = test_data[:-1, :]
test_label = test_data[-1, :]

#%% define training and test dataset

# need to including 4 numpy arrays
# train: training set, m*n dimensions, m features and n samples
# train_label: corresponding labels for training set. 1 means true, 0 means false
# test: test set, m*n dimensions, m features and n samples
# test_label: corresponding labels for training set. 1 means true, 0 means false

#%% save to the file
f = open('./result/sheng_moving_static_result/moving_classify.txt', 'w')

# Classification for the fft data
# Linear Discriminant Analysis for fft data
clf = LinearDiscriminantAnalysis(solver='svd')
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('Linear Discriminant Analysis, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('Linear Discriminant Analysis, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('Linear Discriminant Analysis, the training accuracy is:', train_accuracy)
#print('Linear Discriminant Analysis, the test accuracy is:', test_accuracy)
#print('\n')
    
# Quadratic Discriminant Analysis for fft data
clf = QuadraticDiscriminantAnalysis()
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('Quadratic Discriminant Analysis, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('Quadratic Discriminant Analysis, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('Quadratic Discriminant Analysis, the training accuracy is:', train_accuracy)
#print('Quadratic Discriminant Analysis, the test accuracy is:', test_accuracy)
#print('\n')

# Decision Tree for fft data
clf = tree.DecisionTreeClassifier()
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('Decision Tree, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('Decision Tree, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('Decision Tree, the training accuracy is:', train_accuracy)
#print('Decision Tree, the test accuracy is:', test_accuracy)
#print('\n')

# K-nearest Neighbors for fft data
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('KNN, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('KNN, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('KNN, the training accuracy is:', train_accuracy)
#print('KNN, the test accuracy is:', test_accuracy)
 #print('\n')

# Stochastic Gradient Descent for fft data
# hinge loss function
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('SGD with hinge loss, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('SGD with hinge loss, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('SGD with hinge loss, the training accuracy is:', train_accuracy)
#print('SGD with hinge loss, the test accuracy is:', test_accuracy)
#print('\n')

# log loss function
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('SGD with log loss, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('SGD with log loss, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('SGD with log loss, the training accuracy is:', train_accuracy)
#print('SGD with log loss, the test accuracy is:', test_accuracy)
#print('\n')

# modified_huber loss function
clf = SGDClassifier(loss="modified_huber", penalty="l2")
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('SGD with modified_huber loss, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('SGD with modified_huber loss, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('SGD with modified_huber loss, the training accuracy is:', train_accuracy)
#print('SGD with modified_huber loss, the test accuracy is:', test_accuracy)
#print('\n')

# Support Vector Machine
# Linear kernel
clf = svm.SVC(kernel='linear')
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction) 
f.write('SVM with linear kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('SVM with linear kernel, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('SVM with linear kernel, the training accuracy is:', train_accuracy)
#print('SVM with linear kernel, the test accuracy is:', test_accuracy)
    #print('\n')

# Polynomial kernel
clf = svm.SVC(kernel='poly')
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('SVM with Polynomial kernell, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('SVM with Polynomial kernel, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('SVM with Polynomial kernel, the training accuracy is:', train_accuracy)
#print('SVM with Polynomial kernel, the test accuracy is:', test_accuracy)
#print('\n')

# Precomputed kernel
clf = svm.SVC(kernel='precomputed')
gram = np.dot(train.T, train)
clf.fit(gram, train_label)
train_prediction = clf.predict(np.dot(train.T, train))
test_prediction = clf.predict(np.dot(test.T, train))
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
f.write('SVM with precomputed kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('SVM with precomputed kernel, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#print('SVM with precomputed kernel, the training accuracy is:', train_accuracy)
#print('SVM with precomputed kernel, the test accuracy is:', test_accuracy)
#print('\n')

f.close()