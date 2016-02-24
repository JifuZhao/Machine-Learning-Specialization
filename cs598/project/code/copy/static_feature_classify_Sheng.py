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
from self_functions import self_randomShuffle
from self_functions import im2double

#%% load the data
true_1 = sio.loadmat('./features/true_static_label.mat')['good_features1']
true_2 = sio.loadmat('./features/true_static_label.mat')['good_features2']
false_1 = sio.loadmat('./features/false_static_label.mat')['bad_features1']
false_2 = sio.loadmat('./features/false_static_label.mat')['bad_features2']

true = np.concatenate((true_1, true_2), axis=1)
false = np.concatenate((false_1, false_2), axis=1)
true_rows, true_cols = true.shape
false_rows, false_cols = true.shape

train_data = np.zeros([true_rows+1, true_cols+false_cols])
train_data[:-1, :true_cols] = true
train_data[-1, :true_cols] = 1
train_data[:-1, true_cols:] = false

sheng_true = sio.loadmat('./features/sheng_true.mat')['sheng_static_good']
sheng_false = sio.loadmat('./features/sheng_false.mat')['sheng_static_false']

sheng_true_rows, sheng_true_cols = sheng_true.shape
sheng_false_rows, sheng_false_cols = sheng_false.shape

test_data = np.zeros([sheng_true_rows+1, sheng_true_cols+sheng_false_cols])
test_data[:-1, :sheng_true_cols] = sheng_true
test_data[-1, :sheng_true_cols] = 1
test_data[:-1, sheng_true_cols:] = sheng_false

#%% show the data
#img = im2double(data[:-1, :], multiDimension=False)
#plt.imshow(img)
#plt.axis('tight')
#plt.savefig('./result/sheng_feature_result/raw_data.eps', dpi=100)
#plt.savefig('./result/sheng_feature_result/raw_data.png', dpi=500)

#%% define array to record the accuracy
iternation = 1 # repeate for 10 times
mean_accuracy_train = np.zeros([10, iternation]) # 10 different classifiers
mean_accuracy_test = np.zeros([10, iternation]) # 10 different classifiers

f = open('./result/sheng_feature_result/static_random_classify.txt', 'w')
    
for i in range(iternation):
    f.write('********* Iternation ' + str(i+1) + '*********' + '\n' + '\n')
    
    train = train_data[:-1, :]
    train_label = train_data[-1, :]
    test = test_data[:-1, :]
    test_label = test_data[-1, :]

    # Classification for the fft data
    # Linear Discriminant Analysis for fft data
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(train.T, train_label)
    train_prediction = clf.predict(train.T)
    test_prediction = clf.predict(test.T)
    train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
    test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
    mean_accuracy_train[0, i] = train_accuracy
    mean_accuracy_test[0, i] = test_accuracy
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
    mean_accuracy_train[1, i] = train_accuracy
    mean_accuracy_test[1, i] = test_accuracy
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
    mean_accuracy_train[2, i] = train_accuracy
    mean_accuracy_test[2, i] = test_accuracy
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
    mean_accuracy_train[3, i] = train_accuracy
    mean_accuracy_test[3, i] = test_accuracy
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
    mean_accuracy_train[4, i] = train_accuracy
    mean_accuracy_test[4, i] = test_accuracy
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
    mean_accuracy_train[5, i] = train_accuracy
    mean_accuracy_test[5, i] = test_accuracy
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
    mean_accuracy_train[6, i] = train_accuracy
    mean_accuracy_test[6, i] = test_accuracy
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
    mean_accuracy_train[7, i] = train_accuracy
    mean_accuracy_test[7, i] = test_accuracy
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
    mean_accuracy_train[8, i] = train_accuracy
    mean_accuracy_test[8, i] = test_accuracy
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
    mean_accuracy_train[9, i] = train_accuracy
    mean_accuracy_test[9, i] = test_accuracy
    f.write('SVM with precomputed kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('SVM with precomputed kernel, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
    #print('SVM with precomputed kernel, the training accuracy is:', train_accuracy)
    #print('SVM with precomputed kernel, the test accuracy is:', test_accuracy)
    #print('\n')

    ## Radial basis function kernel
    #clf = svm.SVC(kernel='rbf')
    #clf.fit(train_fft.T, train_label)
    #train_prediction = clf.predict(train_fft.T)
    #test_prediction = clf.predict(test_fft.T)
    #train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
    #test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
    #print('SVM with rbf kernel, the training accuracy is:', train_accuracy)
    #print('SVM with rbf kernel, the test accuracy is:', test_accuracy)
    #print('\n')
    #
    ## Sigmoid kernel
    #clf = svm.SVC(kernel='sigmoid')
    #clf.fit(train_fft.T, train_label)
    #train_prediction = clf.predict(train_fft.T)
    #test_prediction = clf.predict(test_fft.T)
    #train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
    #test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
    #print('SVM with sigmoid kernel, the training accuracy is:', train_accuracy)
    #print('SVM with sigmoid kernel, the test accuracy is:', test_accuracy)
    #print('\n')

#f.write('************************************' + '\n')
#f.write('********* Average Accuracy *********' + '\n')
#f.write('************************************' + '\n' + '\n')
#
#f.write('LDA, the training accuracy is: ' + str(np.mean(mean_accuracy_train[0, :])) + '\n')
#f.write('LDA, the test accuracy is: ' + str(np.mean(mean_accuracy_test[0, :])) + '\n' + '\n')
#f.write('QDA, the training accuracy is: ' + str(np.mean(mean_accuracy_train[1, :])) + '\n')
#f.write('QDA, the test accuracy is: ' + str(np.mean(mean_accuracy_test[1, :])) + '\n' + '\n')
#f.write('Decision Tree, the training accuracy is: ' + str(np.mean(mean_accuracy_train[2, :])) + '\n')
#f.write('Decision Tree, the test accuracy is: ' + str(np.mean(mean_accuracy_test[2, :])) + '\n' + '\n')
#f.write('KNN, the training accuracy is: ' + str(np.mean(mean_accuracy_train[3, :])) + '\n')
#f.write('KNN, the test accuracy is: ' + str(np.mean(mean_accuracy_test[3, :])) + '\n' + '\n')
#f.write('SGD hinge loss, the training accuracy is: ' + str(np.mean(mean_accuracy_train[4, :])) + '\n')
#f.write('SGD hinge loss, the test accuracy is: ' + str(np.mean(mean_accuracy_test[4, :])) + '\n' + '\n')
#f.write('SGD log loss, the training accuracy is: ' + str(np.mean(mean_accuracy_train[5, :])) + '\n')
#f.write('SGD log loss, the test accuracy is: ' + str(np.mean(mean_accuracy_test[5, :])) + '\n' + '\n')
#f.write('SGD modified_huber loss, the training accuracy is: ' + str(np.mean(mean_accuracy_train[6, :])) + '\n')
#f.write('SGD modified_huber loss, the test accuracy is: ' + str(np.mean(mean_accuracy_test[6, :])) + '\n' + '\n')
#f.write('SVM linear kernel, the training accuracy is: ' + str(np.mean(mean_accuracy_train[7, :])) + '\n')
#f.write('SVM linear kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[7, :])) + '\n' + '\n')
#f.write('SVM Polynomial kernell, the training accuracy is: ' + str(np.mean(mean_accuracy_train[8, :])) + '\n')
#f.write('SVM Polynomial kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[8, :])) + '\n' + '\n')
#f.write('SVM precomputed kernel, the training accuracy is: ' + str(np.mean(mean_accuracy_train[9, :])) + '\n')
#f.write('SVM precomputed kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[9, :])) + '\n' + '\n')
    
f.close()