#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from self_functions import self_fft
from self_functions import compute_accuracy
from self_functions import self_randomShuffle

#%% load the data
acce_X = sio.loadmat('./data/acce_X')['acce_X']
acce_Y = sio.loadmat('./data/acce_Y')['acce_Y']
acce_Z = sio.loadmat('./data/acce_Z')['acce_Z']
gyro_X = sio.loadmat('./data/gyro_X')['gyro_X']
gyro_Y = sio.loadmat('./data/gyro_Y')['gyro_Y']
gyro_Z = sio.loadmat('./data/gyro_Z')['gyro_Z']

# define array to record the accuracy
iternation = 10 # repeate for 10 times
mean_accuracy_train = np.zeros([10, iternation]) # 10 different classifiers
mean_accuracy_test = np.zeros([10, iternation]) # 10 different classifiers

f = open('./result/fft_result/static_random_classify.txt', 'w')
    
for i in range(iternation):
    f.write('********* Iternation ' + str(i+1) + '*********' + '\n' + '\n')
    # random shuffule the data 
    # 75% as training set, 25% as test set
    delta = 0.75
    f.write('Training set is: ' + str(delta) + '\n' + '\n')
    rows, cols = acce_X.shape
    sequence = np.arange(cols)
    sequence = self_randomShuffle(sequence, dimension=1, delta=1)
    train_sequence, test_sequence = self_randomShuffle(sequence, dimension=1, delta=delta)
    
    acce_X_train = acce_X[:, train_sequence]; acce_X_test = acce_X[:, test_sequence]
    acce_Y_train = acce_Y[:, train_sequence]; acce_Y_test = acce_X[:, test_sequence]
    acce_Z_train = acce_Z[:, train_sequence]; acce_Z_test = acce_X[:, test_sequence]
    gyro_X_train = gyro_X[:, train_sequence]; gyro_X_test = gyro_X[:, test_sequence]
    gyro_Y_train = gyro_Y[:, train_sequence]; gyro_Y_test = gyro_Y[:, test_sequence]
    gyro_Z_train = gyro_Z[:, train_sequence]; gyro_Z_test = gyro_Z[:, test_sequence]
    
    acce_X_train_fft = self_fft(acce_X_train[:-1, :], scaled=False)
    acce_Y_train_fft = self_fft(acce_Y_train[:-1, :], scaled=False)
    acce_Z_train_fft = self_fft(acce_Z_train[:-1, :], scaled=False)
    acce_X_test_fft = self_fft(acce_X_test[:-1, :], scaled=False)
    acce_Y_test_fft = self_fft(acce_Y_test[:-1, :], scaled=False)
    acce_Z_test_fft = self_fft(acce_Z_test[:-1, :], scaled=False)
    gyro_X_train_fft = self_fft(gyro_X_train[:-1, :], scaled=False)
    gyro_Y_train_fft = self_fft(gyro_Y_train[:-1, :], scaled=False)
    gyro_Z_train_fft = self_fft(gyro_Z_train[:-1, :], scaled=False)
    gyro_X_test_fft = self_fft(gyro_X_test[:-1, :], scaled=False)
    gyro_Y_test_fft = self_fft(gyro_Y_test[:-1, :], scaled=False)
    gyro_Z_test_fft = self_fft(gyro_Z_test[:-1, :], scaled=False)
    
    # concatenate the data
    # original data
    train = np.concatenate((acce_X_train[:-1, :], acce_Y_train[:-1, :], acce_Z_train[:-1, :], 
                            gyro_X_train[:-1, :], gyro_Y_train[:-1, :], gyro_Z_train[:-1, :]), axis=0)
    train_label = acce_X_train[-1, :]
    
    test = np.concatenate((acce_X_test[:-1, :], acce_Y_test[:-1, :], acce_Z_test[:-1, :],
                           gyro_X_test[:-1, :], gyro_Y_test[:-1, :], gyro_Z_test[:-1, :]), axis=0)
    test_label = acce_X_test[-1, :]

    # fft data
    train_fft = np.concatenate((acce_X_train_fft, acce_Y_train_fft, acce_Z_train_fft,
                                gyro_X_train_fft, gyro_Y_train_fft, gyro_Z_train_fft), axis=0)
    test_fft = np.concatenate((acce_X_test_fft, acce_Y_test_fft, acce_Z_test_fft,
                               gyro_X_test_fft, gyro_Y_test_fft, gyro_Z_test_fft), axis=0)

    # Classification for the fft data
    # Linear Discriminant Analysis for fft data
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    clf.fit(train_fft.T, train_label)
    train_prediction = clf.predict(train_fft.T)
    test_prediction = clf.predict(test_fft.T)
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
    gram = np.dot(train_fft.T, train_fft)
    clf.fit(gram, train_label)
    train_prediction = clf.predict(np.dot(train_fft.T, train_fft))
    test_prediction = clf.predict(np.dot(test_fft.T, train_fft))
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

f.write('************************************' + '\n')
f.write('********* Average Accuracy *********' + '\n')
f.write('************************************' + '\n' + '\n')

f.write('LDA, the training accuracy is: ' + str(np.mean(mean_accuracy_train[0, :])) + '\n')
f.write('LDA, the test accuracy is: ' + str(np.mean(mean_accuracy_test[0, :])) + '\n' + '\n')
f.write('QDA, the training accuracy is: ' + str(np.mean(mean_accuracy_train[1, :])) + '\n')
f.write('QDA, the test accuracy is: ' + str(np.mean(mean_accuracy_test[1, :])) + '\n' + '\n')
f.write('Decision Tree, the training accuracy is: ' + str(np.mean(mean_accuracy_train[2, :])) + '\n')
f.write('Decision Tree, the test accuracy is: ' + str(np.mean(mean_accuracy_test[2, :])) + '\n' + '\n')
f.write('KNN, the training accuracy is: ' + str(np.mean(mean_accuracy_train[3, :])) + '\n')
f.write('KNN, the test accuracy is: ' + str(np.mean(mean_accuracy_test[3, :])) + '\n' + '\n')
f.write('SGD hinge loss, the training accuracy is: ' + str(np.mean(mean_accuracy_train[4, :])) + '\n')
f.write('SGD hinge loss, the test accuracy is: ' + str(np.mean(mean_accuracy_test[4, :])) + '\n' + '\n')
f.write('SGD log loss, the training accuracy is: ' + str(np.mean(mean_accuracy_train[5, :])) + '\n')
f.write('SGD log loss, the test accuracy is: ' + str(np.mean(mean_accuracy_test[5, :])) + '\n' + '\n')
f.write('SGD modified_huber loss, the training accuracy is: ' + str(np.mean(mean_accuracy_train[6, :])) + '\n')
f.write('SGD modified_huber loss, the test accuracy is: ' + str(np.mean(mean_accuracy_test[6, :])) + '\n' + '\n')
f.write('SVM linear kernel, the training accuracy is: ' + str(np.mean(mean_accuracy_train[7, :])) + '\n')
f.write('SVM linear kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[7, :])) + '\n' + '\n')
f.write('SVM Polynomial kernell, the training accuracy is: ' + str(np.mean(mean_accuracy_train[8, :])) + '\n')
f.write('SVM Polynomial kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[8, :])) + '\n' + '\n')
f.write('SVM precomputed kernel, the training accuracy is: ' + str(np.mean(mean_accuracy_train[9, :])) + '\n')
f.write('SVM precomputed kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[9, :])) + '\n' + '\n')
    
f.close()