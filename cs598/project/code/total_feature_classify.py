#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

from self_functions import compute_accuracy
from self_functions import positive_compute
from self_functions import self_randomShuffle

#%% load the data
true_static_1 = sio.loadmat('./features/true_static_label.mat')['good_features1']
true_static_2 = sio.loadmat('./features/true_static_label.mat')['good_features2']
true_moving = sio.loadmat('./features/True_Moving.mat')['moving_good_features']
sheng_true_static = sio.loadmat('./features/sheng_true.mat')['sheng_static_good']
sheng_true_moving = sio.loadmat('./features/sheng_true.mat')['sheng_moving_good']

false_static_1 = sio.loadmat('./features/false_static_label.mat')['bad_features1']
false_static_2 = sio.loadmat('./features/false_static_label.mat')['bad_features2']
false_moving = sio.loadmat('./features/false_moving.mat')['false_moving']
sheng_false_static = sio.loadmat('./features/sheng_false.mat')['sheng_static_false']
sheng_false_moving = sio.loadmat('./features/sheng_false.mat')['sheng_moving_false']

true = np.concatenate((true_static_1, true_static_2, true_moving, sheng_true_static, sheng_true_moving), axis=1)
false = np.concatenate((false_static_1, false_static_2, false_moving, sheng_false_static, sheng_false_moving), axis=1)
true_rows, true_cols = true.shape
false_rows, false_cols = false.shape

data = np.zeros([true_rows+1, true_cols+false_cols])
data[:-1, :true_cols] = true
data[-1, :true_cols] = 1
data[:-1, true_cols:] = false

#%% define array to record the accuracy
iternation = 10 # repeate for 10 times
mean_accuracy_train = np.zeros([3, iternation]) # 3 different classifiers
mean_accuracy_test = np.zeros([3, iternation]) # 3 different classifiers
false_positive_train = np.zeros([3, iternation])
false_positive_test = np.zeros([3, iternation])
false_negative_train = np.zeros([3, iternation])
false_negative_test = np.zeros([3, iternation])

# 75% as training set, 25% as test set
delta = 0.80
filename = './result/total_feature_result/' + str(round(delta*100)) + '_total_random_classify.txt'
f = open(filename, 'w')
    
for i in range(iternation):
    f.write('********* Iternation ' + str(i+1) + '*********' + '\n' + '\n')
    # random shuffule the data 
    f.write('Training set is: ' + str(delta) + '\n' + '\n')
    rows, cols = data.shape
    sequence = np.arange(cols)
    sequence = self_randomShuffle(sequence, dimension=1, delta=1)
    train_sequence, test_sequence = self_randomShuffle(sequence, dimension=1, delta=delta)
    
    train = data[:-1, train_sequence]
    train_label = data[-1, train_sequence]
    test = data[:-1, test_sequence]
    test_label = data[-1, test_sequence]

    # Classification for the fft data
    # Linear Discriminant Analysis for fft data
    clf = LinearDiscriminantAnalysis(solver='svd')
    clf.fit(train.T, train_label)
    train_prediction = clf.predict(train.T)
    test_prediction = clf.predict(test.T)
    train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
    test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
    train_fp, train_fn = positive_compute(train_label, train_prediction)
    test_fp, test_fn = positive_compute(test_label, test_prediction)
    mean_accuracy_train[0, i] = train_accuracy
    mean_accuracy_test[0, i] = test_accuracy
    false_positive_train[0, i] = train_fp
    false_positive_test[0, i] = test_fp
    false_negative_train[0, i] = train_fn
    false_negative_test[0, i] = test_fn
    f.write('Linear Discriminant Analysis, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('Linear Discriminant Analysis, the test accuracy is: ' + str(test_accuracy) + '\n')
    f.write('Linear Discriminant Analysis, the training FP is: ' + str(train_fp) + '\n')
    f.write('Linear Discriminant Analysis, the test FP is: ' + str(test_fp) + '\n')
    f.write('Linear Discriminant Analysis, the training FN is: ' + str(train_fn) + '\n')
    f.write('Linear Discriminant Analysis, the test FN is: ' + str(test_fn) + '\n' + '\n')

    # Support Vector Machine
    # Linear kernel
    clf = svm.SVC(kernel='linear')
    clf.fit(train.T, train_label)
    train_prediction = clf.predict(train.T)
    test_prediction = clf.predict(test.T)
    train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
    test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
    train_fp, train_fn = positive_compute(train_label, train_prediction)
    test_fp, test_fn = positive_compute(test_label, test_prediction)
    mean_accuracy_train[1, i] = train_accuracy
    mean_accuracy_test[1, i] = test_accuracy
    false_positive_train[1, i] = train_fp
    false_positive_test[1, i] = test_fp
    false_negative_train[1, i] = train_fn
    false_negative_test[1, i] = test_fn
    f.write('SVM with linear kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('SVM with linear kernel, the test accuracy is: ' + str(test_accuracy) + '\n')
    f.write('SVM with linear kernel, the training FP is: ' + str(train_fp) + '\n')
    f.write('SVM with linear kernel, the test FP is: ' + str(test_fp) + '\n')
    f.write('SVM with linear kernel, the training FN is: ' + str(train_fn) + '\n')
    f.write('SVM with linear kernel, the test FN is: ' + str(test_fn) + '\n' + '\n')

    # Polynomial kernel
    clf = svm.SVC(kernel='poly')
    clf.fit(train.T, train_label)
    train_prediction = clf.predict(train.T)
    test_prediction = clf.predict(test.T)
    train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
    test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
    mean_accuracy_train[2, i] = train_accuracy
    mean_accuracy_test[2, i] = test_accuracy
    false_positive_train[2, i] = train_fp
    false_positive_test[2, i] = test_fp
    false_negative_train[2, i] = train_fn
    false_negative_test[2, i] = test_fn
    f.write('SVM with Polynomial kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('SVM with Polynomial kernel, the test accuracy is: ' + str(test_accuracy) + '\n')
    f.write('SVM with Polynomial kernel, the training FP is: ' + str(train_fp) + '\n')
    f.write('SVM with Polynomial kernel, the test FP is: ' + str(test_fp) + '\n')
    f.write('SVM with Polynomial kernel, the training FN is: ' + str(train_fn) + '\n')
    f.write('SVM with Polynomial kernel, the test FN is: ' + str(test_fn) + '\n' + '\n')

f.write('************************************' + '\n')
f.write('********* Average Accuracy *********' + '\n')
f.write('************************************' + '\n' + '\n')

f.write('LDA, the training accuracy is: ' + str(np.mean(mean_accuracy_train[0, :])) + '\n')
f.write('LDA, the test accuracy is: ' + str(np.mean(mean_accuracy_test[0, :])) + '\n')
f.write('LDA, the training FP is: ' + str(np.mean(false_positive_train[0, :])) + '\n')
f.write('LDA, the test FP is: ' + str(np.mean(false_positive_test[0, :])) + '\n')
f.write('LDA, the training FN is: ' + str(np.mean(false_negative_train[0, :])) + '\n')
f.write('LDA, the test FN is: ' + str(np.mean(false_negative_test[0, :])) + '\n' + '\n')

f.write('SVM linear kernel, the training accuracy is: ' + str(np.mean(mean_accuracy_train[1, :])) + '\n')
f.write('SVM linear kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[1, :])) + '\n')
f.write('SVM linear kernel, the training FP is: ' + str(np.mean(false_positive_train[1, :])) + '\n')
f.write('SVM linear kernel, the test FP is: ' + str(np.mean(false_positive_test[1, :])) + '\n')
f.write('SVM linear kernel, the training FN is: ' + str(np.mean(false_negative_train[1, :])) + '\n')
f.write('SVM linear kernel, the test FN is: ' + str(np.mean(false_negative_test[1, :])) + '\n' + '\n')

f.write('SVM Polynomial kernell, the training accuracy is: ' + str(np.mean(mean_accuracy_train[2, :])) + '\n')
f.write('SVM Polynomial kernel, the test accuracy is: ' + str(np.mean(mean_accuracy_test[2, :])) + '\n')
f.write('SVM Polynomial kernell, the training FP is: ' + str(np.mean(false_positive_train[2, :])) + '\n')
f.write('SVM Polynomial kernell, the test FP is: ' + str(np.mean(false_positive_test[2, :])) + '\n')
f.write('SVM Polynomial kernell, the training FN is: ' + str(np.mean(false_negative_train[2, :])) + '\n')
f.write('SVM Polynomial kernell, the test FN is: ' + str(np.mean(false_negative_test[2, :])) + '\n' + '\n')
    
f.close()




