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

#%% save to the file
f = open('./result/sheng_total_feature_result/sheng_total_feature_classify.txt', 'w')

# Classification for the feature data
# Linear Discriminant Analysis for feature data
clf = LinearDiscriminantAnalysis(solver='svd')
clf.fit(train.T, train_label)
train_prediction = clf.predict(train.T)
test_prediction = clf.predict(test.T)
train_total, train_right, train_accuracy = compute_accuracy(train_label, train_prediction)
test_total, test_right, test_accuracy = compute_accuracy(test_label, test_prediction)
train_fp, train_fn = positive_compute(train_label, train_prediction)
test_fp, test_fn = positive_compute(test_label, test_prediction)
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
train_fp, train_fn = positive_compute(train_label, train_prediction)
test_fp, test_fn = positive_compute(test_label, test_prediction)
f.write('SVM with Polynomial kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
f.write('SVM with Polynomial kernel, the test accuracy is: ' + str(test_accuracy) + '\n')
f.write('SVM with Polynomial kernel, the training FP is: ' + str(train_fp) + '\n')
f.write('SVM with Polynomial kernel, the test FP is: ' + str(test_fp) + '\n')
f.write('SVM with Polynomial kernel, the training FN is: ' + str(train_fn) + '\n')
f.write('SVM with Polynomial kernel, the test FN is: ' + str(test_fn) + '\n' + '\n')

f.close()