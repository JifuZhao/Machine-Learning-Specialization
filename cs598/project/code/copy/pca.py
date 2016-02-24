#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

#%% pca analysis

pca = PCA()
pca.fit(train.T)
mean = pca.mean_
ratio = pca.explained_variance_ratio_
eigen_vector = pca.components_

# center the data
rows, cols = train.shape
train_center = np.zeros([rows, cols])
for i in range(cols):
    train_center[:, i] = train[:, i] - mean

# plot the result
plt.figure()
plt.bar(range(1,5), ratio, width=0.5, align='center', label='Variance Ratio')
plt.grid('on')
plt.title('PCA variance ratio')
plt.xlabel('PCA components')
plt.ylabel('Relative variance ratio')
plt.xlim([0.5, 4.5])
plt.xticks([1, 2, 3, 4], [1, 2, 3, 4])

plt.savefig('./result/sheng_moving_static_result/pca_ratio.png', dpi=500)

#%% pca projection
projection = np.dot(eigen_vector[:, :2].T, train_center)

plt.figure()
plt.plot(projection[0, :true_cols], projection[1, :true_cols], 'ro', label='True')
plt.hold('on')
plt.plot(projection[0, true_cols:], projection[1, true_cols:], 'b*', label='False')
plt.legend()
plt.xlabel('First component')
plt.ylabel('Second component')
plt.title('Dimensionality reduced result')

plt.savefig('./result/sheng_moving_static_result/pca_projection.png', dpi=500)


