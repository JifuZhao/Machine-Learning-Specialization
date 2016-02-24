#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from self_functions import self_fft
from self_functions import self_randomShuffle
from self_functions import cross_shuffle
from self_functions import self_concatenate

#%% load the data
acce_X = sio.loadmat('./data/acce_X')['acce_X']
acce_Y = sio.loadmat('./data/acce_Y')['acce_Y']
acce_Z = sio.loadmat('./data/acce_Z')['acce_Z']
gyro_X = sio.loadmat('./data/gyro_X')['gyro_X']
gyro_Y = sio.loadmat('./data/gyro_Y')['gyro_Y']
gyro_Z = sio.loadmat('./data/gyro_Z')['gyro_Z']

#%% plot the whole data
acce_X_reshape = np.reshape(acce_X[:-1, :], [-1, 1], 'F')
acce_Y_reshape = np.reshape(acce_Y[:-1, :], [-1, 1], 'F')
acce_Z_reshape = np.reshape(acce_Z[:-1, :], [-1, 1], 'F')
gyro_X_reshape = np.reshape(gyro_X[:-1, :], [-1, 1], 'F')
gyro_Y_reshape = np.reshape(gyro_Y[:-1, :], [-1, 1], 'F')
gyro_Z_reshape = np.reshape(gyro_Z[:-1, :], [-1, 1], 'F')

# original signal
begin = 0
end = -1

plt.figure()
plt.suptitle('Original Signal')
plt.subplot(6, 1, 1)
plt.plot(acce_X_reshape[begin:end])
plt.ylabel('Acce_X', size=6); plt.xticks([]); plt.yticks([])
plt.subplot(6, 1, 2)
plt.plot(acce_Y_reshape[begin:end])
plt.ylabel('Acce_Y', size=6); plt.xticks([]); plt.yticks([])
plt.subplot(6, 1, 3)
plt.plot(acce_Z_reshape[begin:end])
plt.ylabel('Acce_Z', size=6); plt.xticks([]); plt.yticks([])
plt.subplot(6, 1, 4)
plt.plot(gyro_X_reshape[begin:end])
plt.ylabel('Gyro_X', size=6); plt.xticks([]); plt.yticks([])
plt.subplot(6, 1, 5)
plt.plot(gyro_Y_reshape[begin:end])
plt.ylabel('Gyro_Y', size=6); plt.xticks([]); plt.yticks([])
plt.subplot(6, 1, 6)
plt.plot(gyro_Z_reshape[begin:end])
plt.ylabel('Gyro_Z', size=6); plt.xticks([]); plt.yticks([])

plt.savefig('./result/original_signal.eps')
plt.savefig('./result/original_signal.png', dpi=500)

# fft analysis
acce_X_fft = self_fft(acce_X[:-1, :], scaled=True)
acce_Y_fft = self_fft(acce_X[:-1, :], scaled=True)
acce_Z_fft = self_fft(acce_X[:-1, :], scaled=True)
gyro_X_fft = self_fft(gyro_X[:-1, :], scaled=True)
gyro_Y_fft = self_fft(gyro_Y[:-1, :], scaled=True)
gyro_Z_fft = self_fft(gyro_Z[:-1, :], scaled=True)

N = 100
plt.figure()
plt.suptitle('Fourier Transfrom Result', size=12)
plt.subplot(3, 2, 1)
plt.imshow(acce_X_fft[:N, :])
plt.axis('image'); plt.title('Acce_X', size=8); plt.xticks([])
plt.subplot(3, 2, 3)
plt.imshow(acce_X_fft[:N, :])
plt.axis('image'); plt.title('Acce_Y', size=8); plt.xticks([])
plt.subplot(3, 2, 5)
plt.imshow(acce_X_fft[:N, :])
plt.axis('image'); plt.title('Acce_Z', size=8); plt.xticks([])
plt.subplot(3, 2, 2)
plt.imshow(gyro_X_fft[:N, :])
plt.axis('image'); plt.title('Gyro_X', size=8); plt.xticks([])
plt.subplot(3, 2, 4)
plt.imshow(gyro_Y_fft[:N, :])
plt.axis('image'); plt.title('Gyro_Y', size=8); plt.xticks([])
plt.subplot(3, 2, 6)
plt.imshow(gyro_Z_fft[:N, :])
plt.axis('image'); plt.title('Gyro_Z', size=8); plt.xticks([])

plt.savefig('./result/fft.eps')
plt.savefig('./result/fft.png', dpi=500)

#%% 5 folder random shuffule 
## separate the data into 5 folders
#
#rows, cols = acce_X.shape
#sequence = np.arange(cols)
#sequence = self_randomShuffle(sequence, dimension=1, delta=1)
#sequence1, sequence2, sequence3, sequence4, sequence5 = cross_shuffle(sequence, dimension=1)
#
#acce_X_1 = acce_X[:, sequence1];
#acce_Y_1 = acce_Y[:, sequence1];
#acce_Z_1 = acce_Z[:, sequence1];
#gyro_X_1 = gyro_X[:, sequence1];
#gyro_Y_1 = gyro_Y[:, sequence1];
#gyro_Z_1 = gyro_Z[:, sequence1];
#
#acce_X_2 = acce_X[:, sequence2];
#acce_Y_2 = acce_Y[:, sequence2];
#acce_Z_2 = acce_Z[:, sequence2];
#gyro_X_2 = gyro_X[:, sequence2];
#gyro_Y_2 = gyro_Y[:, sequence2];
#gyro_Z_2 = gyro_Z[:, sequence2];
#
#acce_X_3 = acce_X[:, sequence3];
#acce_Y_3 = acce_Y[:, sequence3];
#acce_Z_3 = acce_Z[:, sequence3];
#gyro_X_3 = gyro_X[:, sequence3];
#gyro_Y_3 = gyro_Y[:, sequence3];
#gyro_Z_3 = gyro_Z[:, sequence3];
#
#acce_X_4 = acce_X[:, sequence4];
#acce_Y_4 = acce_Y[:, sequence4];
#acce_Z_4 = acce_Z[:, sequence4];
#gyro_X_4 = gyro_X[:, sequence4];
#gyro_Y_4 = gyro_Y[:, sequence4];
#gyro_Z_4 = gyro_Z[:, sequence4];
#
#acce_X_5 = acce_X[:, sequence5];
#acce_Y_5 = acce_Y[:, sequence5];
#acce_Z_5 = acce_Z[:, sequence5];
#gyro_X_5 = gyro_X[:, sequence5];
#gyro_Y_5 = gyro_Y[:, sequence5];
#gyro_Z_5 = gyro_Z[:, sequence5];
#
#acce_X_train1, acce_X_train2, acce_X_train3, acce_X_train4, acce_X_train5, acce_X_test1, acce_X_test2, acce_X_test3, acce_X_test4, acce_X_test5 = self_concatenate(acce_X_1, acce_X_2, acce_X_3, acce_X_4, acce_X_5)
# 
#acce_Y_train1, acce_Y_train2, acce_Y_train3, acce_Y_train4, acce_Y_train5, acce_Y_test1, acce_Y_test2, acce_Y_test3, acce_Y_test4, acce_Y_test5 = self_concatenate(acce_Y_1, acce_Y_2, acce_Y_3, acce_Y_4, acce_Y_5)
# 
#acce_Z_train1, acce_Z_train2, acce_Z_train3, acce_Z_train4, acce_Z_train5, acce_Z_test1, acce_Z_test2, acce_Z_test3, acce_Z_test4, acce_Z_test5 = self_concatenate(acce_Z_1, acce_Z_2, acce_Z_3, acce_Z_4, acce_Z_5)
# 
#gyro_X_train1, gyro_X_train2, gyro_X_train3, gyro_X_train4, gyro_X_train5, gyro_X_test1, gyro_X_test2, gyro_X_test3, gyro_X_test4, gyro_X_test5 = self_concatenate(gyro_X_1, gyro_X_2, gyro_X_3, gyro_X_4, gyro_X_5)
# 
#gyro_Y_train1, gyro_Y_train2, gyro_Y_train3, gyro_Y_train4, gyro_Y_train5, gyro_Y_test1, gyro_Y_test2, gyro_Y_test3, gyro_Y_test4, gyro_Y_test5 = self_concatenate(gyro_Y_1, gyro_Y_2, gyro_Y_3, gyro_Y_4, gyro_Y_5)
# 
#gyro_Z_train1, gyro_Z_train2, gyro_Z_train3, gyro_Z_train4, gyro_Z_train5, gyro_Z_test1, gyro_Z_test2, gyro_Z_test3, gyro_Z_test4, gyro_Z_test5 = self_concatenate(gyro_Z_1, gyro_Z_2, gyro_Z_3, gyro_Z_4, gyro_Z_5)
#          
#sio.savemat('./cross_data/acce_X.mat', 
#            {'train1': acce_X_train1, 'train2': acce_X_train2, 'train3': acce_X_train3, 'train4': acce_X_train4, 'train5': acce_X_train5, 
#             'test1': acce_X_test1, 'test2': acce_X_test2, 'test3': acce_X_test3, 'test4': acce_X_test4, 'test5': acce_X_test5})
#sio.savemat('./cross_data/acce_Y.mat', 
#            {'train1': acce_Y_train1, 'train2': acce_Y_train2, 'train3': acce_Y_train3, 'train4': acce_Y_train4, 'train5': acce_Y_train5, 
#             'test1': acce_Y_test1, 'test2': acce_Y_test2, 'test3': acce_Y_test3, 'test4': acce_Y_test4, 'test5': acce_Y_test5})
#sio.savemat('./cross_data/acce_Z.mat', 
#            {'train1': acce_Z_train1, 'train2': acce_Z_train2, 'train3': acce_Z_train3, 'train4': acce_Z_train4, 'train5': acce_Z_train5, 
#             'test1': acce_Z_test1, 'test2': acce_Z_test2, 'test3': acce_Z_test3, 'test4': acce_Z_test4, 'test5': acce_Z_test5})
#sio.savemat('./cross_data/gyro_X.mat', 
#            {'train1': gyro_X_train1, 'train2': gyro_X_train2, 'train3': gyro_X_train3, 'train4': gyro_X_train4, 'train5': gyro_X_train5, 
#             'test1': gyro_X_test1, 'test2': gyro_X_test2, 'test3': gyro_X_test3, 'test4': gyro_X_test4, 'test5': gyro_X_test5})
#sio.savemat('./cross_data/gyro_Y.mat', 
#            {'train1': gyro_Y_train1, 'train2': gyro_Y_train2, 'train3': gyro_Y_train3, 'train4': gyro_Y_train4, 'train5': gyro_Y_train5, 
#             'test1': gyro_Y_test1, 'test2': gyro_Y_test2, 'test3': gyro_Y_test3, 'test4': gyro_Y_test4, 'test5': gyro_Y_test5})
#sio.savemat('./cross_data/gyro_Z.mat', 
#            {'train1': gyro_Z_train1, 'train2': gyro_Z_train2, 'train3': gyro_Z_train3, 'train4': gyro_Z_train4, 'train5': gyro_Z_train5, 
#             'test1': gyro_Z_test1, 'test2': gyro_Z_test2, 'test3': gyro_Z_test3, 'test4': gyro_Z_test4, 'test5': gyro_Z_test5})
