#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import numpy as np
import scipy.io as sio
from self_functions import read_save

#%% define data path
true_static_path = './raw/True_Static/'
#true_moving_path = './raw/True_Moving/'
false_static_path = './raw/False_Static/'
#false_moving_path = './raw/False_Moving_LongTrace/'

#%% read the true_static data from 74 files
acce_X, acce_Y, acce_Z, gyro_X, gyro_Y, gyro_Z = read_save(true_static_path, 600, 74)
    
sio.savemat('./data/all/true_static_acce_X.mat', {'true_static_acce_X': acce_X})
sio.savemat('./data/all/true_static_acce_Y.mat', {'true_static_acce_Y': acce_Y})
sio.savemat('./data/all/true_static_acce_Z.mat', {'true_static_acce_Z': acce_Z})
sio.savemat('./data/all/true_static_gyro_X.mat', {'true_static_gyro_X': gyro_X})
sio.savemat('./data/all/true_static_gyro_Y.mat', {'true_static_gyro_Y': gyro_Y})
sio.savemat('./data/all/true_static_gyro_Z.mat', {'true_static_gyro_Z': gyro_Z})

# read the false_static data from 69 files
acce_X, acce_Y, acce_Z, gyro_X, gyro_Y, gyro_Z = read_save(false_static_path, 600, 69)
    
sio.savemat('./data/all/false_static_acce_X.mat', {'false_static_acce_X': acce_X})
sio.savemat('./data/all/false_static_acce_Y.mat', {'false_static_acce_Y': acce_Y})
sio.savemat('./data/all/false_static_acce_Z.mat', {'false_static_acce_Z': acce_Z})
sio.savemat('./data/all/false_static_gyro_X.mat', {'false_static_gyro_X': gyro_X})
sio.savemat('./data/all/false_static_gyro_Y.mat', {'false_static_gyro_Y': gyro_Y})
sio.savemat('./data/all/false_static_gyro_Z.mat', {'false_static_gyro_Z': gyro_Z})

# read the Sheng's true_static data from 30 files
sheng_true_static_path = './raw/Sheng/True_Static/'
acce_X, acce_Y, acce_Z, gyro_X, gyro_Y, gyro_Z = read_save(sheng_true_static_path, 600, 30)
    
sio.savemat('./data/all/sheng_true_static_acce_X.mat', {'true_static_acce_X': acce_X})
sio.savemat('./data/all/sheng_true_static_acce_Y.mat', {'true_static_acce_Y': acce_Y})
sio.savemat('./data/all/sheng_true_static_acce_Z.mat', {'true_static_acce_Z': acce_Z})
sio.savemat('./data/all/sheng_true_static_gyro_X.mat', {'true_static_gyro_X': gyro_X})
sio.savemat('./data/all/sheng_true_static_gyro_Y.mat', {'true_static_gyro_Y': gyro_Y})
sio.savemat('./data/all/sheng_true_static_gyro_Z.mat', {'true_static_gyro_Z': gyro_Z})

# read the Sheng's false_static data from 30 files
sheng_false_static_path = './raw/Sheng/False_Static/'
acce_X, acce_Y, acce_Z, gyro_X, gyro_Y, gyro_Z = read_save(sheng_false_static_path, 600, 30)
    
sio.savemat('./data/all/sheng_false_static_acce_X.mat', {'false_static_acce_X': acce_X})
sio.savemat('./data/all/sheng_false_static_acce_Y.mat', {'false_static_acce_Y': acce_Y})
sio.savemat('./data/all/sheng_false_static_acce_Z.mat', {'false_static_acce_Z': acce_Z})
sio.savemat('./data/all/sheng_false_static_gyro_X.mat', {'false_static_gyro_X': gyro_X})
sio.savemat('./data/all/sheng_false_static_gyro_Y.mat', {'false_static_gyro_Y': gyro_Y})
sio.savemat('./data/all/sheng_false_static_gyro_Z.mat', {'false_static_gyro_Z': gyro_Z})

## read the true_moving data from 69 files
#acce_X, acce_Y, acce_Z, gyro_X, gyro_Y, gyro_Z = read_save(true_moving_path, 600, 69)
#    
#sio.savemat('./data/true_moving_acce_X.mat', {'true_moving_acce_X': acce_X})
#sio.savemat('./data/true_moving_acce_Y.mat', {'true_moving_acce_Y': acce_Y})
#sio.savemat('./data/true_moving_acce_Z.mat', {'true_moving_acce_Z': acce_Z})
#sio.savemat('./data/true_moving_gyro_X.mat', {'true_moving_gyro_X': gyro_X})
#sio.savemat('./data/true_moving_gyro_Y.mat', {'true_moving_gyro_Y': gyro_Y})
#sio.savemat('./data/true_moving_gyro_Z.mat', {'true_moving_gyro_Z': gyro_Z})
    
## read the false_moving data from 31 files
#acce_X, acce_Y, acce_Z, gyro_X, gyro_Y, gyro_Z = read_save(false_moving_path, 600, 2)
#    
#sio.savemat('./data/false_moving_acce_X.mat', {'false_moving_acce_X': acce_X})
#sio.savemat('./data/false_moving_acce_Y.mat', {'false_moving_acce_Y': acce_Y})
#sio.savemat('./data/false_moving_acce_Z.mat', {'false_moving_acce_Z': acce_Z})
#sio.savemat('./data/false_moving_gyro_X.mat', {'false_moving_gyro_X': gyro_X})
#sio.savemat('./data/false_moving_gyro_Y.mat', {'false_moving_gyro_Y': gyro_Y})
#sio.savemat('./data/false_moving_gyro_Z.mat', {'false_moving_gyro_Z': gyro_Z})

#%% load the data fro ture_static and false_static
true_static_acce_X = sio.loadmat('./data/all/true_static_acce_X')['true_static_acce_X']
true_static_acce_Y = sio.loadmat('./data/all/true_static_acce_Y')['true_static_acce_Y']
true_static_acce_Z = sio.loadmat('./data/all/true_static_acce_Z')['true_static_acce_Z']
true_static_gyro_X = sio.loadmat('./data/all/true_static_gyro_X')['true_static_gyro_X']
true_static_gyro_Y = sio.loadmat('./data/all/true_static_gyro_Y')['true_static_gyro_Y']
true_static_gyro_Z = sio.loadmat('./data/all/true_static_gyro_Z')['true_static_gyro_Z']

false_static_acce_X = sio.loadmat('./data/all/false_static_acce_X')['false_static_acce_X']
false_static_acce_Y = sio.loadmat('./data/all/false_static_acce_Y')['false_static_acce_Y']
false_static_acce_Z = sio.loadmat('./data/all/false_static_acce_Z')['false_static_acce_Z']
false_static_gyro_X = sio.loadmat('./data/all/false_static_gyro_X')['false_static_gyro_X']
false_static_gyro_Y = sio.loadmat('./data/all/false_static_gyro_Y')['false_static_gyro_Y']
false_static_gyro_Z = sio.loadmat('./data/all/false_static_gyro_Z')['false_static_gyro_Z']

rows_true, cols_true = true_static_acce_X.shape
rows_false, cols_false = false_static_acce_X.shape

rows = max(rows_true, rows_false)
cols = cols_true + cols_false

acce_X = np.zeros([rows, cols])
acce_Y = np.zeros([rows, cols])
acce_Z = np.zeros([rows, cols])
gyro_X = np.zeros([rows, cols])
gyro_Y = np.zeros([rows, cols])
gyro_Z = np.zeros([rows, cols])

acce_X[:rows_true-1, :cols_true] = true_static_acce_X[1:, :]; acce_X[-1, :cols_true] = 1
acce_Y[:rows_true-1, :cols_true] = true_static_acce_Y[1:, :]; acce_Y[-1, :cols_true] = 1 
acce_Z[:rows_true-1, :cols_true] = true_static_acce_Z[1:, :]; acce_Z[-1, :cols_true] = 1 
gyro_X[:rows_true-1, :cols_true] = true_static_gyro_X[1:, :]; gyro_X[-1, :cols_true] = 1 
gyro_Y[:rows_true-1, :cols_true] = true_static_gyro_Y[1:, :]; gyro_Y[-1, :cols_true] = 1 
gyro_Z[:rows_true-1, :cols_true] = true_static_gyro_Z[1:, :]; gyro_Z[-1, :cols_true] = 1 

acce_X[:rows_false-1, cols_true:] = false_static_acce_X[1:, :]
acce_Y[:rows_false-1, cols_true:] = false_static_acce_Y[1:, :]
acce_Z[:rows_false-1, cols_true:] = false_static_acce_Z[1:, :]
gyro_X[:rows_false-1, cols_true:] = false_static_gyro_X[1:, :]
gyro_Y[:rows_false-1, cols_true:] = false_static_gyro_Y[1:, :]
gyro_Z[:rows_false-1, cols_true:] = false_static_gyro_Z[1:, :]

sio.savemat('./data/acce_X.mat', {'acce_X': acce_X})
sio.savemat('./data/acce_Y.mat', {'acce_Y': acce_Y})
sio.savemat('./data/acce_Z.mat', {'acce_Z': acce_Z})
sio.savemat('./data/gyro_X.mat', {'gyro_X': gyro_X})
sio.savemat('./data/gyro_Y.mat', {'gyro_Y': gyro_Y})
sio.savemat('./data/gyro_Z.mat', {'gyro_Z': gyro_Z})

#%% sheng's data
sheng_true_static_acce_X = sio.loadmat('./data/all/sheng_true_static_acce_X')['true_static_acce_X']
sheng_true_static_acce_Y = sio.loadmat('./data/all/sheng_true_static_acce_Y')['true_static_acce_Y']
sheng_true_static_acce_Z = sio.loadmat('./data/all/sheng_true_static_acce_Z')['true_static_acce_Z']
sheng_true_static_gyro_X = sio.loadmat('./data/all/sheng_true_static_gyro_X')['true_static_gyro_X']
sheng_true_static_gyro_Y = sio.loadmat('./data/all/sheng_true_static_gyro_Y')['true_static_gyro_Y']
sheng_true_static_gyro_Z = sio.loadmat('./data/all/sheng_true_static_gyro_Z')['true_static_gyro_Z']

sheng_false_static_acce_X = sio.loadmat('./data/all/sheng_false_static_acce_X')['false_static_acce_X']
sheng_false_static_acce_Y = sio.loadmat('./data/all/sheng_false_static_acce_Y')['false_static_acce_Y']
sheng_false_static_acce_Z = sio.loadmat('./data/all/sheng_false_static_acce_Z')['false_static_acce_Z']
sheng_false_static_gyro_X = sio.loadmat('./data/all/sheng_false_static_gyro_X')['false_static_gyro_X']
sheng_false_static_gyro_Y = sio.loadmat('./data/all/sheng_false_static_gyro_Y')['false_static_gyro_Y']
sheng_false_static_gyro_Z = sio.loadmat('./data/all/sheng_false_static_gyro_Z')['false_static_gyro_Z']

rows_true, cols_true = sheng_true_static_acce_X.shape
rows_false, cols_false = sheng_false_static_acce_X.shape

rows = max(rows_true, rows_false)
cols = cols_true + cols_false

acce_X = np.zeros([rows, cols])
acce_Y = np.zeros([rows, cols])
acce_Z = np.zeros([rows, cols])
gyro_X = np.zeros([rows, cols])
gyro_Y = np.zeros([rows, cols])
gyro_Z = np.zeros([rows, cols])

acce_X[:rows_true-1, :cols_true] = sheng_true_static_acce_X[1:, :]; acce_X[-1, :cols_true] = 1
acce_Y[:rows_true-1, :cols_true] = sheng_true_static_acce_Y[1:, :]; acce_Y[-1, :cols_true] = 1 
acce_Z[:rows_true-1, :cols_true] = sheng_true_static_acce_Z[1:, :]; acce_Z[-1, :cols_true] = 1 
gyro_X[:rows_true-1, :cols_true] = sheng_true_static_gyro_X[1:, :]; gyro_X[-1, :cols_true] = 1 
gyro_Y[:rows_true-1, :cols_true] = sheng_true_static_gyro_Y[1:, :]; gyro_Y[-1, :cols_true] = 1 
gyro_Z[:rows_true-1, :cols_true] = sheng_true_static_gyro_Z[1:, :]; gyro_Z[-1, :cols_true] = 1 

acce_X[:rows_false-1, cols_true:] = sheng_false_static_acce_X[1:, :]
acce_Y[:rows_false-1, cols_true:] = sheng_false_static_acce_Y[1:, :]
acce_Z[:rows_false-1, cols_true:] = sheng_false_static_acce_Z[1:, :]
gyro_X[:rows_false-1, cols_true:] = sheng_false_static_gyro_X[1:, :]
gyro_Y[:rows_false-1, cols_true:] = sheng_false_static_gyro_Y[1:, :]
gyro_Z[:rows_false-1, cols_true:] = sheng_false_static_gyro_Z[1:, :]

sio.savemat('./data/sheng_acce_X.mat', {'acce_X': acce_X})
sio.savemat('./data/sheng_acce_Y.mat', {'acce_Y': acce_Y})
sio.savemat('./data/sheng_acce_Z.mat', {'acce_Z': acce_Z})
sio.savemat('./data/sheng_gyro_X.mat', {'gyro_X': gyro_X})
sio.savemat('./data/sheng_gyro_Y.mat', {'gyro_Y': gyro_Y})
sio.savemat('./data/sheng_gyro_Z.mat', {'gyro_Z': gyro_Z})


