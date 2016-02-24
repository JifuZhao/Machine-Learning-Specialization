#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import numpy as np
    
from cs446_functions import cross_shuffle_cs446
from cs446_functions import concatenate_cs446
    
#%% load the data
colorSmile = np.load('./FEIfaces/colorface/colorSmileArray.npz')['colorSmile'] # 280801
colorNonSmile = np.load('./FEIfaces/colorface/colorNonSmileArray.npz')['colorNonSmile']
frontSmile = np.load('./FEIfaces/frontface/frontSmileArray.npz')['frontSmile'] # 75001
frontNonSmile = np.load('./FEIfaces/frontface/frontNonSmileArray.npz')['frontNonSmile']
littleSmile = np.load('./FEIfaces/littleface/littleSmileArray.npz')['littleSmile'] # 31267
littleNonSmile = np.load('./FEIfaces/littleface/littleNonSmileArray.npz')['littleNonSmile']

#%% random select 75%
number = np.arange(200)
n1, n2, n3, n4, n5 = cross_shuffle_cs446(number)
train1, train2, train3, train4, train5, test1, test2, test3, test4, test5 = concatenate_cs446(n1, n2, n3, n4, n5)

#%%
for i in range(3):
    if i == 0:
        name = 'little'
        simle = littleSmile
        non_smile = littleNonSmile
    elif i==1:
        name = 'front'
        simle = frontSmile
        non_smile = frontNonSmile
    else:
        name = 'color'
        simle = colorSmile
        non_smile = colorNonSmile
        
    folder = './cross_data/' + name + '_'
    
    train_1 = np.concatenate((simle[:, train1], non_smile[:, train1]), axis=1)
    train_2 = np.concatenate((simle[:, train2], non_smile[:, train2]), axis=1)
    train_3 = np.concatenate((simle[:, train3], non_smile[:, train3]), axis=1)
    train_4 = np.concatenate((simle[:, train4], non_smile[:, train4]), axis=1)
    train_5 = np.concatenate((simle[:, train5], non_smile[:, train5]), axis=1)
    test_1 = np.concatenate((simle[:, test1], non_smile[:, test1]), axis=1)
    test_2 = np.concatenate((simle[:, test2], non_smile[:, test2]), axis=1)
    test_3 = np.concatenate((simle[:, test3], non_smile[:, test3]), axis=1)
    test_4 = np.concatenate((simle[:, test4], non_smile[:, test4]), axis=1)
    test_5 = np.concatenate((simle[:, test5], non_smile[:, test5]), axis=1)

    # save the data
    np.savez_compressed(folder + 'train_1', train = train_1)
    np.savez_compressed(folder + 'train_2', train = train_2)
    np.savez_compressed(folder + 'train_3', train = train_3)
    np.savez_compressed(folder + 'train_4', train = train_4)
    np.savez_compressed(folder + 'train_5', train = train_5)
    np.savez_compressed(folder + 'test_1', test = test_1)
    np.savez_compressed(folder + 'test_2', test = test_2)
    np.savez_compressed(folder + 'test_3', test = test_3)
    np.savez_compressed(folder + 'test_4', test = test_4)
    np.savez_compressed(folder + 'test_5', test = test_5)
    