#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import numpy as np
import scipy.io as sio

#%% Self-definied randomShuffle() function
def randomShuffle(data, dimension=1, delta=1):
    '''
    self-defined function for randomly shuffle
    
    the output will be the original data in random sequence
    
    if dimension = 1, it will shuffle 1d list or array
    
    if dimension = 2, it will shuffle 2d array by columns
    
    when delta is 1, return one array
    
    when delta is less than one, return two arrays
    '''
    import numpy as np
    
    data = np.array(data)
    if dimension == 1:
        cols = len(data)
        sequence = np.arange(0, cols, 1)
        np.random.shuffle(sequence)
        shuffled = data[sequence]
    else:
        rows, cols = data.shape
        sequence = np.arange(0, cols, 1)
        np.random.shuffle(sequence)
        shuffled = data[:, sequence]
        
    if delta == 1:
        result = shuffled
    else:
        N = int(cols*delta)
        if dimension == 1:
            first = shuffled[:N]
            second = shuffled[N:]
        else:
            first = shuffled[:, :N]
            second = shuffled[:, N:]
        result = (first, second)
            
    return result
    
#%% load the data
colorSmile = np.load('./colorface/colorSmileArray.npz')['colorSmile'] # 280801
colorNonSmile = np.load('./colorface/colorNonSmileArray.npz')['colorNonSmile']
frontSmile = np.load('./frontface/frontSmileArray.npz')['frontSmile'] # 75001
frontNonSmile = np.load('./frontface/frontNonSmileArray.npz')['frontNonSmile']
littleSmile = np.load('./littleface/littleSmileArray.npz')['littleSmile'] # 31267
littleNonSmile = np.load('./littleface/littleNonSmileArray.npz')['littleNonSmile']

#%% random select 75%
number = np.arange(200)
trainRand, testRand = randomShuffle(number, delta=0.75)

colorTrain = np.concatenate((colorSmile[:, trainRand], colorNonSmile[:, trainRand]), axis=1)
colorTest = np.concatenate((colorSmile[:, testRand], colorNonSmile[:, testRand]), axis=1)
frontTrain = np.concatenate((frontSmile[:, trainRand], frontNonSmile[:, trainRand]), axis=1)
frontTest = np.concatenate((frontSmile[:, testRand], frontNonSmile[:, testRand]), axis=1)
littleTrain = np.concatenate((littleSmile[:, trainRand], littleNonSmile[:, trainRand]), axis=1)
littleTest = np.concatenate((littleSmile[:, testRand], littleNonSmile[:, testRand]), axis=1)

colorTrain = randomShuffle(colorTrain, dimension=2)
colorTest = randomShuffle(colorTest, dimension=2)
frontTrain = randomShuffle(frontTrain, dimension=2)
frontTest = randomShuffle(frontTest, dimension=2)
littleTrain = randomShuffle(littleTrain, dimension=2)
littleTest = randomShuffle(littleTest, dimension=2)

colorTrain = randomShuffle(colorTrain, dimension=2)
colorTest = randomShuffle(colorTest, dimension=2)
frontTrain = randomShuffle(frontTrain, dimension=2)
frontTest = randomShuffle(frontTest, dimension=2)
littleTrain = randomShuffle(littleTrain, dimension=2)
littleTest = randomShuffle(littleTest, dimension=2)

colorTrain = randomShuffle(colorTrain, dimension=2)
colorTest = randomShuffle(colorTest, dimension=2)
frontTrain = randomShuffle(frontTrain, dimension=2)
frontTest = randomShuffle(frontTest, dimension=2)
littleTrain = randomShuffle(littleTrain, dimension=2)
littleTest = randomShuffle(littleTest, dimension=2)


#%% save the data
sio.savemat('./../data/trainIndex', {'trainIndex':trainRand})
sio.savemat('./../data/testIndex', {'testIndex':testRand})
np.savez_compressed('./../data/colorTrain', colorTrain = colorTrain)
np.savez_compressed('./../data/colorTest', colorTest = colorTest)
np.savez_compressed('./../data/frontTrain', frontTrain = frontTrain)
np.savez_compressed('./../data/frontTest', frontTest = frontTest)
np.savez_compressed('./../data/littleTrain', littleTrain = littleTrain)
np.savez_compressed('./../data/littleTest', littleTest = littleTest)

