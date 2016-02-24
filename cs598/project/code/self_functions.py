#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

#%% data read and save function
def read_save(from_path, num_deleted, num_files):
    '''
    from_path: './raw/True_Static/'
    '''
    
    import numpy as np
    import pandas as pd

    # read the data
    length = 0
    for i in range(1, num_files+1):
        accelerometer_path = from_path + str(i) + '/Accelerometer.csv'
        df = pd.read_csv(accelerometer_path, )
        length = max(length, len(df))
    
    length = length - 2*num_deleted + 1
    acce_X = np.zeros([length + 1, num_files])
    acce_Y = np.zeros([length + 1, num_files])
    acce_Z = np.zeros([length + 1, num_files])
    gyro_X = np.zeros([length + 1, num_files])
    gyro_Y = np.zeros([length + 1, num_files])
    gyro_Z = np.zeros([length + 1, num_files])

    for i in range(1, num_files+1):
        accelerometer_path = from_path + str(i) + '/Accelerometer.csv'
        gyroscope_path = from_path + str(i) + '/Gyroscope.csv'
    
        df_acce = pd.read_csv(accelerometer_path, header=None, names=['time', 'X', 'Y', 'Z'])
        df_gyro = pd.read_csv(gyroscope_path, header=None, names=['time', 'X', 'Y', 'Z'])
    
        df_acce = df_acce[num_deleted: -num_deleted]
        df_gyro = df_gyro[num_deleted: -num_deleted]
        size = len(df_gyro)
 
        acce_X[0, i-1] = size
        acce_Y[0, i-1] = size
        acce_Z[0, i-1] = size
        gyro_X[0, i-1] = size
        gyro_Y[0, i-1] = size
        gyro_Z[0, i-1] = size
   
        acce_X[1: size+1, i-1] = df_acce['X']
        acce_Y[1: size+1, i-1] = df_acce['Y']
        acce_Z[1: size+1, i-1] = df_acce['Z']
        gyro_X[1: size+1, i-1] = df_gyro['X']
        gyro_Y[1: size+1, i-1] = df_gyro['Y']
        gyro_Z[1: size+1, i-1] = df_gyro['Z']
    
    return acce_X, acce_Y, acce_Z, gyro_X, gyro_Y, gyro_Z

#%% Self-definied spectrogram() function
def self_fft(spectrum, scaled=False):
    '''
    self-defined function for spectrum visualization
    through Fast Fourier Transform
    
    spectrum need to be at least 2 * 2 array
    rows means features, columns means samples
    
    the number of rows need to be even integer
    
    return the Fast Fourier Transform of every column
    
    the result have been scaled so that every column will have maximum value of 1
    '''
    import numpy as np
    
    rows, cols = spectrum.shape
    N = int(rows/2 + 1)
    result = np.zeros([N, cols])
    if scaled == True:
        for i in range(cols):
            temp = np.fft.fft(spectrum[:, i])
            result[:, i] = abs(temp[:N])
            result[1:-1, i] = 2 * result[1:-1, i]
            result[:, i] = (result[:, i] - np.amin(result[:, i])) / (np.amax(result[:, i]) - np.amin(result[:, i]))
    else:
        for i in range(cols):
            temp = np.fft.fft(spectrum[:, i])
            result[:, i] = abs(temp[:N])
            result[1:-1, i] = 2 * result[1:-1, i]
        
    return result
    
#%% Self-definied randomShuffle() function
def self_randomShuffle(data, dimension=1, delta=1):
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
    
#%% Self-definied cross_shuffle() function
def cross_shuffle(data, dimension=1):
    '''
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
        
    N = round(cols*0.2)
    if dimension == 1:
        data1 = shuffled[: N]
        data2 = shuffled[N: 2*N]
        data3 = shuffled[2*N: 3*N]
        data4 = shuffled[3*N: 4*N]
        data5 = shuffled[4*N:]
    else:
        data1 = shuffled[:, : N]
        data2 = shuffled[:, N: 2*N]
        data3 = shuffled[:, 2*N: 3*N]
        data4 = shuffled[:, 3*N: 4*N]
        data5 = shuffled[:, 4*N:]
            
    return data1, data2, data3, data4, data5
    
#%% self-definied cross-combine() function
def self_concatenate(data1, data2, data3, data4, data5):
    '''
    '''
    import numpy as np
    
    train1 = np.concatenate((data2, data3, data4, data5), axis=1)
    test1 = data1
    
    train2 = np.concatenate((data1, data3, data4, data5), axis=1)
    test2 = data2
    
    train3 = np.concatenate((data1, data2, data4, data5), axis=1)
    test3 = data3
        
    train4 = np.concatenate((data1, data2, data3, data5), axis=1)
    test4 = data4
    
    train5 = np.concatenate((data1, data2, data3, data4), axis=1)
    test5 = data5
    
    return train1, train2, train3, train4, train5, test1, test2, test3, test4, test5
        
#%% self-definied accuracy computing function
def compute_accuracy(input1, input2):
    '''
    self-definied function to compute the accuracy
    
    input1 and input1 should be a list or one dimension np.array
    
    output is the total number, correct number, accuracy
    '''
    length = len(input1)
    count = 0
    for i in range(length):
        if input1[i] == input2[i]:
            count = count + 1
    
    accuracy = count / length
    
    return length, count, accuracy
    
#%% self-definied true_positive and false_positive computing function
def positive_compute(input1, input2):
    '''
    self-definied function to compute the accuracy
    
    input1 and input1 should be a list or one dimension np.array
    
    output is the total number, correct number, accuracy
    '''
    length = len(input1)
    positive = sum(input1)
    negative = length - positive
    false_positive = 0
    false_negative = 0
    
    for i in range(length):
        if (input1[i] == 0) and (input2[i] == 1):
            false_positive = false_positive + 1
        if (input1[i] == 1) and (input2[i] == 0):
            false_negative = false_negative + 1
    
    FP = float(false_positive / positive)
    FN = float(false_negative / negative)
    
    return FP, FN
    
#%% Self-definied im2double() function
def im2double(img, multiDimension=False):
    '''
    self-defined image process function
    
    it will scale the input image into [0.0, 1.0]
    
    default value of multiDimension will be False,
    if the image is 3D, like RGB image, you need to 
    change multiDimension into True
    '''
    import numpy as np
    
    def rescale(img, rows, cols):
        result = np.zeros([rows, cols])
        Max = np.amax(img)
        Min = np.amin(img)
        result = (img - Min)/(Max - Min)  
        return result
    
    if multiDimension == False:
        rows, cols = img.shape
        result = rescale(img, rows, cols)
    else:
        rows, cols, cells = img.shape
        result = np.zeros([rows, cols, cells])
        for i in range(cells):
            result[:, :, i] = rescale(img[:, :, i], rows, cols)
    
    return result      

            
        
