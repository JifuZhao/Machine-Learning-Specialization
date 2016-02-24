#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

#%%
#def percentage(X, percent):
#    '''
#        function to choose the desired percentage
#    '''
#    Sum = 0
#    if percent == 1.0:
#        return len(X)
#    else:
#        for i in range(len(X)):
#            Sum = Sum + X[i]
#            if Sum/sum(X) >= percent:
#                return i+1

#%%
def percentage(X, percent):
    '''
        function to choose the desired percentage
    '''
    Sum = 0
    for i in range(len(X)):
        Sum = Sum + X[i]
        if Sum/sum(X) >= percent:
            return i+1
                
#%% Self-definied im2double() function
def im2double(img, multiDimension = False):
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

#%% self-definied accuracy computing function
def face_accuracy(input1, input2):
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
    
    return accuracy
    
#%% Self-definied cross_shuffle() function
def cross_shuffle_cs446(data, dimension=1):
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
def concatenate_cs446(data1, data2, data3, data4, data5, axis=0):
    '''
    '''
    import numpy as np
    
    train1 = np.concatenate((data2, data3, data4, data5), axis=axis)
    test1 = data1
    
    train2 = np.concatenate((data1, data3, data4, data5), axis=axis)
    test2 = data2
    
    train3 = np.concatenate((data1, data2, data4, data5), axis=axis)
    test3 = data3
        
    train4 = np.concatenate((data1, data2, data3, data5), axis=axis)
    test4 = data4
    
    train5 = np.concatenate((data1, data2, data3, data4), axis=axis)
    test5 = data5
    
    return train1, train2, train3, train4, train5, test1, test2, test3, test4, test5    
