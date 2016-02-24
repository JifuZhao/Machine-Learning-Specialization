#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import numpy as np
import matplotlib.pyplot as plt
   
#%% read the colored face from ./colorface/* files
man = np.loadtxt('man.txt')

colorSmile = np.zeros((360*260*3 + 1, 200))
for i in range(200):
    imgName = './colorface/smile/%db.jpg' % (i+1)
    img = plt.imread(imgName)
    img = np.reshape(img, (360*260*3, 1))/255
    colorSmile[1:, i] = img[:, 0]
    
colorSmile[0, :] = 2
for i in man:
    colorSmile[0, int(i-1)] = 1
    
np.savez_compressed('./colorface/colorSmileArray', colorSmile = colorSmile)

colorNonSmile = np.zeros((360*260*3 + 1, 200))
for i in range(200):
    imgName = './colorface/nonsmile/%da.jpg' % (i+1)
    img = plt.imread(imgName)
    img = np.reshape(img, (360*260*3, 1))/255
    colorNonSmile[1:, i] = img[:, 0]

colorNonSmile[0, :] = 4    
for i in man:
    colorNonSmile[0, int(i-1)] = 3

np.savez_compressed('./colorface/colorNonSmileArray', colorNonSmile = colorNonSmile)

#%% read the front face from ./frontface/* files
frontSmile = np.zeros((300*250 + 1, 200))
for i in range(200):
    imgName = './frontface/smile/%db.jpg' % (i+1)
    img = plt.imread(imgName)
    img = np.reshape(img, (300*250, 1))/255
    frontSmile[1:, i] = img[:, 0]

frontSmile[0, :] = 2   
for i in man:
    frontSmile[0, int(i-1)] = 1
    
np.savez_compressed('./frontface/frontSmileArray', frontSmile = frontSmile)

frontNonSmile = np.zeros((300*250 + 1, 200))
for i in range(200):
    imgName = './frontface/nonsmile/%da.jpg' % (i+1)
    img = plt.imread(imgName)
    img = np.reshape(img, (300*250, 1))/255
    frontNonSmile[1:, i] = img[:, 0]
    
frontNonSmile[0, :] = 4
for i in man:
    frontNonSmile[0, int(i-1)] = 3

np.savez_compressed('./frontface/frontNonSmileArray', frontNonSmile = frontNonSmile)


#%% read the little face from ./littleface/* files
littleSmile = np.zeros((193*162 + 1, 200))
for i in range(200):
    imgName = './littleface/smile/%db.jpg' % (i+1)
    img = plt.imread(imgName)
    img = np.reshape(img, (193*162, 1))/255
    littleSmile[1:, i] = img[:, 0]

littleSmile[0, :] = 2
for i in man:
    littleSmile[0, int(i-1)] = 1
    
np.savez_compressed('./littleface/littleSmileArray', littleSmile = littleSmile)

littleNonSmile = np.zeros((193*162 + 1, 200))
for i in range(200):
    imgName = './littleface/nonsmile/%da.jpg' % (i+1)
    img = plt.imread(imgName)
    img = np.reshape(img, (193*162, 1))/255
    littleNonSmile[1:, i] = img[:, 0]
    
littleNonSmile[0, :] = 4
for i in man:
    littleNonSmile[0, int(i-1)] = 3

np.savez_compressed('./littleface/littleNonSmileArray', littleNonSmile = littleNonSmile)







