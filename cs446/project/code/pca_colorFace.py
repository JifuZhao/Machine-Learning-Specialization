# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from cs446_functions import im2double

#%% load the data
colorTrain = np.load('./data/colorTrain.npz')['colorTrain']
colorTest = np.load('./data/colorTest.npz')['colorTest']

#%% pca analysis
pca = PCA()
pca.fit(colorTrain[1:, :].T)

# calculate the average face
colorMean = pca.mean_
eigenVector = pca.components_.T
ratio = pca.explained_variance_ratio_

#%%
# center the data
rowsTrain, colsTrain = colorTrain.shape
rowsTest, colsTest = colorTest.shape

colorTrainCentered = colorTrain.copy()
colorTestCentered = colorTest.copy()

for i in range(colsTrain):
    colorTrainCentered[1:, i] = colorTrainCentered[1:, i] - colorMean
    
for i in range(colsTest):
    colorTestCentered[1:, i] = colorTestCentered[1:, i] - colorMean

#%% plot the faces

# mean face
plt.figure()
colorMeanFace = np.reshape(colorMean, [360, 260, 3])
colorMeanFace = im2double(colorMeanFace, multiDimension=True)
plt.imshow(colorMeanFace)
plt.title('Mean Faces')
plt.savefig('./figures/color/meanFace.eps', dpi=100)
plt.savefig('./figures/color/meanFace.png', dpi=500)

# original faces
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = np.reshape(colorTrain[1:, i], [360, 260, 3])
    img = im2double(img, multiDimension=True)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('Original Faces', size=16)

plt.savefig('./figures/color/originalFace.eps', dpi=100)
plt.savefig('./figures/color/originalFace.png', dpi=500)


# centered train faces
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = np.reshape(colorTrainCentered[1:, i], [360, 260, 3])
    img = im2double(img, multiDimension=True)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('Centered Faces', size=16)

plt.savefig('./figures/color/trainFace.eps', dpi=100)
plt.savefig('./figures/color/trainFace.png', dpi=500)

# eigen faces
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = np.reshape(eigenVector[:, i], [360, 260, 3])
    img = im2double(img, multiDimension=True)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('Eigen Faces', size=16)

plt.savefig('./figures/color/eigenFace.eps', dpi=100)
plt.savefig('./figures/color/eigenFace.png', dpi=500)

#%% ratio change
ratioSum = np.zeros(300)
for i in range(300):
    ratioSum[i] = np.sum(ratio[:i+1])

fig, ax = plt.subplots(3, 1)
plt.suptitle('Colored Face Explained Variance Ratio', size=16)

ax[0].plot(ratio, 'r-',label='Variance Ratio')
#ax[0].set_ylabel('Variance Ratio')
ax[0].legend()

ax[1].plot(ratioSum, 'g', label='Variance Ratio Summation')
ax[1].set_ylim([0, 1.0])
#ax[1].set_ylabel('Ratio Sum')
ax[1].legend(loc='right')

N = 30
ax[2].bar(range(N), ratio[:N], label='Variance Ratio')
plt.xlabel('Dimensions')
#plt.ylabel('Variance Ratio')
ax[2].legend()

plt.savefig('./figures/color/ratioChange.eps', dpi=100)
plt.savefig('./figures/color/ratioChange.png', dpi=500)

#%% project according to the eigen vector
manSmile = np.zeros([280800, 300]); i = 0;
womanSmile = np.zeros([280800, 300]); j = 0;
manNonSmile = np.zeros([280800, 300]); k = 0;
womanNonSmile = np.zeros([280800, 300]); l = 0;

for n in range(300):
    if int(colorTrainCentered[0, n]) == 1:
        manSmile[:, i] = colorTrainCentered[1:, n]
        i = i + 1
    elif int(colorTrainCentered[0, n]) == 2:
        womanSmile[:, j] = colorTrainCentered[1:, n]
        j = j + 1
    elif int(colorTrainCentered[0, n]) == 3:
        manNonSmile[:, k] = colorTrainCentered[1:, n]
        k = k + 1
    elif int(colorTrainCentered[0, n]) == 4:
        womanNonSmile[:, l] = colorTrainCentered[1:, n]
        l = l + 1
        
manSmile = manSmile[:, :i]
womanSmile = womanSmile[:, :j]
manNonSmile = manNonSmile[:, :k]
womanNonSmile = womanNonSmile[:, :l]

#%% 2 dimension projection and plot
N = 2
chooseEigenVector = eigenVector[:, :N]
trainProjection = np.dot(chooseEigenVector.T, colorTrainCentered[1:, :])

manSmilePorjection = np.dot(chooseEigenVector.T, manSmile)
womanSmilePorjection = np.dot(chooseEigenVector.T, womanSmile)
manNonSmilePorjection = np.dot(chooseEigenVector.T, manNonSmile)
womanNonSmilePorjection = np.dot(chooseEigenVector.T, womanNonSmile)

plt.figure()
plt.plot(manSmilePorjection[0, :], manSmilePorjection[1, :], 'b*', label='man smile')
plt.plot(womanSmilePorjection[0, :], womanSmilePorjection[1, :], 'r*', label='woman smile')
plt.plot(manNonSmilePorjection[0, :], manNonSmilePorjection[1, :], 'bo', label='man non-smile')
plt.plot(womanNonSmilePorjection[0, :], womanNonSmilePorjection[1, :], 'ro', label='woman non-smile')
plt.grid(); plt.xlabel('First Dimension'); plt.ylabel('Second Dimension'); plt.title('Colored Face 2D Projection');
plt.legend(loc=1, prop={'size':7})

plt.savefig('./figures/color/2dProjection.eps', dpi=100)
plt.savefig('./figures/color/2dProjection.png', dpi=500)
    
#%% 3 dimension plot
N = 3
chooseEigenVector = eigenVector[:, :N]
trainProjection = np.dot(chooseEigenVector.T, colorTrainCentered[1:, :])

manSmilePorjection = np.dot(chooseEigenVector.T, manSmile)
womanSmilePorjection = np.dot(chooseEigenVector.T, womanSmile)
manNonSmilePorjection = np.dot(chooseEigenVector.T, manNonSmile)
womanNonSmilePorjection = np.dot(chooseEigenVector.T, womanNonSmile)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(manSmilePorjection[0, :], manSmilePorjection[1, :], manSmilePorjection[1, :], 'b*', label='man smile')
ax.plot(womanSmilePorjection[0, :], womanSmilePorjection[1, :], womanSmilePorjection[1, :],'r*', label='woman smile')
ax.plot(manNonSmilePorjection[0, :], manNonSmilePorjection[1, :], manNonSmilePorjection[1, :], 'bo', label='man non-smile')
ax.plot(womanNonSmilePorjection[0, :], womanNonSmilePorjection[1, :], womanNonSmilePorjection[1, :], 'ro', label='woman non-smile')
ax.set_xlabel('First Dimension'); ax.set_ylabel('Second Dimension'); ax.set_zlabel('Third Dimension'); 
ax.set_title('Colored Face 3D Projection'); plt.legend(loc=1, prop={'size':7})

plt.savefig('./figures/color/3dProjection.eps', dpi=100)
plt.savefig('./figures/color/3dProjection.png', dpi=500)

