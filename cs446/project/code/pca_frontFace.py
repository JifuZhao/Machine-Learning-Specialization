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
frontTrain = np.load('./data/frontTrain.npz')['frontTrain']
frontTest = np.load('./data/frontTest.npz')['frontTest']

#%% pca analysis
pca = PCA()
pca.fit(frontTrain[1:, :].T)

# calculate the average face
frontMean = pca.mean_
eigenVector = pca.components_.T
ratio = pca.explained_variance_ratio_

#%%
# center the data
rowsTrain, colsTrain = frontTrain.shape
rowsTest, colsTest = frontTest.shape

frontTrainCentered = frontTrain.copy()
frontTestCentered = frontTest.copy()

for i in range(colsTrain):
    frontTrainCentered[1:, i] = frontTrainCentered[1:, i] - frontMean
    
for i in range(colsTest):
    frontTestCentered[1:, i] = frontTestCentered[1:, i] - frontMean

#%% plot the faces

# mean face
plt.figure()
frontMeanFace = np.reshape(frontMean, [300, 250])
frontMeanFace = im2double(frontMeanFace)
plt.imshow(frontMeanFace, cmap='gray')
plt.title('Mean Faces')
plt.savefig('./figures/front/meanFace.eps', dpi=100)
plt.savefig('./figures/front/meanFace.png', dpi=500)

# original faces
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = np.reshape(frontTrain[1:, i], [300, 250])
    img = im2double(img)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle('Original Faces', size=16)

plt.savefig('./figures/front/originalFace.eps', dpi=100)
plt.savefig('./figures/front/originalFace.png', dpi=500)

# centered train faces
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = np.reshape(frontTrainCentered[1:, i], [300, 250])
    img = im2double(img)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle('Centered Faces', size=16)

plt.savefig('./figures/front/trainFace.eps', dpi=100)
plt.savefig('./figures/front/trainFace.png', dpi=500)

# eigen faces
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = np.reshape(eigenVector[:, i], [300, 250])
    img = im2double(img)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle('Eigen Faces', size=16)

plt.savefig('./figures/front/eigenFace.eps', dpi=100)
plt.savefig('./figures/front/eigenFace.png', dpi=500)

#%% ratio change
ratioSum = np.zeros(300)
for i in range(300):
    ratioSum[i] = np.sum(ratio[:i+1])

fig, ax = plt.subplots(3, 1)
plt.suptitle('Explained Variance Ratio', size=16)

ax[0].plot(ratio, 'r-',label='Variance Ratio')
ax[1].set_ylim([0, 1.0])
#ax[0].set_ylabel('Variance Ratio')
ax[0].legend()

ax[1].plot(ratioSum, 'g', label='Variance Ratio Summation')
#ax[1].set_ylabel('Ratio Sum')
ax[1].legend(loc='right')

N = 30
ax[2].bar(range(N), ratio[:N], label='Variance Ratio')
plt.xlabel('Dimensions')
#plt.ylabel('Variance Ratio')
ax[2].legend()

plt.savefig('./figures/front/ratioChange.eps', dpi=100)
plt.savefig('./figures/front/ratioChange.png', dpi=500)

#%% project according to the eigen vector
manSmile = np.zeros([75000, 300]); i = 0;
womanSmile = np.zeros([75000, 300]); j = 0;
manNonSmile = np.zeros([75000, 300]); k = 0;
womanNonSmile = np.zeros([75000, 300]); l = 0;

for n in range(300):
    if int(frontTrainCentered[0, n]) == 1:
        manSmile[:, i] = frontTrainCentered[1:, n]
        i = i + 1
    elif int(frontTrainCentered[0, n]) == 2:
        womanSmile[:, j] = frontTrainCentered[1:, n]
        j = j + 1
    elif int(frontTrainCentered[0, n]) == 3:
        manNonSmile[:, k] = frontTrainCentered[1:, n]
        k = k + 1
    elif int(frontTrainCentered[0, n]) == 4:
        womanNonSmile[:, l] = frontTrainCentered[1:, n]
        l = l + 1
        
manSmile = manSmile[:, :i]
womanSmile = womanSmile[:, :j]
manNonSmile = manNonSmile[:, :k]
womanNonSmile = womanNonSmile[:, :l]

#%% 2 dimension projection and plot
N = 2
chooseEigenVector = eigenVector[:, :N]
trainProjection = np.dot(chooseEigenVector.T, frontTrainCentered[1:, :])

manSmilePorjection = np.dot(chooseEigenVector.T, manSmile)
womanSmilePorjection = np.dot(chooseEigenVector.T, womanSmile)
manNonSmilePorjection = np.dot(chooseEigenVector.T, manNonSmile)
womanNonSmilePorjection = np.dot(chooseEigenVector.T, womanNonSmile)

plt.figure()
plt.plot(manSmilePorjection[0, :], manSmilePorjection[1, :], 'b*', label='man smile')
plt.plot(womanSmilePorjection[0, :], womanSmilePorjection[1, :], 'r*', label='woman smile')
plt.plot(manNonSmilePorjection[0, :], manNonSmilePorjection[1, :], 'bo', label='man non-smile')
plt.plot(womanNonSmilePorjection[0, :], womanNonSmilePorjection[1, :], 'ro', label='woman non-smile')
plt.grid(); plt.xlabel('First Dimension'); plt.ylabel('Second Dimension'); plt.title('2D Projection');
plt.legend(loc=1, prop={'size':7})

plt.savefig('./figures/front/2dProjection.eps', dpi=100)
plt.savefig('./figures/front/2dProjection.png', dpi=500)
    
#%% 3 dimension plot
N = 3
chooseEigenVector = eigenVector[:, :N]
trainProjection = np.dot(chooseEigenVector.T, frontTrainCentered[1:, :])

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
ax.set_title('3D Projection'); plt.legend(loc=1, prop={'size':7})

plt.savefig('./figures/front/3dProjection.eps', dpi=100)
plt.savefig('./figures/front/3dProjection.png', dpi=500)
