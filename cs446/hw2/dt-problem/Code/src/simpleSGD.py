#!/usr/bin/env python

import numpy as np

trainDataFile = ['./../feature/attribute/badges.attribute.train1.arff',
                 './../feature/attribute/badges.attribute.train2.arff',
                 './../feature/attribute/badges.attribute.train3.arff', 
                 './../feature/attribute/badges.attribute.train4.arff', 
                 './../feature/attribute/badges.attribute.train5.arff']
                 
testDataFile = ['./../feature/attribute/badges.attribute.test1.arff', 
                './../feature/attribute/badges.attribute.test2.arff',
                './../feature/attribute/badges.attribute.test3.arff',
                './../feature/attribute/badges.attribute.test4.arff',
                './../feature/attribute/badges.attribute.test5.arff']

def readArff(filePath, N):
    # read the data from the arff file
    # store the feature in feature 
    # and store result in result in 1 or 0
    
    with open(filePath, 'r') as f:
        a = f.readlines()
    
    trainData = a[N:]

    instanceNumber = len(trainData)
    length = len(trainData[0].split(',')) - 1

    feature = np.zeros((instanceNumber, length))
    result = str()
    for i in range(instanceNumber):
        temp = trainData[i].split(',')
        feature[i, :] = np.array(temp[:-1])
        result = result + temp[-1][0] + ','   

    result = result.split(',')[:-1]
    for i in range(len(result)):
        if result[i] == '+':
            result[i] = 1.0
        else:
            result[i] = -1.0
            
    return feature, result

def simpleSGD(trainPath, testPath, N):
    # SGD algorithm

    alpha = 0.005
    threshold = 0.001
    
    featureTrain, resultTrain = readArff(trainPath, N)
    
    trainLength = len(featureTrain[1, :])
    instanceNumber = len(resultTrain)
    
    # give w a initial number
    #w = np.random.rand(trainLength)
    w = np.ones(trainLength)/2
    
    count = 0
    while True:
        j = np.random.randint(0, instanceNumber)
        deltaW = alpha*(resultTrain[j] - np.dot(w, featureTrain[j, :]))*featureTrain[j, :];
        w = w + deltaW      
        tempResult = np.dot(w, featureTrain[j, :])
        error = abs(tempResult - resultTrain[j])
        if error < threshold:
            count = count + 1
            if count > 70:
                break   
           
    ## read the test data         
    featureTest, resultTest = readArff(testPath, N)
    testTempResult = np.dot(w, featureTest.T)
    a = testTempResult
    testLength = len(testTempResult)
    for i in range(testLength):
        if testTempResult[i] > 0.0:
            testTempResult[i] = 1.0
        else:
            testTempResult[i] = -1.0
        
    rightNumber = 0
    for i in range(testLength):
        if testTempResult[i] == resultTest[i]:
            rightNumber = rightNumber + 1
        
    # calculate the result    
    accurateRate = rightNumber/testLength
    
    return rightNumber, accurateRate, testLength

resultPath = './../result/simpleSGD_result.txt'
with open(resultPath, 'w') as f:
    for i in range(5):
        trainPath = trainDataFile[i]
        testPath = testDataFile[i]
        rightNumber, accurateRate, totalNumber = simpleSGD(trainPath, testPath, 265)
        
        f.write('Cross Validation')
        f.write('Correctly classified Instances' + '\t' + str(rightNumber) + '\t' + str(accurateRate) + '\n')
        f.write('Total Number of Instances' + '\t' + str(totalNumber) + '\n')



