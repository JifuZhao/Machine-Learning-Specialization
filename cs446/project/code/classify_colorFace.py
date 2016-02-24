# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from cs446_functions import face_accuracy
from cs446_functions import percentage

#%% Original Space
print('*********** Original Space ***********\n')
f = open('./figures/color_classify/color_original_space.txt', 'w')

DT_train = np.zeros(5); DT_test = np.zeros(5)
LDA_train = np.zeros(5); LDA_test = np.zeros(5)
QDA_train = np.zeros(5); QDA_test = np.zeros(5)
KNN_train = np.zeros(5); KNN_test = np.zeros(5)
SVM_train_rbf = np.zeros(5); SVM_test_rbf = np.zeros(5)
SVM_train_linear = np.zeros(5); SVM_test_linear = np.zeros(5)
SVM_train_poly = np.zeros(5); SVM_test_poly = np.zeros(5)
SVM_train_sig = np.zeros(5); SVM_test_sig = np.zeros(5)

for i in range(5):
    print('*********** Data Set ' + str(i+1) + ' ***********\n')
    f.write('*********** Data Set ' + str(i+1) + ' ***********\n')
    
    train_name = './cross_data/color_train_' + str(i+1) + '.npz'
    test_name = './cross_data/color_test_' + str(i+1) + '.npz'

    # load the data
    train_raw = np.load(train_name)['train']
    test_raw = np.load(test_name)['test']
    train = train_raw[1:, :]
    train_label = train_raw[0, :]
    test = test_raw[1:, :]
    test_label = test_raw[0, :]

    # Classification in original space
    # Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(train.T, train_label)
    prediction_train = clf.predict(train.T)
    prediction_test = clf.predict(test.T)
    train_accuracy = face_accuracy(train_label, prediction_train)
    test_accuracy = face_accuracy(test_label, prediction_test)
    DT_train[i] = train_accuracy
    DT_test[i] = test_accuracy
    f.write('Decision Tree, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('Decision Tree, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#    print('Decision Tree, the training accuracy is:', train_accuracy)
#    print('Decision Tree, the test accuracy is:', test_accuracy)
#    print('\n')

    # Linear Discriminant Analysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(train.T, train_label)
    prediction_train = clf.predict(train.T)
    prediction_test = clf.predict(test.T)
    train_accuracy = face_accuracy(train_label, prediction_train)
    test_accuracy = face_accuracy(test_label, prediction_test)    
    LDA_train[i] = train_accuracy
    LDA_test[i] = test_accuracy
    f.write('LDA, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('LDA, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#    print('LDA, the training accuracy is:', train_accuracy)
#    print('LDA, the test accuracy is:', test_accuracy)
#    print('\n')

    # Quadratic Discriminant Analysis
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(train.T, train_label)
    prediction_train = clf.predict(train.T)
    prediction_test = clf.predict(test.T)
    train_accuracy = face_accuracy(train_label, prediction_train)
    test_accuracy = face_accuracy(test_label, prediction_test)
    QDA_train[i] = train_accuracy
    QDA_test[i] = test_accuracy
    f.write('QDA, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('QDA, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#    print('QDA, the training accuracy is:', train_accuracy)
#    print('QDA, the test accuracy is:', test_accuracy)
#    print('\n')

    # K-nearest Neighbors
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train.T, train_label)
    prediction_train = neigh.predict(train.T)
    prediction_test = neigh.predict(test.T)
    train_accuracy = face_accuracy(train_label, prediction_train)
    test_accuracy = face_accuracy(test_label, prediction_test)
    KNN_train[i] = train_accuracy
    KNN_test[i] = test_accuracy
    f.write('KNN, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('KNN, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#    print('KNN, the training accuracy is:', train_accuracy)
#    print('KNN, the test accuracy is:', test_accuracy)
#    print('\n')

    # Support Vector Machine
    # rbf kernel
    clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
    clf.fit(train.T, train_label)
    prediction_train = clf.predict(train.T)
    prediction_test = clf.predict(test.T)
    train_accuracy = face_accuracy(train_label, prediction_train)
    test_accuracy = face_accuracy(test_label, prediction_test)
    SVM_train_rbf[i] = train_accuracy
    SVM_test_rbf[i] = test_accuracy
    f.write('SVM with rbf kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('SVM with rbf kernel, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#    print('SVM with rbf kernel, the training accuracy is:', train_accuracy)
#    print('SVM with rbf kernel, the test accuracy is:', test_accuracy)
#    print('\n')

    # linear kernel
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    clf.fit(train.T, train_label)
    prediction_train = clf.predict(train.T)
    prediction_test = clf.predict(test.T)
    train_accuracy = face_accuracy(train_label, prediction_train)
    test_accuracy = face_accuracy(test_label, prediction_test)
    SVM_train_linear[i] = train_accuracy
    SVM_test_linear[i] = test_accuracy
    f.write('SVM with linear kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('SVM with linear kernel, the training accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#    print('SVM with linear kernel, the training accuracy is:', train_accuracy)
#    print('SVM with linear kernel, the test accuracy is:', test_accuracy)
#    print('\n')

#    # polynomial kernel
#    clf = svm.SVC(kernel='poly', decision_function_shape='ovo')
#    clf.fit(train.T, train_label)
#    prediction_train = clf.predict(train.T)
#    prediction_test = clf.predict(test.T)
#    train_accuracy = face_accuracy(train_label, prediction_train)
#    test_accuracy = face_accuracy(test_label, prediction_test)
#    SVM_train_poly[i] = train_accuracy
#    SVM_test_poly[i] = test_accuracy
#    f.write('SVM with polynomial kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
#    f.write('SVM with polynomial kernel, the training accuracy is: ' + str(test_accuracy) + '\n' + '\n')
##    print('SVM with polynomial kernel, the training accuracy is:', train_accuracy)
##    print('SVM with polynomial kernel, the test accuracy is:', test_accuracy)
##    print('\n')

    # sigmoid kernel
    clf = svm.SVC(kernel='sigmoid', decision_function_shape='ovo')
    clf.fit(train.T, train_label)
    prediction_train = clf.predict(train.T)
    prediction_test = clf.predict(test.T)
    train_accuracy = face_accuracy(train_label, prediction_train)
    test_accuracy = face_accuracy(test_label, prediction_test)
    SVM_train_sig[i] = train_accuracy
    SVM_test_sig[i] = test_accuracy
    f.write('SVM with sigmoid kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
    f.write('SVM with sigmoid kernel, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
#    print('SVM with sigmoid kernel, the training accuracy is:', train_accuracy)
#    print('SVM with sigmoid kernel, the test accuracy is:', test_accuracy)
#    print('\n')
   
f.write('************************************' + '\n')
f.write('********* Average Accuracy *********' + '\n')
f.write('************************************' + '\n' + '\n')

f.write('DT, the mean training accuracy is: ' + str(np.mean(DT_train)) + '\n')
f.write('DT, the mean test accuracy is: ' + str(np.mean(DT_test)) + '\n' + '\n')
f.write('LDA, the mean training accuracy is: ' + str(np.mean(LDA_train)) + '\n')
f.write('LDA, the mean test accuracy is: ' + str(np.mean(LDA_test)) + '\n' + '\n')
f.write('QDA, the mean training accuracy is: ' + str(np.mean(QDA_train)) + '\n')
f.write('QDA, the mean test accuracy is: ' + str(np.mean(QDA_test)) + '\n' + '\n')
f.write('KNN, the mean training accuracy is: ' + str(np.mean(KNN_train)) + '\n')
f.write('KNN, the mean test accuracy is: ' + str(np.mean(KNN_test)) + '\n' + '\n')
f.write('SVM rbf, the mean training accuracy is: ' + str(np.mean(SVM_train_rbf)) + '\n')
f.write('SVM rbf, the mean test accuracy is: ' + str(np.mean(SVM_test_rbf)) + '\n' + '\n')
f.write('SVM linear, the mean training accuracy is: ' + str(np.mean(SVM_train_linear)) + '\n')
f.write('SVM linear, the mean test accuracy is: ' + str(np.mean(SVM_test_linear)) + '\n' + '\n')
f.write('SVM poly, the mean training accuracy is: ' + str(np.mean(SVM_train_poly)) + '\n')
f.write('SVM poly, the mean test accuracy is: ' + str(np.mean(SVM_test_poly)) + '\n' + '\n')
f.write('SVM sig, the mean training accuracy is: ' + str(np.mean(SVM_train_sig)) + '\n')
f.write('SVM sig, the mean test accuracy is: ' + str(np.mean(SVM_test_sig)) + '\n' + '\n')

f.close()
    
#%% PCA Space
print('*********** PCA Space ***********\n')
f = open('./figures/color_classify/color_pca_space.txt', 'w')

component_ratio = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
length = len(component_ratio)

DT_train = np.zeros([5, length]); DT_test = np.zeros([5, length]);
LDA_train = np.zeros([5, length]); LDA_test = np.zeros([5, length]);
QDA_train = np.zeros([5, length]); QDA_test = np.zeros([5, length]);
KNN_train = np.zeros([5, length]); KNN_test = np.zeros([5, length]);
SVM_train_rbf = np.zeros([5, length]); SVM_test_rbf = np.zeros([5, length]);
SVM_train_linear = np.zeros([5, length]); SVM_test_linear = np.zeros([5, length]);

for i in range(5):
    print('*********** Data Set ' + str(i+1) + ' ***********\n')
    f.write('*********** Data Set ' + str(i+1) + ' ***********\n\n')
    
    train_name = './cross_data/color_train_' + str(i+1) + '.npz'
    test_name = './cross_data/color_test_' + str(i+1) + '.npz'

    # load the data
    train_raw = np.load(train_name)['train']
    test_raw = np.load(test_name)['test']
    train_all = train_raw[1:, :]
    train_label = train_raw[0, :]
    test_all = test_raw[1:, :]
    test_label = test_raw[0, :]
    
    # pca analysis
    pca = PCA()
    pca.fit(train_all.T)
    
    # calculate the average face
    mean = pca.mean_
    eigenVector = pca.components_.T
    ratio = pca.explained_variance_ratio_

    # center the data
    rows_train, cols_train = train_all.shape
    rows_test, cols_test = test_all.shape
    
    train_centered = train_all.copy()
    test_centered = test_all.copy()
    
    for k in range(cols_train):
        train_centered[:, k] = train_centered[:, k] - mean
        
    for k in range(cols_test):
        test_centered[:, k] = test_centered[:, k] - mean
            
    for j in range(length):
        N = percentage(ratio, component_ratio[j])
        components = eigenVector[:, :N]
        f.write('*********** ' + str(N) + ' components, ' + str(component_ratio[j]) + ' ratio' ' ***********\n\n')
        
        if N == 1:
            components = np.mat(components)
            
        train = train_centered
        test = test_centered
        train = np.dot(components.T, train)
        test = np.dot(components.T, test)
    
        # Classification in PCA space
        # Decision Tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(train.T, train_label)
        prediction_train = clf.predict(train.T)
        prediction_test = clf.predict(test.T)
        train_accuracy = face_accuracy(train_label, prediction_train)
        test_accuracy = face_accuracy(test_label, prediction_test)
        DT_train[i, j] = train_accuracy
        DT_test[i, j] = test_accuracy
        f.write('Decision Tree, the training accuracy is: ' + str(train_accuracy) + '\n')
        f.write('Decision Tree, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
        #    print('Decision Tree, the training accuracy is:', train_accuracy)
        #    print('Decision Tree, the test accuracy is:', test_accuracy)
        #    print('\n')

        # Linear Discriminant Analysis
        clf = LinearDiscriminantAnalysis()
        clf.fit(train.T, train_label)
        prediction_train = clf.predict(train.T)
        prediction_test = clf.predict(test.T)
        train_accuracy = face_accuracy(train_label, prediction_train)
        test_accuracy = face_accuracy(test_label, prediction_test)    
        LDA_train[i, j] = train_accuracy
        LDA_test[i, j] = test_accuracy
        f.write('LDA, the training accuracy is: ' + str(train_accuracy) + '\n')
        f.write('LDA, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
        #    print('LDA, the training accuracy is:', train_accuracy)
        #    print('LDA, the test accuracy is:', test_accuracy)
        #    print('\n')
       
        # Quadratic Discriminant Analysis
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(train.T, train_label)
        prediction_train = clf.predict(train.T)
        prediction_test = clf.predict(test.T)
        train_accuracy = face_accuracy(train_label, prediction_train)
        test_accuracy = face_accuracy(test_label, prediction_test)
        QDA_train[i, j] = train_accuracy
        QDA_test[i, j] = test_accuracy
        f.write('QDA, the training accuracy is: ' + str(train_accuracy) + '\n')
        f.write('QDA, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
        #    print('QDA, the training accuracy is:', train_accuracy)
        #    print('QDA, the test accuracy is:', test_accuracy)
        #    print('\n')

        # K-nearest neighbors
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(train.T, train_label)
        prediction_train = neigh.predict(train.T)
        prediction_test = neigh.predict(test.T)
        train_accuracy = face_accuracy(train_label, prediction_train)
        test_accuracy = face_accuracy(test_label, prediction_test)
        KNN_train[i, j] = train_accuracy
        KNN_test[i, j] = test_accuracy
        f.write('KNN, the training accuracy is: ' + str(train_accuracy) + '\n')
        f.write('KNN, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
        #    print('KNN, the training accuracy is:', train_accuracy)
        #    print('KNN, the test accuracy is:', test_accuracy)
        #    print('\n')
        
        # Support Vector Machine
        # rbf kernel
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train.T, train_label)
        prediction_train = clf.predict(train.T)
        prediction_test = clf.predict(test.T)
        train_accuracy = face_accuracy(train_label, prediction_train)
        test_accuracy = face_accuracy(test_label, prediction_test)
        SVM_train_rbf[i, j] = train_accuracy
        SVM_test_rbf[i, j] = test_accuracy
        f.write('SVM with rbf kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
        f.write('SVM with rbf kernel, the test accuracy is: ' + str(test_accuracy) + '\n' + '\n')
        #    print('SVM with rbf kernel, the training accuracy is:', train_accuracy)
        #    print('SVM with rbf kernel, the test accuracy is:', test_accuracy)
        #    print('\n')

        # linear kernel
        clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
        clf.fit(train.T, train_label)
        prediction_train = clf.predict(train.T)
        prediction_test = clf.predict(test.T)
        train_accuracy = face_accuracy(train_label, prediction_train)
        test_accuracy = face_accuracy(test_label, prediction_test)
        SVM_train_linear[i, j] = train_accuracy
        SVM_test_linear[i, j] = test_accuracy
        f.write('SVM with linear kernel, the training accuracy is: ' + str(train_accuracy) + '\n')
        f.write('SVM with linear kernel, the training accuracy is: ' + str(test_accuracy) + '\n' + '\n')
        #    print('SVM with linear kernel, the training accuracy is:', train_accuracy)
        #    print('SVM with linear kernel, the test accuracy is:', test_accuracy)
        #    print('\n')
          
f.close()

#sio.savemat('./figures/color_classify/train_accuracy.mat', {'DT_train': DT_train, 'LDA_train': LDA_train, 
#                                                            'QDA_train': QDA_train, 'KNN_train': KNN_train, 
#                                                            'SVM_train_rbf': SVM_train_rbf, 
#                                                            'SVM_train_linear': SVM_train_linear })
#                                                            
#sio.savemat('./figures/color_classify/test_accuracy.mat', {'DT_test': DT_test, 'LDA_test': LDA_test, 
#                                                            'QDA_test': QDA_test, 'KNN_test': KNN_test, 
#                                                            'SVM_test_rbf': SVM_test_rbf, 
#                                                            'SVM_test_linear': SVM_test_linear })

                                                            
#%%                                                            
plt.figure()
plt.plot(component_ratio, np.mean(DT_train, axis=0), 'o-', label='Training error')
plt.hold('True')
plt.plot(component_ratio, np.mean(DT_test, axis=0), 'o-', label='Test error')
plt.title('Decision Tree'); plt.legend(loc='lower right');
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/DecisionTree.eps', dpi=100)
plt.savefig('./figures/color_classify/DecisionTree.png', dpi=400)

plt.figure()
plt.plot(component_ratio, np.mean(LDA_train, axis=0), 'o-', label='Training error'); plt.hold('True');
plt.plot(component_ratio, np.mean(LDA_test, axis=0), 'o-', label='Test error')
plt.title('LDA'); plt.legend(loc='lower right');
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/LDA.eps', dpi=100)
plt.savefig('./figures/color_classify/LDA.png', dpi=400)

plt.figure()
plt.plot(component_ratio, np.mean(QDA_train, axis=0), 'o-', label='Training error'); plt.hold('True');
plt.plot(component_ratio, np.mean(QDA_test, axis=0), 'o-', label='Test error')
plt.title('QDA'); plt.legend(loc='lower right');
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/QDA.eps', dpi=100)
plt.savefig('./figures/color_classify/QDA.png', dpi=400)

plt.figure()
plt.plot(component_ratio, np.mean(KNN_train, axis=0), 'o-', label='Training error'); plt.hold('True');
plt.plot(component_ratio, np.mean(KNN_test, axis=0), 'o-', label='Test error')
plt.title('KNN'); plt.legend(loc='lower right');
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/KNN.eps', dpi=100)
plt.savefig('./figures/color_classify/KNN.png', dpi=400)

plt.figure()
plt.plot(component_ratio, np.mean(SVM_train_rbf, axis=0), 'o-', label='Training error'); plt.hold('True');
plt.plot(component_ratio, np.mean(SVM_test_rbf, axis=0), 'o-', label='Test error')
plt.title('SVM with rbf kernel'); plt.legend(loc='lower right');
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/SVM_train_rbf.eps', dpi=100)
plt.savefig('./figures/color_classify/SVM_train_rbf.png', dpi=400)

plt.figure()
plt.plot(component_ratio, np.mean(SVM_train_linear, axis=0), 'o-', label='Training error'); plt.hold('True');
plt.plot(component_ratio, np.mean(SVM_test_linear, axis=0), 'o-', label='Test error')
plt.title('SVM with linear kernel'); plt.legend(loc='lower right');
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/SVM_train_linear.eps', dpi=100)
plt.savefig('./figures/color_classify/SVM_train_linear.png', dpi=400)

plt.figure()
plt.plot(component_ratio, np.mean(LDA_train, axis=0), 'o-', label='LDA'); plt.hold('True');
plt.plot(component_ratio, np.mean(KNN_train, axis=0), 'o-', label='KNN'); plt.hold('True');
plt.plot(component_ratio, np.mean(SVM_train_rbf, axis=0), 'o-', label='SVM rbf kernel'); plt.hold('True');
plt.plot(component_ratio, np.mean(SVM_train_linear, axis=0), 'o-', label='SVM liner kernel'); plt.hold('True');
plt.title('Algorithm training accuracy in PCA space'); plt.legend(loc='lower right', prop={'size':8});
plt.xlabel('PCA component ratio')
plt.ylabel('Accuracy')
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/total_train.eps', dpi=100)
plt.savefig('./figures/color_classify/total_train.png', dpi=400)

plt.figure()
plt.plot(component_ratio, np.mean(LDA_test, axis=0), 'o-', label='LDA'); plt.hold('True');
plt.plot(component_ratio, np.mean(KNN_test, axis=0), 'o-', label='KNN'); plt.hold('True');
plt.plot(component_ratio, np.mean(SVM_test_rbf, axis=0), 'o-', label='SVM rbf kernel'); plt.hold('True');
plt.plot(component_ratio, np.mean(SVM_test_linear, axis=0), 'o-', label='SVM liner kernel'); plt.hold('True');
plt.title('Algorithm test accuracy in PCA space'); plt.legend(loc='lower right', prop={'size':8});
plt.xlabel('PCA component ratio')
plt.ylabel('Accuracy')
plt.xlim([0.1, 0.96]); plt.ylim([0, 1.05]); plt.grid('on')
plt.savefig('./figures/color_classify/total_test.eps', dpi=100)
plt.savefig('./figures/color_classify/total_test.png', dpi=400)