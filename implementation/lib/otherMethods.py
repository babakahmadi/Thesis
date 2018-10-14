import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn import neighbors

#-------------------------------------------------------------------------------------  LDA ---------------------------------------
from sklearn.lda import LDA
def LDA10Fold(X,y):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf = LDA()
        clf.fit(X[train_index], yTrain)
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
        nclf = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
        # print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)


#-------------------------------------------------------------------------------------  PCA ---------------------------------------
from sklearn.decomposition import PCA
def PCA10Fold(X,y):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf = PCA()
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
        nclf = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)


#-------------------------------------------------------------------------------------  KPCA ---------------------------------------
from sklearn.decomposition import PCA, KernelPCA
def KPCA10Fold(X,y):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
        nclf = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc),np.std(acc)



#-------------------------------------------------------------------------------------  ICA ---------------------------------------
from sklearn.decomposition import FastICA
def ICA10Fold(X,y):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf =  FastICA()
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
        nclf = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)



#-------------------------------------------------------------------------------------  LLE ---------------------------------------
from sklearn.manifold import LocallyLinearEmbedding
def LLE10Fold(X,y):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        n_neighbors = 30
        clf =  LocallyLinearEmbedding(n_neighbors, n_components=2,method='standard')
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
        nclf = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)


#-------------------------------------------------------------------------------------  Isomap ---------------------------------------
from sklearn.manifold import  Isomap
def isomap10Fold(X,y):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        n_neighbors = 30
        clf =  Isomap(n_neighbors, n_components=2)
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
        nclf = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)


# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================
# ================================================================================================================================================================================







from sklearn.lda import LDA
def LDA10FoldClf(X,y,nclf):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf = LDA()
        clf.fit(X[train_index], yTrain)
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
#         NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
        # print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)


#-------------------------------------------------------------------------------------  PCA ---------------------------------------
from sklearn.decomposition import PCA
def PCA10FoldClf(X,y,nclf):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf = PCA()
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
#         NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)


#-------------------------------------------------------------------------------------  KPCA ---------------------------------------
from sklearn.decomposition import PCA, KernelPCA
def KPCA10FoldClf(X,y,nclf):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
#         NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc),np.std(acc)



#-------------------------------------------------------------------------------------  ICA ---------------------------------------
from sklearn.decomposition import FastICA
def ICA10FoldClf(X,y,nclf):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        clf =  FastICA()
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
#         NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)



#-------------------------------------------------------------------------------------  LLE ---------------------------------------
from sklearn.manifold import LocallyLinearEmbedding
def LLE10FoldClf(X,y,nclf):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        n_neighbors = 30
        clf =  LocallyLinearEmbedding(n_neighbors, n_components=2,method='standard')
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
#         NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)


#-------------------------------------------------------------------------------------  Isomap ---------------------------------------
from sklearn.manifold import  Isomap
def isomap10FoldClf(X,y,nclf):
    acc = []
    kf = KFold(X.shape[0],n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        yTest = y[test_index]
        yTrain = y[train_index]
        n_neighbors = 30
        clf =  Isomap(n_neighbors, n_components=2)
        clf.fit(X[train_index])
        newRepTrain = clf.transform(X[train_index])
        newRepTest = clf.transform(X[test_index])
#         NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        nclf.fit(newRepTrain,yTrain)
        XPred = nclf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
#         print i,":",acc[i]
        i += 1
    return np.mean(acc), np.std(acc)
