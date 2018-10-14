import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn import neighbors
from scipy import sparse as sps
import time


def Prototyping(X,numP):
    from sklearn.cluster import KMeans as km
    kmeans = km(init='k-means++',n_clusters=numP)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    return centers

def distance(p1,p2):
    ans = 0
    for i in xrange(p1.shape[0]):
        ans += (p1[i] - p2[i]) ** 2
    return ans

def similarity(p1,p2,sigma):
    return np.exp((-distance(p1,p2))/(2*(sigma**2)))



#------------------------------------------------------------------------------------------  Supervised --------------------------

def supervised(X,y,centers,sigma):
    numS = X.shape[0]
    numP = centers.shape[0]
    numL = len(np.unique(sorted(y)))
    print numS,numP,numL
    N = numS+numP+numL
    # W = np.zeros((N,N))
    W = sps.lil_matrix((N,N))
    for i in xrange(numS):
        for j in xrange(numP):
            W[i,numS+j] = similarity(X[i],centers[j],sigma)
            W[numS+j,i] = W[i,numS+j]

    P1 = np.sum(y)
    P2 = y.shape[0] - P1
    for i in xrange(numS):
        W[i,numS+numP+y[i]] = 1.
        W[numS+numP+y[i],i] = 1.


    d = np.array(W.sum(axis=0))
    D = np.diag(d)
    W = csc_matrix(W)
    L = D - W
    return W,D,L



def supervisedGraph10Fold(X,y,centers,sigma,train_index,test_index):
    numS = X.shape[0]
    numP = centers.shape[0]
    numL = len(np.unique(sorted(y)))
    N = numS+numP+numL
    # W = np.zeros((N,N))
    W = sps.lil_matrix((N,N))
    for i in xrange(numS):
        for j in xrange(numP):
            W[i,numS+j] = similarity(X[i],centers[j],sigma)
            W[numS+j,i] = W[i,numS+j]
    P = []
    for i in xrange(numL):
        P.append(sum(y==i))

    for i in train_index:
        W[i,numS+numP+y[i]] = 1./P[y[i]]
        W[numS+numP+y[i],i] = 1./P[y[i]]

    d = np.array(W.sum(axis=0))
    D = np.diag(d)
    W = csc_matrix(W)
    L = D - W
    return W,D,L


def supervised10Fold(X,y,numP,sigma):
    numS = X.shape[0]
    t1 = time.clock()
    centers = Prototyping(X,numP)
    t2 = time.clock()
    print "Supervised Prototype: ", t2-t1
    acc = []
    kf = KFold(numS,n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        W,D,L = supervisedGraph10Fold(X,y,centers,sigma,train_index,test_index)
        vals, vecs = sps.linalg.eigsh(L, M=D, k=7)
        vals = vals.real
        vecs = vecs.real[:numS]
        yTrain = y[train_index]
        yTest = y[test_index]
        newRepTrain = vecs[:,1:7][train_index]
        newRepTest = vecs[:,1:7][test_index]
        NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        NN.fit(newRepTrain,yTrain)
        XPred = NN.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
        i += 1
    return np.mean(acc), np.std(acc)


def supervised10FoldClf(X,y,numP,sigma,clf):
    numS = X.shape[0]
    centers = Prototyping(X,numP)
    acc = []
    kf = KFold(numS,n_folds=10,shuffle=True)
    i = 0
    for train_index,test_index in kf:
        W,D,L = supervisedGraph10Fold(X,y,centers,sigma,train_index,test_index)
        vals, vecs = sps.linalg.eigsh(L, M=D, k=7)
        vals = vals.real
        vecs = vecs.real[:numS]
        yTrain = y[train_index]
        yTest = y[test_index]
        newRepTrain = vecs[:,1:7][train_index]
        newRepTest = vecs[:,1:7][test_index]
        clf.fit(newRepTrain,yTrain)
        XPred = clf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
        i += 1
    return np.mean(acc), np.std(acc)

#------------------------------------------------------------------------------------------  Unsupervised --------------------------
def UnsupervisedGraph(X,centers,sigma):
    numS = X.shape[0]
    numP = centers.shape[0]
    print numS,numP
    N = numS + numP
    # W = np.zeros((N,N))
    W = csc_matrix((N,N))
    for i in xrange(numS):
        for j in xrange(numP):
            W[i,numS+j] = similarity(X[i],centers[j],sigma)
            W[numS+j,i] = W[i,numS+j]
    d = np.array(W.sum(axis=0))
    D = np.diag(d)
    W = csc_matrix(W)
    L = D - W
    return W,D,L


def unSupervisedGraph10Fold(X,y,centers,sigma):
    numS = X.shape[0]
    numP = centers.shape[0]
    N = numS+numP
    # W = np.zeros((N,N))
    W = csc_matrix((N,N))
    for i in xrange(numS):
        for j in xrange(numP):
            W[i,numS+j] = similarity(X[i],centers[j],sigma)
            W[numS+j,i] = W[i,numS+j]

    d = np.array(W.sum(axis=0))
    D = np.diag(d)
    W = csc_matrix(W)
    L = D - W
    return W,D,L


def unSupervised10Fold(X,y,numP,sigma):
    numS = X.shape[0]
    t1 = time.clock()
    centers = Prototyping(X,numP)
    t2 = time.clock()
    print 'unsupervised Time:', t2-t1
    acc = []
    kf = KFold(numS,n_folds=10,shuffle=True)
    i = 0
    W,D,L = unSupervisedGraph10Fold(X,y,centers,sigma)
    vals, vecs = sps.linalg.eigsh(L, M=D, k=7)
    vals = vals.real
    vecs = vecs.real[:numS]

    for train_index,test_index in kf:
        yTrain = y[train_index]
        yTest = y[test_index]
        newRepTrain = vecs[:,1:7][train_index]
        newRepTest = vecs[:,1:7][test_index]
        NN = neighbors.KNeighborsClassifier(n_neighbors=2)
        NN.fit(newRepTrain,yTrain)
        XPred = NN.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
        i += 1
    return np.mean(acc), np.std(acc)


def unSupervised10FoldClf(X,y,numP,sigma,clf):
    numS = X.shape[0]
    centers = Prototyping(X,numP)
    acc = []
    kf = KFold(numS,n_folds=10,shuffle=True)
    i = 0
    W,D,L = unSupervisedGraph10Fold(X,y,centers,sigma)
    vals, vecs = sps.linalg.eigsh(L, M=D, k=7)
    vals = vals.real
    vecs = vecs.real[:numS]
    for train_index,test_index in kf:
        yTrain = y[train_index]
        yTest = y[test_index]
        newRepTrain = vecs[:,1:7][train_index]
        newRepTest = vecs[:,1:7][test_index]
        clf.fit(newRepTrain,yTrain)
        XPred = clf.predict(newRepTest)
        acc.append(np.sum(XPred==yTest)*1.0/yTest.shape[0])
        i += 1
    return np.mean(acc), np.std(acc)
