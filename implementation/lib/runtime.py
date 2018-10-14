from time import clock
from sklearn.decomposition import FastICA
def myICA(X,y):
    t1 = clock()
    clf =  FastICA()
    clf.fit(X)
    newRep = clf.transform(X)
    t2 = clock()
    return t2-t1

from sklearn.decomposition import PCA, KernelPCA
def myKPCA(X,y):
    t1 = clock()
    clf = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1)
    clf.fit(X)
    newRep = clf.transform(X)
    t2 = clock()
    return t2-t1


from sklearn.decomposition import PCA
def myPCA(X,y):
    t1 = clock()
    clf = PCA()
    clf.fit(X)
    newRep = clf.transform(X)
    t2 = clock()
    return t2-t1

from sklearn.lda import LDA
def myLDA(X,y):
    t1 = clock()
    clf = LDA()
    clf.fit(X, y)
    newRep = clf.transform(X)
    t2 = clock()
    return t2-t1

from sklearn import manifold
def myLLE(X,y):
    t1 = clock()
    n_neighbors = 30
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
    clf.fit(X,y)
    newRep = clf.transform(X)
    t2 = clock()
    return t2-t1

def myMDS(X,y):
    t1 = clock()
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)

    newRep = clf.fit_transform(X)
    t2 = clock()
    return t2-t1

def myIsomap(X,y):
    t1 = clock()
    n_neighbors = 30
    clf = manifold.Isomap(n_neighbors, n_components=2)
    clf.fit(X)
    newRep = clf.transform(X)
    t2 = clock()
    return t2-t1

def myTSNE(X,y):
    t1 = clock()
    clf = manifold.t_sne(n_components=4, init='pca', random_state=0)
    newRep = clf.fit_transform(X)
    t2 = clock()
    return t2-t1



#---------------------------------------------  PROPOSED
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn import neighbors
from scipy import sparse as sps

def Prototyping(X,numP):
    from sklearn.cluster import KMeans as km
    kmeans = km(init='k-means++',n_clusters=numP)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    return centers

def distance2(p1,p2):
    ans = 0
    for i in xrange(p1.shape[0]):
        ans += (p1[i] - p2[i]) ** 2
    return ans

def similarity(p1,p2,sigma):
    return np.exp((-distance2(p1,p2))/(2*(sigma**2)))





def unsupervisedGraph(X,centers,sigma):
    numS = X.shape[0]
    numP = centers.shape[0]
    # print numS,numP
    N = numS + numP
    W = np.zeros((N,N))
    for i in xrange(numS):
        for j in xrange(numP):
            W[i,numS+j] = similarity(X[i],centers[j],sigma)
            W[numS+j,i] = W[i,numS+j]
    d = np.array(W.sum(axis=0))
    D = np.diag(d)
    L = D - W
    return W,D,L

def unsupervised(X,y,numP,sigma):
    s1 = clock()
    numS = X.shape[0]
    t1 = clock()
    centers = Prototyping(X,numP)
    t2 = clock()
    # print 'unsupervised prototype: ', t2-t1
    W,D,L = unsupervisedGraph(X,centers,sigma)
    vals, vecs = sps.linalg.eigsh(L, M=D, k=7)
    s2 = clock()
    # print 'runtime:', s2-s1
    return t2-t1, s2-s1



def supervisedGraph(X,y,centers,sigma):
    numS = X.shape[0]
    numP = centers.shape[0]
    numL = len(np.unique(sorted(y)))
    # print numS,numP,numL
    N = numS+numP+numL
    W = np.zeros((N,N))
    for i in xrange(numS):
        for j in xrange(numP):
            W[i,numS+j] = similarity(X[i],centers[j],sigma)
            W[numS+j,i] = W[i,numS+j]

    P1 = np.sum(y)
    P2 = len(y) - P1
    for i in xrange(numS):
        W[i,numS+numP+y[i]] = 1.
        W[numS+numP+y[i],i] = 1.


    d = np.array(W.sum(axis=0))
    D = np.diag(d)
    L = D - W
    return W,D,L

def supervised(X,y,numP,sigma):
    s1 = clock()
    numS = X.shape[0]
    t1 = clock()
    centers = Prototyping(X,numP)
    t2 = clock()
    # print "supervised prototype: ", t2-t1
    W,D,L = supervisedGraph(X,y,centers,sigma)
    vals, vecs = sps.linalg.eigsh(L, M=D, k=7)
    s2 = clock()
    # print 'runtime:', s2-s1
    return t2-t1, s2-s1
