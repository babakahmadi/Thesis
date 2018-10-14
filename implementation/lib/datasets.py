import numpy as np
import pandas as pd
import networkx as nx
import scipy as sp
def loadHeart():
    heart = pd.read_csv('data/new/Heart_disease.data',header=None)
    numeric = [0,3,4,7,9,11]
    nominal = [1,2,5,6,8,10,12]
    heartNum = heart[numeric].values
    heartCat = heart[nominal].values-1
    heartY = heart[13].values
    return heartCat,heartNum,heartY-1

def f12(x):
    if x==3:
        return 0
    if x==6:
        return 1
    return 2
def loadStatlog():
    statlog = pd.read_csv('data/Categorical/statlog.data',header=None)
    numeric = [0,3,4,7,9,11]
    nominal = [1,2,5,6,8,10,12]
    statlogNum = statlog[numeric].values
    statlogCat = statlog[nominal]
    statlogCat.loc[:,2] = statlogCat[2]-1
    statlogCat.loc[:,10] = statlogCat[10]-1
    statlogCat.loc[:,12] = statlogCat[12].apply(f12)
    statlogCat = statlogCat.values
    statlogY = statlog[13].values
    return statlogCat,statlogNum,statlogY-1

valDict = {}
def loadGerman():
    german = pd.read_csv('data/Categorical/german.data',header=None)
    categoricG = [0,2,3,5,6,7,8,9,11,13,14,15,16,17,18,19]
    numericG = [1,4,12]
    for col in categoricG:
        values = np.unique(sorted(german[col]))
        valDict = {}
        for idx in xrange(len(values)):
            valDict[values[idx]] = idx
        german.loc[:,col] = german[col].apply(lambda x: valDict[x])
    germanNum = german[numericG].values
    germanCat = german[categoricG].values
    germanY = german[20].values-1
    return germanCat,germanNum,germanY

def loadSoybean():
    soybean = pd.read_csv('data/Categorical/soyBean.data',header=None)
    for col in soybean.columns:
        values = np.unique(sorted(soybean[col]))
        valDict = {}
        for idx in xrange(len(values)):
            valDict[values[idx]] = idx
        soybean.loc[:,col] = soybean[col].apply(lambda x: valDict[x])
    soybeanCat = soybean.drop(20,axis=1).values
    soybeanY = soybean[20].values-1
    return soybeanCat,soybeanY

def loadDermat():
    dermat = pd.read_csv('data/Categorical/dermatology.data',header=None)
    dermatCat = dermat.drop([33,34],axis=1).values
    dermatNum = dermat[[33]].values
    dermatY = dermat[34].values-1
    return dermatCat,dermatNum,dermatY
	

def loadHeart2():
    heart = pd.read_csv('data/new/Heart_disease.data',header=None)
    numeric = [0,3,4,7,9,11]
    nominal = [1,2,5,6,8,10,12]
    for col in nominal:
        values = np.unique(sorted(heart[col]))
        if len(values)<=2:
            continue
        for val in values:
            name = str(col)+'_'+str(val)
            heart[name] = heart[col].apply(lambda x: 1 if x==val else 0)
        heart = heart.drop([col],axis=1)
    heartX = heart.drop(13).values
    heartY = heart[13].values-1
    return heartX,heartY

def f12(x):
    if x==3:
        return 0
    if x==6:
        return 1
    return 2

def loadStatlog2():
    statlog = pd.read_csv('data/Categorical/statlog.data',header=None)
    statlog.loc[:,2] = statlog[2]-1
    statlog.loc[:,10] = statlog[10]-1
    statlog.loc[:,12] = statlog[12].apply(f12)
    statlogX = statlog.drop(13,axis=1).values
    statlogY = statlog[13].values
    return statlogX,statlogY-1

valDict = {}
def loadGerman2():
    german = pd.read_csv('data/Categorical/german.data',header=None)
    categoricG = [0,2,3,5,6,7,8,9,11,13,14,15,16,17,18,19]
    for col in categoricG:
        values = np.unique(sorted(german[col]))
        valDict = {}
        for idx in xrange(len(values)):
            valDict[values[idx]] = idx
        german.loc[:,col] = german[col].apply(lambda x: valDict[x])
    germanX = german.drop(20).values
    germanY = german[20].values-1
    return germanX,germanY


def loadDermat2():
    dermat = pd.read_csv('data/Categorical/dermatology.data',header=None)
    dermatX = dermat.drop([34],axis=1).values
    dermatY = dermat[34].values-1
    return dermatX,dermatY
	

def generateTwoGaussian():
    mean1 = [-2, -2]
    cov1 = [[1, 0], [0, 1]]  # diagonal covariance
    mean2 = [2,2]
    cov2 = [[1, 0], [0, 1]]  # diagonal covariance

    # data
    X = np.zeros((200,2))
    y = np.zeros(200,dtype=int)
    X[:100] = np.random.multivariate_normal(mean1, cov1, 100)
    X[100:] = np.random.multivariate_normal(mean2, cov2, 100)
    y[100:] = np.ones(100)
    return X,y,"Gaussian"

def generateXOR():
    mean1 = [-2, -2]
    cov1 = [[1, 0], [0, 1]]  # diagonal covariance
    mean2 = [2,2]
    cov2 = [[1, 0], [0, 1]]  # diagonal covariance

    # data
    X = np.zeros((400,2))
    y = np.zeros(400,dtype=int)
    X[:100] = np.random.multivariate_normal(mean1, cov1, 100)
    X[100:200] = np.random.multivariate_normal(mean2, cov2, 100)
    
    
    mean3 = [2, -2]
    cov3 = [[1, 0], [0, 1]]  # diagonal covariance
    mean4 = [-2, 2]
    cov4 = [[1, 0], [0, 1]]  # diagonal covariance

    X[200:300] = np.random.multivariate_normal(mean3, cov3, 100)
    X[300:] = np.random.multivariate_normal(mean4, cov4, 100)
    y[200:] = np.ones(200)
    
    return X,y,"XOR"

def generateTwoParabola():
    x1 = np.linspace(-10, 10, 40)
    y1 = .22*x1**2 + .024*x1 + .04  

    x2 = np.linspace(0, 20, 40)
    y2 = -.2*(x2-10)**2 - .024*(x2-10) + 35

    size = x1.shape[0]
    EACH = 10
    X = np.zeros((2*size*EACH,2))
    y = np.zeros(2*size*EACH,dtype=int)
    y[size*EACH:] = np.ones(size*EACH)

    cov = [[1, 0], [0, 1]]  # diagonal covariance
    for i in range(size):
        X[i*EACH:(i+1)*EACH] = np.random.multivariate_normal([x1[i],y1[i]], cov, EACH)
        X[(i+size)*EACH:(i+size+1)*EACH] = np.random.multivariate_normal([x2[i],y2[i]], cov, EACH)
    return X,y,"Parabolas"
	

# wine  -------------------------------------------------------------------------------------------------
def loadWine():
    wineData = pd.read_csv('data/wine.data',header=None)
    wineY = wineData[0].apply(lambda x: x-1).values
    wineX = wineData.drop(0,axis=1)
    wineX = wineX.values
    return wineX,wineY

# wine  -------------------------------------------------------------------------------------------------
def loadWineNorm():
    wineData = pd.read_csv('data/wine.data',header=None)
    wineY = wineData[0].apply(lambda x: x-1).values
    wineX = wineData.drop(0,axis=1)
    wineX = wineX.values
    wineX = (wineX - wineX.mean(axis=0))/(wineX.max(axis=0)-wineX.min(axis=0))
    return wineX,wineY


# glass  -------------------------------------------------------------------------------------------------
def loadGlass():
    glassData = pd.read_csv('data/glass.data',header=None)
    glassY = glassData[10].apply(lambda x: x-1 if x<4 else x-2).values
    glassX = glassData.drop([0,10],axis=1)
    glassX = glassX.values
    return glassX,glassY
    
# sonar  -------------------------------------------------------------------------------------------------
def loadSonar():
    sonarData = pd.read_csv('data/sonar.data',header=None)
    sonarY = (sonarData[60].apply(lambda x: 0 if x=='M' else 1)).values
    sonarX = sonarData.drop(60,axis=1)
    sonarX = sonarX.values
    return sonarX,sonarY

# iris  -------------------------------------------------------------------------------------------------
def irisLableSet(name):
    names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    for i in xrange(3):
        if names[i] == name:
            return i
def loadIris():
    irisData = pd.read_csv('data/iris.data',header=None)
    irisY = irisData[4].apply(irisLableSet).values
    irisX = irisData.drop(4,axis=1)
    irisX = irisX.values
    return irisX,irisY
    
# mnist   -------------------------------------------------------------------------------------------------
import scipy.io
def loadMnist():
    mnist = scipy.io.loadmat('data/mnistAll.mat')
    trains = []
    labels = []
    tests = []
    for i in range(10):
        trains.append(mnist['train'+str(i)])
        num = mnist['train'+str(i)].shape[0]
        labels.append(i*np.ones(num,dtype=int))
    mnistX = np.concatenate(trains)
    mnistY = np.concatenate(labels)
    return mnistX,mnistY
    
# breast cancer  -------------------------------------------------------------------------------------------------
def loadWdbc():
    wdbcData = pd.read_csv('data/wdbc.data',header=None)
    wdbcY = (wdbcData[1].apply(lambda x: 0 if x=='M' else 1)).values
    wdbcX = wdbcData.drop([0,1],axis=1)
    wdbcX = wdbcX.values
    return wdbcX,wdbcY

# usps   -------------------------------------------------------------------------------------------------
def loadUsps():
    with open('data/usps/usps_train.jf','r') as f:
        i = 0
        lines = f.readlines()
        uspsX = np.zeros((len(lines),256))
        uspsY = np.zeros(len(lines),dtype=int)
        for line in lines:
            splitedLine = line.split()
            features = map(float,splitedLine[1:])
            uspsY[i] = int(splitedLine[0])
            for j in xrange(256):
                uspsX[i,j] = features[j]
            i += 1
    return uspsX,uspsY

# waveform  -------------------------------------------------------------------------------------------------
def loadWaveform():
    waveformData = pd.read_csv('data/waveForm/waveform.data',header=None)
    waveformY = waveformData[21].values
    waveformX = waveformData.drop(21,axis=1)
    waveformX = waveformX.values
    return waveformX,waveformY 
	