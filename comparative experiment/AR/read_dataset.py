import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name,dim,feature,noise=0.0):
    if name == 'lorenz':
        return lorenz(dim,feature,noise)
    if name == 'gene':
        return gene(dim,feature)
    if name == 'solar':
        return solar(dim,feature)
    if name == 'traffic':
        return traffic(dim,feature)
    if name == 'stock':
        return stock(dim,feature)
    if name == 'cardiovas':
        return cardiovas(dim,feature)
    if name == 'plankton':
        return plankton(dim,feature)
    if name == 'wind':
        return wind(dim,feature)
    if name == 'typhoon':
        return typhoon(dim,feature)
    if name == 'ST':
        return ST(dim,feature)

##########  lorenz generator  ######
def  lorenzData(time=1.6, stepsize=0.02, N=30):
    np.random.seed(1)
    n = 3 * N
    m = round(time / stepsize)
    X = np.zeros((n, m))
    X[:, 0] = np.random.rand(1, n)
    C = 0.1
    for i in range(m - 1):
        X[0, i + 1] = X[0, i] + stepsize * (10 * (X[1, i] - X[0, i]) + C * X[0 + (N - 1) * 3, i])
        X[1, i + 1] = X[1, i] + stepsize * (20 * X[0, i] - X[1, i] - X[0, i] * X[2, i])
        X[2, i + 1] = X[2, i] + stepsize * (-8/3 * X[2, i] + X[0, i] * X[1, i])
        for j in range(1, N):
            X[0 + 3 * j, i + 1] = X[0 + 3 * j, i] + stepsize * (10 * (X[1 + 3 * j, i] - X[0 + 3 * j, i]) + C * X[0 + 3 * (j - 1), i])
            X[1 + 3 * j, i + 1] = X[1 + 3 * j, i] + stepsize * (20 * X[0 + 3 * j, i] - X[1 + 3 * j, i] - X[0 + 3 * j, i] * X[2 + 3 * j, i])
            X[2 + 3 * j, i + 1] = X[2 + 3 * j, i] + stepsize * (-8 / 3 * X[2 + 3 * j, i] + X[0 + 3 * j, i] * X[1 + 3 * j, i])

    return X.T


def lorenz(dim,F,noise):

    X = lorenzData()
    X = X + np.random.standard_normal(X.shape)*noise

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/lorenz'+str(dim)+'_'+('free' if noise == 0 else '')+'Noise.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 45:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]

    return X


def gene(dim,F):

    X = pd.read_csv('../../data/gene.csv')
    X = np.array(X).T

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/gene'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 50:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]

    return X

def solar(dim,F):

    X = np.loadtxt('../../data/solar.txt')

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/solar'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 30:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X

def traffic(dim,F):

    X = pd.read_csv('../../data/traffic.csv')
    X = np.array(X)

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/traffic'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 100:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X

def stock(dim,F):

    X = np.loadtxt('../../data/stock.txt')

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/stock'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 50:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X[:70,:]


def cardiovas(dim,F):

    X = pd.read_csv('../../data/cardiovas.csv')
    X = np.array(X)

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/cardiovas'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 10:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X[:100,:]

def plankton(dim,F):

    X = np.loadtxt('../../data/plankton.txt')

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/plankton'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 10:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X[:33,:20]

def wind(dim,F):

    X = pd.read_csv('../../data/wind.csv')
    X = np.array(X)
    X = savgol_filter(X.T, 21,3).T

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/wind'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 70:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X[:70,:]

def typhoon(dim,F):

    X = np.loadtxt('../../data/typhoon.txt')

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/typhoon'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 30:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X

def ST(dim,F):

    X = pd.read_csv('../../data/ST.csv')
    X = np.array(X)
    X = savgol_filter(X.T, 11,4).T

    if F == 1:
        feature = np.array(pd.read_csv('../../data/feature/ST'+str(dim)+'.csv'))
        tmp = []
        tmp.append(X[:,dim])
        j = 1
        for i in feature:
            tmp.append(X[:,int(i)])
            j = j + 1
            if j == 50:break
        X = np.array(tmp).T
    else:
        for i in np.arange(0,X.shape[0],1):
            X[i,0],X[i,dim] = X[i,dim],X[i,0]
            
    return X[:80,:]