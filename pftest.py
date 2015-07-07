import pfilter
import numpy as np


def ycond(y, X, dt):
    # vectorized:
    #return np.exp(-(y-X[:,0])**2/X[:,1]**2)/X[:,1]
    return np.exp(-(y-X[0])**2/X[1]**2)/X[1]


def xupdate(X, dt) :
    sigma = np.array((1,1))
    X[:,0] = X[:,0]  + sigma[0]*np.random.randn(X.shape[0])*np.sqrt(dt)

    #geometric diffusion:
    X[:,1] = X[:,1]*np.exp(sigma[1] *np.random.randn(X.shape[0])*np.sqrt(dt))

n = 15

xupdate2 = lambda X, dt: pfilter.diffus:(X, dt, 1, 1, 1)

pf = pfilter.pfilter(2, 1, n, xupdate2, pfilter.normCond, 1)
prior = np.concatenate((pfilter.norm(1, 1, n), np.abs(pfilter.norm(1, 1, n))), axis=1)
pf.setPrior(prior)

print(pf.mean())

pf2 = pfilter.pfilter(2,1,n,xupdate,ycond,1)
pf2.setPrior(np.random.randn(n,2)) 

