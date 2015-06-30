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

xupdate2 = lambda X, dt: pfilter.updateNorm(X, dt, 1, 1, 1)

pf = pfilter.pfilter(2, 1, n, xupdate2, pfilter.normCond, 0.01)
pf.setPrior(pfilter.norm(2, 1, n))

print(pf.mean())

pf2 = pfilter.pfilter(2,1,n,xupdate,ycond,0.01)
pf2.setPrior(np.random.randn(n,2)) 

