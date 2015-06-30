import numpy as np
import scipy.stats as stats
import wishart 

class pfilter():
    
    # Takes a xupdate that changes the particles in the argument
    # Takes ycond that returns the weight to multiply by for one particle.
    # (This is applied over each particle)

    def __init__(self, xsize, ysize, nparts, xupdate, ycond, dt = 1):
        self.xsize = xsize
        self.ysize = ysize
        self.nparts = nparts
        self.ycond = np.vectorize(ycond)
        self.xupdate = xupdate
        self.dt = dt

    def setPrior(self, X, w=None):
        self.X = X
        if w is None:
            self.w = np.ones(self.nparts)/self.nparts
        else:
            self.w = w

    def resample(self):
        self.w = self.w/np.sum(self.w)
        neff = 1/np.sum(np.power(self.w,2))

        if 2*neff < self.nparts:
            # Weights sum to 1. create a grid + U/nparts and a cumsum of w
            # Loop over gridpoints.
            # Take particle i as long as gridpoint[x] < cumsumw[i].
            # Then increase i.

            
            X = np.empty((self.nparts, self.xsize))
            xpoints = np.random.uniform() + np.arange(0, self.nparts)
            cumw = np.cumsum(self.w)
            cumw[-1] = 1 #machine eps problem fix
            i = 0
            x = 0
            
            while x < self.nparts:
                if xpoints[x] <= self.nparts*cumw[i]:
                    X[x] = self.X[i]
                    x += 1
                else:
                    i += 1

            #if changed npart:
            if np.size(self.w) != self.nparts:
                self.w = np.empty(self.nparts)

            self.w.fill(1/self.nparts)
            self.X = X




    def update(self, y, dt=None):
        self.resample()
        if dt is None:
            dt = self.dt
        
        self.xupdate(self.X, dt)
        #add weights to a matrix to use ycond over each row
        import pdb; pdb.set_trace()
        self.w = self.w*np.apply_along_axis(lambda x: self.ycond(y, x, dt), 1, self.X)
        
    def compute(self, f): 
        return np.dot(f(self.X).T, self.w)/np.sum(self.w)

    def mean(self):
        return self.compute(lambda x: x)

    def var(self):
        return self.compute(lambda x: x**2) - self.mean()**2

def invWishart(ysize, sigma, nparts):
    w = wishart.Wishart(ysize)
    w.setDof(ysize) #most uninformative
    w.setChol(np.eye(ysize)/sigma)

    return np.array([w.sample().ravel() for x in range(nparts)])

    #Possible with newer scipy:
    #return np.linalg.inv(stats.wishart.rvs(df=n, scale=np.eye(n)/n/sigma**2, size=nparts))

def norm(n, sigma, nparts):
    return sigma*np.random.randn(nparts, n)

def normCond(y, x, dt):
    # y is logreturns ? for one dt.
    # y ~ N(X, dt)
    # n = y.size
    # mean must be a vector.
    _y = np.array(y)
    return stats.multivariate_normal.pdf(_y, mean=dt*x[0:y.size], 
        cov=dt*x[y.size:])

# Updates have a time factor T where it is back to prior.

def updateNorm(X, dt, ysize, sigmaalpha, sigmasigma):
# Thoughts about wishart:
# Measuring covariance sample adds df for each observation, and emperical Cov. 
# The mean is Psi/(dof - p -1), p = dimensions.
# So measuring gives the mean of all the observations.

# X.shape[0] = nparts

    X[:, 0:ysize] += norm(ysize, sigmaalpha, X.shape[0])*np.sqrt(dt)
    traceX = np.trace(X[:, ysize:])
    X[:, ysize:] = (X[:, ysize:] + invWishart(ysize, sigmasigma*np.sqrt(dt), X.shape[0]) )*traceX/(traceX + sigmasigma**2*dt*ysize )

#THIS IS BAD. We cant just add a positive. When does it go down?
#And does that mean it is not positive definite?

