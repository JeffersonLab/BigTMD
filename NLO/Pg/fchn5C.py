#!/usr/bin/env python
import numpy as np
from mpmath import fp
from numba import jit
import numpy as np
EulerGamma=np.euler_gamma
@jit(cache=True)
def _PolyLOG(s, z):
    tol = 1e-10
    l = 0
    k = 1
    zk = z
    while 1:
        term = zk / k**s
        l += term
        if abs(term) < tol:
            break
        zk *= z
        k += 1
    return l
@jit(cache=True)
def PolyLOG(s, z):
    #return fp.polylog(s,z)
    #if abs(z) > 0.75:
    #  return -PolyLOG(s,1-z) + np.pi**2/6 - np.log(z)*np.log(1-z)
    if abs(z) >1: 
      return -PolyLOG(s, 1/z) - np.pi**2/6 - 0.5*np.log(-z)**2
    return _PolyLOG(s, z)
@jit(cache=True)
def regular(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,nf=None):
    return 0
@jit(cache=True)
def delta(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
@jit(cache=True)
def plus1B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
@jit(cache=True)
def plus2B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
