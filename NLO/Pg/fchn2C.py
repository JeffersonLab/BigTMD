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
    return 0.0416666666666667*(s23 - t)**2*(2 - (2*Q**2 + s + t)**2/(4*Q**2*s23 + (s + t)**2) - (4*Q**2*s23 + (s + t)**2)**(-2.5)*(-4*Q**2 - 2*s - 2*t + 2*np.sqrt(4*Q**2*s23 + (s + t)**2))*(2*Q**2 + s + t + np.sqrt(4*Q**2*s23 + (s + t)**2))*(6*Q**6*s23**2 + Q**4*s23*(9*s*s23 + 6*s*t + 2*s23**2 - 3*s23*t + 4*t**2) + Q**2*(s**2*(5*s23**2 + 6*s23*t + t**2) + s*(s23**3 + 3*s23*t**2 + 2*t**3) + t*(-3*s23**3 + 5*s23**2*t - 3*s23*t**2 + t**3)) + s*s23*(s + t)*(s*(s23 + 2*t) + 2*t*(-s23 + t)))*np.log((2*Q**2 + s + t - np.sqrt(4*Q**2*s23 + (s + t)**2))/(2*Q**2 + s + t + np.sqrt(4*Q**2*s23 + (s + t)**2)))/(s23 - t)**2 + (12*Q**4 + 4*Q**2*(3*s - 2*s23 + 3*t) + (s + t)**2)*(2*Q**2*s23 + s*(s23 + t) + t*(-s23 + t))**2/((s23 - t)**2*(4*Q**2*s23 + (s + t)**2)**2) - (4*Q**2 + 2*s + 2*t)*(2*Q**2*s23 + s*(s23 + t) + t*(-s23 + t))/((-s23 + t)*(4*Q**2*s23 + (s + t)**2)))/(np.pi**5*(Q**2 + s - s23 + t)**3) - 0.0208333333333333*(s23 - t)**2*(s - s23 + t)*(-(6 - 2*(2*Q**2 + s + t)**2/(4*Q**2*s23 + (s + t)**2) + 4*(6*Q**4 + Q**2*(6*s - 2*s23 + 6*t) + (s + t)**2)*(2*Q**2*s23 + s*(s23 + t) + t*(-s23 + t))**2/((s23 - t)**2*(4*Q**2*s23 + (s + t)**2)**2) - 2*(8*Q**2 + 4*s + 4*t)*(2*Q**2*s23 + s*(s23 + t) + t*(-s23 + t))/((-s23 + t)*(4*Q**2*s23 + (s + t)**2)))*np.log((2*Q**2 + s + t - np.sqrt(4*Q**2*s23 + (s + t)**2))/(2*Q**2 + s + t + np.sqrt(4*Q**2*s23 + (s + t)**2))) + (8*Q**2 + 4*s + 4*t)/np.sqrt(4*Q**2*s23 + (s + t)**2) - (4*Q**2*s23 + (s + t)**2)**(-1.5)*(24*Q**2 + 12*s + 12*t)*(2*Q**2*s23 + s*(s23 + t) + t*(-s23 + t))**2/(s23 - t)**2 + (32*Q**2*s23 + 16*s*(s23 + t) + 16*t*(-s23 + t))/((-s23 + t)*np.sqrt(4*Q**2*s23 + (s + t)**2)))/(np.pi**5*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2*np.sqrt(4*Q**2*s23 + s**2 + 2*s*t + t**2)) + 0.0138888888888889*(s23 - t)*((24*Q**4*s23 + 12*Q**2*(s*(2*s23 + t) + 2*s23**2 - 2*s23*t + t**2) + 12*s*s23*(s + t))*np.log((2*Q**2 + s + t - np.sqrt(4*Q**2*s23 + (s + t)**2))/(2*Q**2 + s + t + np.sqrt(4*Q**2*s23 + (s + t)**2)))/((-s23 + t)*(4*Q**2*s23 + (s + t)**2)) + (24*Q**2*s23 + 12*s*(s23 + t) + 12*t*(-s23 + t))/((-s23 + t)*np.sqrt(4*Q**2*s23 + (s + t)**2)))*(2*Q**4 + 3*Q**2*s - 2*Q**2*s23 + 3*Q**2*t + s**2 + s*s23 - 2*s23**2 + 3*s23*t - t**2)/(np.pi**5*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2*np.sqrt(4*Q**2*s23 + s**2 + 2*s*t + t**2)) - 0.00347222222222222*(s23 - t)*(24 - 12*(4*Q**2*s23 + (s + t)**2)**(-1.5)*(2*Q**2 + s + t)**2*(2*Q**2*s23 + s*(s23 + t) + t*(-s23 + t))*np.log((2*Q**2 + s + t - np.sqrt(4*Q**2*s23 + (s + t)**2))/(2*Q**2 + s + t + np.sqrt(4*Q**2*s23 + (s + t)**2)))/(-s23 + t) - (48*Q**2 + 24*s + 24*t)*(2*Q**2*s23 + s*(s23 + t) + t*(-s23 + t))/((-s23 + t)*(4*Q**2*s23 + (s + t)**2)) + (24*Q**2*s23 + 12*s*(s23 + t) + 12*t*(-s23 + t))*np.log((2*Q**2 + s + t - np.sqrt(4*Q**2*s23 + (s + t)**2))/(2*Q**2 + s + t + np.sqrt(4*Q**2*s23 + (s + t)**2)))/((-s23 + t)*np.sqrt(4*Q**2*s23 + (s + t)**2)))*(Q**2 + s + s23 - t)/(np.pi**5*(Q**2 + s - s23 + t)**3) + 0.0833333333333333*(Q**4 + 2*Q**2*s + s**2 + s23**2 - 2*s23*t + t**2)/(np.pi**5*(Q**2 + s - s23 + t)**3) - 0.0833333333333333*(4*Q**4 + 4*Q**2*s + s**2 + 2*s23**2 - 2*s23*t + t**2)*np.log((2*Q**2 + s + t + np.sqrt(4*Q**2*s23 + s**2 + 2*s*t + t**2))/(2*Q**2 + s + t - np.sqrt(4*Q**2*s23 + s**2 + 2*s*t + t**2)))/(np.pi**5*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)*np.sqrt(4*Q**2*s23 + s**2 + 2*s*t + t**2)) - 0.0104166666666667*(s23 - t)*(8*s**2*t - 8*s*s23*t + 8*s*t**2)/(np.pi**5*s*t*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) - 0.0416666666666667*(s23 - t)*(2*s**2*t**2 - 2*s*s23*t**2 + 2*s*t**3)/(np.pi**5*s*t**2*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) + 0.0416666666666667*(4*Q**4*s**2*t**2 + 6*Q**2*s**3*t**2 - 4*Q**2*s**2*s23*t**2 + 6*Q**2*s**2*t**3 + 2*s**4*t**2 + 2*s**3*s23*t**2 - 4*s**2*s23**2*t**2 + 6*s**2*s23*t**3 - 2*s**2*t**4)/(np.pi**5*s**2*t**2*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2)
@jit(cache=True)
def delta(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
@jit(cache=True)
def plus1B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
@jit(cache=True)
def plus2B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
