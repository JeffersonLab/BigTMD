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
    return -0.0208333333333333*(s23 - t)**2*(2*Q**4 + 3*Q**2*s - 2*Q**2*s23 + 3*Q**2*t + s**2 - s*s23 + 2*s*t - s23*t + t**2)/(np.pi**5*t*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) - 0.00520833333333333*(s23 - t)*(-4*Q**6 - 10*Q**4*s - 4*Q**4*s23 + 10*Q**4*t - 8*Q**2*s**2 - 6*Q**2*s*s23 + 14*Q**2*s*t + 8*Q**2*s23**2 - 26*Q**2*s23*t + 22*Q**2*t**2 - 2*s**3 - 2*s**2*s23 + 4*s**2*t + 4*s*s23**2 - 14*s*s23*t + 14*s*t**2 + 4*s23**2*t - 12*s23*t**2 + 8*t**3)/(np.pi**5*t*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) - 0.00694444444444444*(s23 - t)**2*(-6*Q**4*t - 9*Q**2*s*t + 6*Q**2*s23*t - 9*Q**2*t**2 - 3*s**2*t + 3*s*s23*t - 6*s*t**2 + 3*s23*t**2 - 3*t**3)/(np.pi**5*t**2*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) + 0.0104166666666667*(s23 - t)*(2*Q**6*t + 5*Q**4*s*t - 10*Q**4*s23*t + 19*Q**4*t**2 + 4*Q**2*s**2*t - 15*Q**2*s*s23*t + 29*Q**2*s*t**2 + 8*Q**2*s23**2*t - 29*Q**2*s23*t**2 + 25*Q**2*t**3 + s**3*t - 5*s**2*s23*t + 10*s**2*t**2 + 4*s*s23**2*t - 17*s*s23*t**2 + 17*s*t**3 + 4*s23**2*t**2 - 12*s23*t**3 + 8*t**4)/(np.pi**5*t**2*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) + 0.0208333333333333*(-2*Q**4*s23*t + 4*Q**4*t**2 - 3*Q**2*s*s23*t + 6*Q**2*s*t**2 + 2*Q**2*s23**2*t - 7*Q**2*s23*t**2 + 6*Q**2*t**3 - s**2*s23*t + 2*s**2*t**2 + s*s23**2*t - 4*s*s23*t**2 + 4*s*t**3 + s23**2*t**2 - 3*s23*t**3 + 2*t**4)/(np.pi**5*t**2*(2*Q**2 + s + t)*(Q**2 + s - s23 + t))
@jit(cache=True)
def delta(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return -0.0208333333333333*t*(np.log(B) - 2*np.log(mu) - 2*np.log(2) - np.log(np.pi) - 1 + EulerGamma)/np.pi**5 + (s23 - t)*(0.0625*(-Q**2 - s + 2*t)/np.pi**5 + 0.03125*(Q**2 + s - 2*t)*np.log(B)/np.pi**5 - 0.0625*(Q**2 + s - 2*t)*np.log(mu)/np.pi**5 + 0.03125*EulerGamma*(Q**2 + s - 2*t)/np.pi**5 + 0.03125*(Q**2 + s + t)/np.pi**5 - 0.5*(0.03125*np.log(np.pi)/np.pi**5 + 0.0625*np.log(2)/np.pi**5)*(2*Q**2 + 2*s - 4*t))/(-3.0*Q**2 - 3.0*s + 3.0*s23 - 3.0*t) + 0.00115740740740741*(s23 - t)**2*(Q**2 + s + 3*t)*(6*np.log(B) - 12*np.log(mu) - 13 - 12*np.log(2) - 6*np.log(np.pi) + 6*EulerGamma)/(np.pi**5*t*(Q**2 + s - s23 + t)) + (s23 - t)*(-0.015625*(-2*Q**2*t - 2*s*t - 8*t**2)*np.log(B)/np.pi**5 + 0.03125*(-2*Q**2*t - 2*s*t - 8*t**2)*np.log(mu)/np.pi**5 - 0.015625*EulerGamma*(-2*Q**2*t - 2*s*t - 8*t**2)/np.pi**5 + 0.03125*(-2*Q**2*t - 2*s*t - 8*t**2)/np.pi**5 + 0.5*(0.03125*np.log(np.pi)/np.pi**5 + 0.0625*np.log(2)/np.pi**5)*(-2*Q**2*t - 2*s*t - 8*t**2) + 0.03125*(Q**2*t + s*t + t**2)/np.pi**5)/(t*(-3.0*Q**2 - 3.0*s + 3.0*s23 - 3.0*t)) + 0.00115740740740741*(s23 - t)**2*(-Q**2*t - s*t - 3*t**2)*(6*np.log(B) - 12*np.log(mu) - 13 - 12*np.log(2) - 6*np.log(np.pi) + 6*EulerGamma)/(np.pi**5*t**2*(Q**2 + s - s23 + t))
@jit(cache=True)
def plus1B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return -0.0208333333333333*s23*(Q**2 + s)/(np.pi**5*(Q**2 + s - s23 + t))
@jit(cache=True)
def plus2B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
