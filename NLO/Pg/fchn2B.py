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
    return -0.0104166666666667*(s23 - t)*(36*Q**4*s + 12*Q**4*t + 54*Q**2*s**2 - 36*Q**2*s*s23 + 72*Q**2*s*t - 12*Q**2*s23*t + 18*Q**2*t**2 + 18*s**3 - 18*s**2*s23 + 42*s**2*t - 24*s*s23*t + 30*s*t**2 - 6*s23*t**2 + 6*t**3)/(np.pi**5*s*t*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) - 0.0416666666666667*(s23 - t)*(-8*Q**4*s*t - 4*Q**4*t**2 - 12*Q**2*s**2*t + 8*Q**2*s*s23*t - 18*Q**2*s*t**2 + 4*Q**2*s23*t**2 - 6*Q**2*t**3 - 4*s**3*t + 4*s**2*s23*t - 10*s**2*t**2 + 6*s*s23*t**2 - 8*s*t**3 + 2*s23*t**3 - 2*t**4)/(np.pi**5*s*t**2*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2) + 0.0208333333333333*(s - s23)*(s**2 - s*t)/(np.pi**5*s**2*t*(Q**2 + s - s23 + t)) + 0.0416666666666667*(-6*Q**6*s**2*t + 6*Q**6*s*t**2 - 13*Q**4*s**3*t + 8*Q**4*s**2*s23*t - 8*Q**4*s*s23*t**2 + 13*Q**4*s*t**3 - 9*Q**2*s**4*t + 10*Q**2*s**3*s23*t - 9*Q**2*s**3*t**2 - 2*Q**2*s**2*s23**2*t + 9*Q**2*s**2*t**3 + 2*Q**2*s*s23**2*t**2 - 10*Q**2*s*s23*t**3 + 9*Q**2*s*t**4 - 2*s**5*t + 3*s**4*s23*t - 4*s**4*t**2 - s**3*s23**2*t + 3*s**3*s23*t**2 - 3*s**2*s23*t**3 + 4*s**2*t**4 + s*s23**2*t**3 - 3*s*s23*t**4 + 2*s*t**5)/(np.pi**5*s**2*t**2*(2*Q**2 + s + t)*(Q**2 + s - s23 + t)**2)
@jit(cache=True)
def delta(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0.00231481481481481*(7*s + 5*t)*(s23 - t)**2*(6*np.log(B) - 12*np.log(mu) - 13 - 12*np.log(2) - 6*np.log(np.pi) + 6*EulerGamma)/(np.pi**5*s*t*(Q**2 + s - s23 + t)) + 0.333333333333333*(s23 - t)**2*(0.0416666666666667*(-7*s - 5*t)*np.log(B)/np.pi**5 - 0.0833333333333333*(-7*s - 5*t)*np.log(mu)/np.pi**5 - 0.333333333333333*(-7*s - 5*t)*(0.125*np.log(np.pi)/np.pi**5 + 0.25*np.log(2)/np.pi**5) + 0.0416666666666667*EulerGamma*(-7*s - 5*t)/np.pi**5 + 0.00694444444444444*(91*s + 65*t)/np.pi**5)/(s*t*(Q**2 + s - s23 + t)) + 0.333333333333333*(s23 - t)*(0.015625*(-4*Q**2*s - 4*Q**2*t - 6*s**2 - 4*s*t - 6*t**2)/np.pi**5 - 0.015625*(16*Q**2*s + 24*Q**2*t + 12*s**2 - 16*s*t - 4*t**2)*np.log(B)/np.pi**5 + 0.03125*(16*Q**2*s + 24*Q**2*t + 12*s**2 - 16*s*t - 4*t**2)*np.log(mu)/np.pi**5 - 0.015625*EulerGamma*(16*Q**2*s + 24*Q**2*t + 12*s**2 - 16*s*t - 4*t**2)/np.pi**5 + 0.03125*(16*Q**2*s + 24*Q**2*t + 12*s**2 - 16*s*t - 4*t**2)/np.pi**5 + 0.5*(0.03125*np.log(np.pi)/np.pi**5 + 0.0625*np.log(2)/np.pi**5)*(16*Q**2*s + 24*Q**2*t + 12*s**2 - 16*s*t - 4*t**2))/(s*t*(Q**2 + s - s23 + t)) + 0.333333333333333*(-0.03125*(-6*Q**2*s**2 + 6*Q**2*t**2 - 6*s**3 - 2*s**2*t + 2*s*t**2 + 6*t**3)/np.pi**5 + 0.25*(-Q**2*s**2 + Q**2*t**2 - s**3 + s**2*t - s*t**2 + t**3)/np.pi**5 + 0.125*(Q**2*s**2 - Q**2*t**2 + s**3 - s**2*t + s*t**2 - t**3)*np.log(B)/np.pi**5 - 0.25*(Q**2*s**2 - Q**2*t**2 + s**3 - s**2*t + s*t**2 - t**3)*np.log(mu)/np.pi**5 + 0.125*EulerGamma*(Q**2*s**2 - Q**2*t**2 + s**3 - s**2*t + s*t**2 - t**3)/np.pi**5 - (0.03125*np.log(np.pi)/np.pi**5 + 0.0625*np.log(2)/np.pi**5)*(4*Q**2*s**2 - 4*Q**2*t**2 + 4*s**3 - 4*s**2*t + 4*s*t**2 - 4*t**3))/(s*t*(Q**2 + s - s23 + t)) + 0.333333333333333*(s23 - t)*(0.03125*(-32*Q**2*s*t - 16*Q**2*t**2 - 16*s**2*t + 4*s*t**2 + 12*t**3)/np.pi**5 + 0.015625*(24*Q**2*s*t - 8*Q**2*t**2 + 14*s**2*t + 8*s*t**2 - 6*t**3)/np.pi**5 + 0.015625*(32*Q**2*s*t + 16*Q**2*t**2 + 16*s**2*t - 4*s*t**2 - 12*t**3)*np.log(B)/np.pi**5 - 0.03125*(32*Q**2*s*t + 16*Q**2*t**2 + 16*s**2*t - 4*s*t**2 - 12*t**3)*np.log(mu)/np.pi**5 - 0.5*(0.03125*np.log(np.pi)/np.pi**5 + 0.0625*np.log(2)/np.pi**5)*(32*Q**2*s*t + 16*Q**2*t**2 + 16*s**2*t - 4*s*t**2 - 12*t**3) + 0.015625*EulerGamma*(32*Q**2*s*t + 16*Q**2*t**2 + 16*s**2*t - 4*s*t**2 - 12*t**3)/np.pi**5)/(s*t**2*(Q**2 + s - s23 + t)) + 0.333333333333333*(s - s23)*(0.03125*(-6*Q**2*s**2 + 10*Q**2*s*t - 6*s**3 + 2*s**2*t + 4*s*t**2)/np.pi**5 - 0.03125*(4*Q**2*s**2 - 8*Q**2*s*t + 4*s**3 - 6*s**2*t - 2*s*t**2)*np.log(B)/np.pi**5 + 0.0625*(4*Q**2*s**2 - 8*Q**2*s*t + 4*s**3 - 6*s**2*t - 2*s*t**2)*np.log(mu)/np.pi**5 - 0.03125*EulerGamma*(4*Q**2*s**2 - 8*Q**2*s*t + 4*s**3 - 6*s**2*t - 2*s*t**2)/np.pi**5 + 0.0625*(4*Q**2*s**2 - 8*Q**2*s*t + 4*s**3 - 6*s**2*t - 2*s*t**2)/np.pi**5 + 0.5*(0.0625*np.log(np.pi)/np.pi**5 + 0.125*np.log(2)/np.pi**5)*(4*Q**2*s**2 - 8*Q**2*s*t + 4*s**3 - 6*s**2*t - 2*s*t**2))/(s**2*t*(Q**2 + s - s23 + t))
@jit(cache=True)
def plus1B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0.0625*s23*(s - t)*(2*Q**2 + s + t)/(np.pi**5*s*t*(Q**2 + s - s23 + t))
@jit(cache=True)
def plus2B(g=None,gp=None,s=None,t=None,Q=None,s23=None,mu=None,B=None,nf=None):
    return 0
