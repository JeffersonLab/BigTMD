#!/usr/bin/env python
import numpy as np
from mpmath import fp
import numpy as np
def PgA(xh,zh,qT2,Q2):
    return 16*(2*Q2**2*xh*(-xh + 1) + Q2**2*(xh - 1)**2 + xh**2*(2*Q2**2 - 2*Q2*(-Q2*(zh - 1) + qT2*zh) + (-Q2*(zh - 1) + qT2*zh)**2))/(3*Q2*xh*(xh - 1)*(-Q2*(zh - 1) + qT2*zh))
def PgB(xh,zh,qT2,Q2):
    return 16*(-2*Q2**2*(xh - 1)**2 + 2*Q2*xh*(xh - 1)*(Q2*(zh - 1) + Q2 - qT2*zh) - xh**2*(Q2**2 + (-Q2*(zh - 1) + qT2*zh)**2))/(3*Q2*(xh - 1)*(Q2*(xh - 1) + xh*(-Q2*(zh - 1) - Q2 + qT2*zh)))
def PgC(xh,zh,qT2,Q2):
    return 2*(Q2**2*(xh - 1)**2 + 2*Q2*xh*(xh - 1)*(-Q2*(zh - 1) + qT2*zh) + xh**2*(Q2**2 - 2*Q2*(-Q2*(zh - 1) + qT2*zh) + 2*(-Q2*(zh - 1) + qT2*zh)**2))/(xh*(Q2*(xh - 1) + xh*(-Q2*(zh - 1) - Q2 + qT2*zh))*(-Q2*(zh - 1) + qT2*zh))
def PgD(xh,zh,qT2,Q2):
    return 2*(Q2**2*(xh - 1)**2 + 2*Q2*xh*(xh - 1)*(-Q2*(zh - 1) + qT2*zh) + xh**2*(Q2**2 - 2*Q2*(-Q2*(zh - 1) + qT2*zh) + 2*(-Q2*(zh - 1) + qT2*zh)**2))/(xh*(Q2*(xh - 1) + xh*(-Q2*(zh - 1) - Q2 + qT2*zh))*(-Q2*(zh - 1) + qT2*zh))
def PppA(xh,zh,qT2,Q2):
    return 8*(-Q2*(xh - 1) + xh*(Q2*(zh - 1) + Q2 - qT2*zh))/(3*xh)
def PppB(xh,zh,qT2,Q2):
    return -8*Q2*(zh - 1)/3 + 8*qT2*zh/3
def PppC(xh,zh,qT2,Q2):
    return -2*Q2 + 2*Q2/xh
def PppD(xh,zh,qT2,Q2):
    return -2*Q2 + 2*Q2/xh
