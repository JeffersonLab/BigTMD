#!/usr/bin/env python
import sys,os
import sidis


x=0.1
z=0.3
Q=2.0 
qT=3.0
E=160.0
y=0.439
tar='p'
had='h+'



print "\ncomputing  SIDIS @ LO"
print sidis.get_xsec(x,z,Q,qT,y,E,tar,had,0)

print "\ncomputing  SIDIS @ NLO delta"
print sidis.get_xsec(x,z,Q,qT,y,E,tar,had,1,'delta')

print "\ncomputing  SIDIS @ NLO plus"
print sidis.get_xsec(x,z,Q,qT,y,E,tar,had,1,'plus')

print "\ncomputing  SIDIS @ NLO regular"
print sidis.get_xsec(x,z,Q,qT,y,E,tar,had,1,'regular')





