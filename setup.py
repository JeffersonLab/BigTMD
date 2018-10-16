#!/usr/bin/env python
import sys
import numpy as np
from NLO import Pg,Ppp

import NLO.Pg.fchn1A,NLO.Ppp.fchn1A 
import NLO.Pg.fchn2A,NLO.Ppp.fchn2A
import NLO.Pg.fchn3A,NLO.Ppp.fchn3A
import NLO.Pg.fchn4A,NLO.Ppp.fchn4A
import NLO.Pg.fchn5A,NLO.Ppp.fchn5A
import NLO.Pg.fchn6A,NLO.Ppp.fchn6A

import NLO.Pg.fchn1B,NLO.Ppp.fchn1B 
import NLO.Pg.fchn2B,NLO.Ppp.fchn2B
import NLO.Pg.fchn3B,NLO.Ppp.fchn3B
import NLO.Pg.fchn4B,NLO.Ppp.fchn4B
import NLO.Pg.fchn5B,NLO.Ppp.fchn5B
import NLO.Pg.fchn6B,NLO.Ppp.fchn6B

import NLO.Pg.fchn1C,NLO.Ppp.fchn1C 
import NLO.Pg.fchn2C,NLO.Ppp.fchn2C
import NLO.Pg.fchn3C,NLO.Ppp.fchn3C
import NLO.Pg.fchn4C,NLO.Ppp.fchn4C
import NLO.Pg.fchn5C,NLO.Ppp.fchn5C
import NLO.Pg.fchn6C,NLO.Ppp.fchn6C


import time


t1=time.time()
cnt=0
for chn in [Pg.fchn1A,Pg.fchn2A,Pg.fchn3A,Pg.fchn4A,Pg.fchn5A,Pg.fchn6A]:
  cnt+=1
  print "optimizing Pg  channel %dA"%cnt
  chn.regular(1,1,20.0,-5.0,10.0,0.01,10.0,4.0)
  chn.delta(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus1B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus2B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
t2=time.time()
print '%0.3e'%(t2-t1)


t1=time.time()
cnt=0
for chn in [Pg.fchn1B,Pg.fchn2B,Pg.fchn3B,Pg.fchn4B,Pg.fchn5B,Pg.fchn6B]:
  cnt+=1
  print "optimizing Pg  channel %dB"%cnt
  chn.regular(1,1,20.0,-5.0,10.0,0.01,10.0,4.0)
  chn.delta(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus1B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus2B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
t2=time.time()
print '%0.3e'%(t2-t1)


t1=time.time()
cnt=0
for chn in [Pg.fchn1C,Pg.fchn2C,Pg.fchn3C,Pg.fchn4C,Pg.fchn5C,Pg.fchn6C]:
  cnt+=1
  print "optimizing Pg  channel %dC"%cnt
  chn.regular(1,1,20.0,-5.0,10.0,0.01,10.0,4.0)
  chn.delta(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus1B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus2B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
t2=time.time()
print '%0.3e'%(t2-t1)



t1=time.time()
cnt=0
for chn in [Ppp.fchn1A,Ppp.fchn2A,Ppp.fchn3A,Ppp.fchn4A,Ppp.fchn5A,Ppp.fchn6A]:
  cnt+=1
  print "optimizing Ppp  channel %dA"%cnt
  chn.regular(1,1,20.0,-5.0,10.0,0.01,10.0,4.0)
  chn.delta(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus1B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus2B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
t2=time.time()
print '%0.3e'%(t2-t1)


t1=time.time()
cnt=0
for chn in [Ppp.fchn1B,Ppp.fchn2B,Ppp.fchn3B,Ppp.fchn4B,Ppp.fchn5B,Ppp.fchn6B]:
  cnt+=1
  print "optimizing Ppp  channel %dB"%cnt
  chn.regular(1,1,20.0,-5.0,10.0,0.01,10.0,4.0)
  chn.delta(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus1B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus2B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
t2=time.time()
print '%0.3e'%(t2-t1)


t1=time.time()
cnt=0
for chn in [Ppp.fchn1C,Ppp.fchn2C,Ppp.fchn3C,Ppp.fchn4C,Ppp.fchn5C,Ppp.fchn6C]:
  cnt+=1
  print "optimizing Ppp  channel %dC"%cnt
  chn.regular(1,1,20.0,-5.0,10.0,0.01,10.0,4.0)
  chn.delta(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus1B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
  chn.plus2B(1,1,20.0,-5.0,10.0,0.01,10.0,1.0,4.0)
t2=time.time()
print '%0.3e'%(t2-t1)












