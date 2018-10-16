import sys,os
import copy
import numpy as np
from scipy.integrate import quad,fixed_quad
import LO
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
import lhapdf

pdf=lhapdf.mkPDF('CJ15nlo')
ff=lhapdf.mkPDF('dsshpNLO')
#ff=lhapdf.mkPDFs('DSShpnlo')
zero=1e-7
alfa=1/137.036
M=0.93891897
eU= 2./3
eD=-1./3
            # g, u, ub, d, db, s, sb, c, cb, b, bb
eq3=np.array([0,eU,-eU,eD,-eD,eD,-eD, 0,  0, 0,  0])
eq4=np.array([0,eU,-eU,eD,-eD,eD,-eD,eU,-eU, 0,  0])
eq5=np.array([0,eU,-eU,eD,-eD,eD,-eD,eU,-eU,eD,-eD])

iflav=[21,2,-2,1,-1,3,-3,4,-4,5,-5]

qqp=np.ones((10,10))
for i in range(10): qqp[i,i]=0
for i in range(5):
    qqp[2*i,2*i+1]=0
    qqp[2*i+1,2*i]=0

def get_parton_lum(xi,zeta,mu2,chn,case,f,d,eq):
    # case A: gg
    # case B: g-gp
    # case C: gp-gp
    # g:0,u:1,ub:2,d:3,db:4,s:5,sb:6,c:7,cb:8,b:9,bb:10



    # 1 V: A g -> (q->h) qb  R: A g -> (q->h)  qb g  
    # 2 V: A q -> (q->h) g   R: A q -> (q->h)  g  g   
    #                        R: A q -> (q->h)  q' qb'
    # 3 V: A q -> (g->h) q   R: A q -> (g->h)  q  g  
    # 4                      R: A g -> (g->h)  q  qb  
    # 5                      R: A q -> (qb->h) q  qb 
    # 6                      R: A q -> (q'->h) q  qb  

    if   chn==1:
      if case=='A':
        lum=np.sum(eq[1:]*eq[1:]*f[0]*d[1:])
      else: lum=0

    elif chn==3:
      if case=='A': 
        lum=np.sum(eq[1:]*eq[1:]*f[1:]*d[0])
      else: lum=0

    elif chn==4:
      if case=='A': 
        lum=np.sum(eq3[1:]*eq3[1:])*f[0]*d[0]/2
      else: lum=0

    elif chn==5:
      if case =='A':
        idx1=[1,3,5,7,2,4,6,8,10]
        idx2=[2,4,6,8,1,3,5,7,9]
        lum=np.sum(eq[idx1]*eq[idx1]*f[idx1]*d[idx2])
      else: lum=0

    elif chn==2:
      if   case=='A':
        lum=np.sum(eq[1:]*eq[1:]*f[1:]*d[1:])
      elif case=='B':
        lum=np.sum(eq[1:]*f[1:]*d[1:]) * np.sum(eq[1:])
      elif case=='C':
        lum = np.einsum('i,j,ij',f[1:]*d[1:],eq4[1:]*eq4[1:],qqp)/2
    
    elif chn==6:
      if   case=='A':
        lum = np.einsum('i,j,ij',eq[1:]*eq[1:]*f[1:],d[1:],qqp)
      elif   case=='B':
        lum = np.einsum('i,j,ij',eq[1:]*f[1:],eq[1:]*d[1:],qqp)
      elif   case=='C':
        lum = np.einsum('i,j,ij',f[1:],eq[1:]*eq[1:]*d[1:],qqp) 

    return lum

def get_dxsec(xi,s23,x,z,Q,qT,y,E,tar,had,order,part=None,z1=1,z2=0):
   
    mu=np.sqrt((z1*Q)**2 + (z2*qT)**2)
    if mu<1: mu=1.0

    mu2=mu**2
    xh=x/xi

    zh=((1-xh)-xh*s23/Q**2)/((1-xh)+xh*qT**2/Q**2)
    zh0=((1-xh))/((1-xh)+xh*qT**2/Q**2)
    zeta=z/zh
    zeta0=z/zh0

    jac  =zeta  * xh/Q**2 / ((1-xh)-xh*s23/Q**2)
    jac0 =zeta0 * xh/Q**2 / (1-xh)


    gs2=pdf.alphasQ2(mu2)*4*np.pi    

    # get pdfs
    # g:0,u:1,ub:2,d:3,db:4,s:5,sb:6,c:7,cb:8,b:9,bb:10
    f=np.array([pdf.xfxQ2(i,xi,mu2)/xi for i in iflav])

    if tar=='n':
      u,d=f[1],f[3]
      f[1],f[3]=d,u

    if tar=='d':
      u,d=f[1],f[3]
      f[1],f[3]=0.5*(u+d),0.5*(u+d)
    
    # get ffs
    # g:0,u:1,ub:2,d:3,db:4,s:5,sb:6,c:7,cb:8,b:9,bb:10
    if zeta0>0.93: return 0
    if zeta>0.93: return 0
    d0=np.array([ff.xfxQ2(i,zeta0,mu2)/zeta0 for i in iflav])
    d =np.array([ff.xfxQ2(i,zeta,mu2)/zeta for i in iflav])

    Nf=3
    if mu>=pdf.quarkThreshold(4): Nf+=1
    if mu>=pdf.quarkThreshold(5): Nf+=1

    if Nf<5: 
      f[9:11]  = np.zeros(2)
      d[9:11]  = np.zeros(2)
      d0[9:11] = np.zeros(2)
    if Nf<4: 
      f[7:9]  = np.zeros(2)
      d[7:9]  = np.zeros(2)
      d0[7:9] = np.zeros(2)
    if   Nf==3: eq=eq3[:]
    elif Nf==4: eq=eq4[:]
    elif Nf==5: eq=eq5[:]

    Fg,Fpp=0,0

    # 1 V: A g -> (q->h) qb  R: A g -> (q->h)  qb g  
    # 2 V: A q -> (q->h) g   R: A q -> (q->h)  g  g   
    #                        R: A q -> (q->h)  q' qb'
    # 3 V: A q -> (g->h) q   R: A q -> (g->h)  q  g  
    # 4                      R: A g -> (g->h)  q  qb  
    # 5                      R: A q -> (qb->h) q  qb 
    # 6                      R: A q -> (q'->h) q  qb  

    if order==0:

      for chn in [1,2,3]:
        lum0=get_parton_lum(xi,zeta0,mu2,chn,'A',f,d0,eq)
        factor0= gs2 * jac0*lum0/xi/zeta0 * zh0

        if   chn==1: _Pg,_Ppp=LO.PgC,LO.PppC  # C: A g -> (q->h) qb  
        elif chn==2: _Pg,_Ppp=LO.PgA,LO.PppA  # A: A q -> (q->h) g   
        elif chn==3: _Pg,_Ppp=LO.PgB,LO.PppB  # B: A q -> (g->h) q   

        Fg  += _Pg( xh, zh0, qT**2, Q**2)/(2*np.pi)**3*factor0  
        Fpp += _Ppp(xh, zh0, qT**2, Q**2)/(2*np.pi)**3*factor0 

    if order==1:

      s=(1-xh)/xh*Q**2
      t=-(1-zh)*Q**2-zh*qT**2
      t0=-(1-zh0)*Q**2-zh0*qT**2
      nf=4.0
      B=Q**2*(1/xh-1)*(1-z)-z*qT**2

      for chn in [1,2,3,4,6]:

        for case in ['A','B','C']:

          if case=='B' or case=='C':
            if chn==1: continue
            if chn==3: continue
            if chn==4: continue

          lum0=get_parton_lum(xi,zeta0,mu2,chn,case,f,d0,eq)
          lum =get_parton_lum(xi,zeta0,mu2,chn,case,f,d,eq)

          factor = gs2**2 * jac*lum/xi/zeta    * zh#*2#multiply by 2 to remove photon spin sum
          factor0= gs2**2 * jac0*lum0/xi/zeta0 * zh0#*2

          if   chn==1 and case=='A': _Pg,_Ppp=Pg.fchn1A,Ppp.fchn1A   
          elif chn==2 and case=='A': _Pg,_Ppp=Pg.fchn2A,Ppp.fchn2A   
          elif chn==3 and case=='A': _Pg,_Ppp=Pg.fchn3A,Ppp.fchn3A   
          elif chn==4 and case=='A': _Pg,_Ppp=Pg.fchn4A,Ppp.fchn4A   
          elif chn==5 and case=='A': _Pg,_Ppp=Pg.fchn5A,Ppp.fchn5A   
          elif chn==6 and case=='A': _Pg,_Ppp=Pg.fchn6A,Ppp.fchn6A   

          elif chn==1 and case=='B': _Pg,_Ppp=Pg.fchn1B,Ppp.fchn1B   
          elif chn==2 and case=='B': _Pg,_Ppp=Pg.fchn2B,Ppp.fchn2B   
          elif chn==3 and case=='B': _Pg,_Ppp=Pg.fchn3B,Ppp.fchn3B   
          elif chn==4 and case=='B': _Pg,_Ppp=Pg.fchn4B,Ppp.fchn4B   
          elif chn==5 and case=='B': _Pg,_Ppp=Pg.fchn5B,Ppp.fchn5B   
          elif chn==6 and case=='B': _Pg,_Ppp=Pg.fchn6B,Ppp.fchn6B   

          elif chn==1 and case=='C': _Pg,_Ppp=Pg.fchn1C,Ppp.fchn1C   
          elif chn==2 and case=='C': _Pg,_Ppp=Pg.fchn2C,Ppp.fchn2C   
          elif chn==3 and case=='C': _Pg,_Ppp=Pg.fchn3C,Ppp.fchn3C   
          elif chn==4 and case=='C': _Pg,_Ppp=Pg.fchn4C,Ppp.fchn4C   
          elif chn==5 and case=='C': _Pg,_Ppp=Pg.fchn5C,Ppp.fchn5C   
          elif chn==6 and case=='C': _Pg,_Ppp=Pg.fchn6C,Ppp.fchn6C   

          if part=='regular':
            Fg += _Pg.regular(1,1,s,t,Q,s23,Q,nf)*factor
            Fpp+= _Ppp.regular(1,1,s,t,Q,s23,Q,nf)*factor

          if chn<4:

             if part=='delta':
              Fg += _Pg.delta(1,1,s,t0,Q,zero,Q,B,nf)*factor0  
              Fpp+= _Ppp.delta(1,1,s,t0,Q,zero,Q,B,nf)*factor0 
              
             elif part=='plus':
              Fg +=  (_Pg.plus1B(1,1,s,t,Q,s23,Q,B,nf)*factor  - _Pg.plus1B(1,1,s,t0,Q,zero,Q,B,nf)*factor0)/s23
              Fpp+=  (_Ppp.plus1B(1,1,s,t,Q,s23,Q,B,nf)*factor - _Ppp.plus1B(1,1,s,t0,Q,zero,Q,B,nf)*factor0)/s23

              Fg += (_Pg.plus2B(1,1,s,t,Q,s23,Q,B,nf)*factor  - _Pg.plus2B(1,1,s,t0,Q,zero,Q,B,nf)*factor0)*np.log(s23)/s23
              Fpp+= (_Ppp.plus2B(1,1,s,t,Q,s23,Q,B,nf)*factor - _Ppp.plus2B(1,1,s,t0,Q,zero,Q,B,nf)*factor0)*np.log(s23)/s23
          
      if np.isnan(Fg) or np.isnan(Fpp): exit()#return 0  
      if np.isinf(Fg) or np.isinf(Fpp): exit()#return 0 

    F1h = -0.5*Fg + 2*xh**2/Q**2*Fpp
    F2h = -xh*Fg   + 12*xh**3/Q**2*Fpp
    dxsec=np.pi**2*alfa**2/Q**4  * (y**2*F1h+(1-y)*F2h/xh)
    dxsec/=z**2  # from qT^2  -> pT^2
    return dxsec

def A(x,z,qT,Q):
    return x*(1+z*qT**2/(1-z)/Q**2)

def B(x,xi,z,qT,Q):
    return Q**2*(1/(x/xi)-1)*(1-z)-z*qT**2

def get_dxsec_ubox(u,x,z,Q,qT,y,E,tar,had,order,part=None,z1=1,z2=0):
    ximin=A(x,z,qT,Q)
    xi=ximin+u[0]*(1-ximin)

    if order==0:
      return get_dxsec(xi,0,x,z,Q,qT,y,E,tar,had,order,part,z1,z2)*(1-ximin)

    elif order==1:

      if part=='delta':
        return get_dxsec(xi,0,x,z,Q,qT,y,E,tar,had,order,part,z1,z2)*(1-ximin)

      else:
        s23min=0
        s23max=B(x,xi,z,qT,Q)
        s23=s23min+u[1]*(s23max-s23min)
        return get_dxsec(xi,s23,x,z,Q,qT,y,E,tar,had,order,part,z1,z2)*(1-ximin)*(s23max-s23min)

n=40  # for squad and dsquad 

def squad(df_du1,x,z,Q,qT):
    return quad(df_du1,0,1)[0]
    #return fixed_quad(np.vectorize(df_du1),0,1,n=n)[0]

def dquad(df_du1_du2,x,z,Q,qT):
    #df_xi=lambda xi: quad(lambda s23: df_dxi_ds23(xi,s23)[idx],zero,B(x,xi,z,qT,Q))[0]
    #return quad(df_xi,A(x,z,qT,Q),1)[0]
    df_du1=lambda u1: fixed_quad(np.vectorize(lambda u2: df_du1_du2(u1,u2)),0,1,n=n)[0]
    return fixed_quad(np.vectorize(df_du1),0,1,n=n)[0]

def get_xsec(x,z,Q,qT,y,E,tar,had,order,part=None):

  if order==0:
    df_du1=lambda u1: get_dxsec_ubox([u1,0],x,z,Q,qT,y,E,tar,had,0,part)
    return squad(df_du1,x,z,Q,qT)

  if order==1:

    if part=='delta':
      df_du1=lambda u1: get_dxsec_ubox([u1,0],x,z,Q,qT,y,E,tar,had,1,part)
      return squad(df_du1,x,z,Q,qT)

    else:
      df_du1_du2=lambda u1,u2: get_dxsec_ubox([u1,u2],x,z,Q,qT,y,E,tar,had,1,part)
      return dquad(df_du1_du2,x,z,Q,qT)

def get_xsec_scale_var(x,z,Q,qT,y,E,tar,had,order,part=None):

  if order==0:
    values=[]
    for z1 in [0.5,1.0,1.5,2.0]: 
      for z2 in [0,0.5,1.0,1.5,2.0]: 
        df_du1=lambda u1: get_dxsec_ubox([u1,0],x,z,Q,qT,y,E,tar,had,0,part,z1,z2)
        values.append(squad(df_du1,x,z,Q,qT))

  fmin=np.amin(values)
  fmax=np.amax(values)

  return fmin,fmax



