# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:00:43 2017

@author: Multivac
"""

# coding: utf-8

import pylab
import random
from numpy import *
from math import *
import numpy
import math
import numpy as np
import scipy 

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("white")


class Kernel:    
    
    def __init__(self):
        
        #Parameters of the Window
        self.m=1
        self.p=0.002
        self.q=-0.000000000001
        self.W0=0.00000000001
        
        #Parameters of 2 defined Cells and Animal Movement
        self.Speed=25
        self.OmegaTheta=2*pi*8                          # Angular frequency of the theta oscillation (experimental value of 8Hz)
        self.Ri=50                                      # Radius of the place field 1
        self.Rj=50                                      # Radius of the place field 2
        self.Pi=50                                      # Peak value of Place Field 1
        self.Pj=50                                      # Peak value of Place Field 2
        self.DeltaOmegai=pi*self.Speed/self.Ri          # Frecuency difference between cell 1 and theta
        self.DeltaOmegaj=pi*self.Speed/self.Rj          # Frecuency difference between cell 2 and theta
        self.wi=self.DeltaOmegai+self.OmegaTheta        # Angular frequency of the firing rate of cell 1
        self.wj=self.DeltaOmegaj+self.OmegaTheta        # Angular frecuency of the firing rate of cell 2
        self.si=self.Ri/(sqrt(2*np.log(10)))            # Std.Dev. of Cell 1 - It is fixed by the size of the place field
        self.sj=self.Rj/(sqrt(2*np.log(10)))            # Std.Dev. of Cell 2 - It is fixed by the size of the place field
        
        #Other definitions
        self.s=sqrt(2*(self.si**2*self.sj**2)/(self.si**2+self.sj**2))
        self.sbar=sqrt((self.si**2+self.sj**2)/2)
        self.sbari=sqrt(self.si**2/(self.si**2+self.sj**2))
        self.sbarj=sqrt(self.sj**2/(self.si**2+self.sj**2))
        self.wbar=self.sbari**2*self.wi+self.sbarj**2*self.wj
        self.Dw=self.wi-self.wj
        self.DR=self.Ri-self.Rj
        self.di = lambda r: self.Ri + self.sbari**2*r                                           # THIS ASSUMES GAMMA=X_C=0 - Linear Path passing through centers
        self.dj = lambda r: self.Rj - self.sbarj**2*r                                           # THIS ASSUMES GAMMA=X_C=0 - Linear Path passing through centers
        self.Xr = lambda r: (self.OmegaTheta*(self.di(r)-self.dj(r))/self.Speed)%(2*pi)         # THIS ASSUMES GAMMA=X_C=0 - Linear Path passing through centers
        
        #Parameters of the Kernel
        G0 = lambda r: sqrt(pi)/4*self.s/self.Speed*self.Pi*self.Pj*exp(-r**2/(4*self.sbar**2))
        K = self.W0*(2*pi*self.m**2*self.p**2)**(-1/2)
        self.G = lambda r: K*G0(r)
        self.G1 = 1/2*exp(-self.s**2*self.Dw**2/(4*self.Speed**2))
        self.Gd = lambda r: self.Xr(r)/pi*sin(pi*self.wj/self.OmegaTheta)
        
        self.a = ((self.Speed*self.m*self.p)**2+2*self.sbar**2)/(4*(self.sbar*self.m*self.p)**2)
        self.b = lambda r: self.Speed*r/(2*self.sbar**2)
        self.f = lambda r: (self.wi+self.wj)/(2*self.OmegaTheta)*(self.OmegaTheta*(self.di(r)-self.dj(r))/self.Speed-self.Xr(r))+self.Dw/(2*self.Speed)*(self.di(r)+self.dj(r))-pi*self.Dw/self.OmegaTheta
    
    
    def Window(self): 
        
        print()
        print('The window parameters \'m\' (variance param.) is:',self.m)
        print('The window parameters \'p\' (quadratic param.) is:',self.p)
        print('The window parameters \'q\' (linear param.) is:',self.q)
        print('The window parameters \'W0\' (window factor) is:',self.W0)
        print()
        
        def win(s):
            
            #Next we define a loop for the polynomial
            if self.q==0:                                       # If no linear term, use this expression
                pol=(1-(s/self.p)**2)
            else:                                          # If linear term included, use this expression
                pol=(1+(s/self.q)-(s/self.p)**2)            
            
            return self.W0*(2*pi*self.m**2*self.p**2)**(-1/2)*pol*exp(-(s**2)/(2*self.p**2*self.m**2))

        interval = list(arange(-0.01,0.01,0.0001))
        result= map(win, interval)
        W=list(result)

        plt.figure(figsize=(20,15))       
        plt.plot(interval,W,lw=5)
        plt.axhline(0, color='black',linestyle='dashed', dashes=(2,10))
        plt.axvline(0, color='black',linestyle='dashed', dashes=(2,10))
        plt.tick_params(labelsize=60)
        plt.xticks(np.arange(-0.01, 0.01+0.001, 0.005))
        plt.xlabel('$τ$ (s)',fontsize=65)
        plt.ylabel('Learning Window W($τ$)',fontsize=65)
        plt.show()
        
        
    def parameters(self):
        
        #Parameter a
        print()
        print('The window parameters \'m\' (variance param.) is:',self.m)
        print('The window parameters \'p\' (quadratic param.) is:',self.p)
        print('The window parameters \'q\' (linear param.) is:',self.q)
        print('The window parameters \'W0\' (window factor) is:',self.W0)
        print()
        print('The Cell 1 has Radius Ri={}, Frequency wi={}, Firing Rate Pi={}, Variance si={} and Averaged Variance sbari={}'.format(self.Ri,self.wi,self.Pi,self.si,self.sbari))
        print()
        print('The Cell 2 has Radius Rj={}, Frequency wj={}, Firing Rate Pj={}, Variance sj={} and Averaged Variance sbarj={}'.format(self.Rj,self.wj,self.Pj,self.sj,self.sbarj))
        print()
        print('The weighted variance s={}'.format(self.s))
        print('The squared average of variances sbar={}'.format(self.sbar))
        print()
        print('Value of parameter \'a\' is:',self.a)
        print()

        #Parameter b
        rvalues= list(arange(-50,50,0.001))
        blist=[self.b(r) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,blist,'g')
        plt.title('Parameter \'b\' ',fontsize=14)
        
        print()
        print('Value of parameter \'G1\' is:',self.G1)
        print()
        
        #Parameter G
        rvalues= list(arange(-50,50,0.001))
        blist=[self.G(r) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,blist,'g')
        plt.title('Parameter \'G(r)\' ',fontsize=14)
        
        #Parameter Gd
        rvalues= list(arange(-50,50,0.001))
        blist=[self.Gd(r) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,blist,'g')
        plt.title('Parameter \'Gd(r)\' ',fontsize=14)
        
        
    def phasevariables(self):
        
        rvalues= list(arange(-50,50,0.001))
        
        #di
        dilist=[self.di(r) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,dilist,'g')
        plt.title('\'di\' ',fontsize=14)
        
        #dj
        djlist=[self.dj(r) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,djlist,'g')
        plt.title('\'di\' ',fontsize=14)
        
        #Xr
        Xrlist=[self.Xr(r) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,Xrlist,'g')
        plt.title('\'Xr\' ',fontsize=14)
    
        #PHASE f
        flist=[self.f(r) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,flist,'g')
        plt.title('Parameter \'f\' ',fontsize=14)
        
        #Term (b*w/2*a)
        termlist=[(self.b(r)*self.wbar/(2*self.a)) for r in arange(-50,50,0.001)]
        plt.figure()
        plt.plot(rvalues,termlist,'g')
        plt.title('term (b*w/2*a) ',fontsize=14)
        
        
    def alpha(self):        #In the following lines, we analyze the alpha terms with each other to see their relevance. We omit common Coef.
            
        Alpha1 = lambda r: 1
        Alpha2 = lambda r: -self.b(r)/(2*self.a*self.q)
        Alpha3 = lambda r: -(2*self.a+self.b(r)**2)/(4*self.a**2*self.p**2)
        Alpha = lambda r: Alpha1(r)+Alpha2(r)+Alpha3(r)
        
        rvalues= list(arange(-50,50,0.1))
        
        Alpha1Map=map(Alpha1, rvalues)
        Alpha1List=list(Alpha1Map)
        Alpha2Map=map(Alpha2, rvalues)
        Alpha2List=list(Alpha2Map)
        Alpha3Map=map(Alpha3, rvalues)
        Alpha3List=list(Alpha3Map)
        AlphaMap=map(Alpha, rvalues)
        AlphaList=list(AlphaMap)
        
        plt.plot(rvalues,Alpha1List,'r')   
        plt.title('Alpha 1 (no Coef.)',fontsize=14)
        plt.show()
        plt.plot(rvalues,Alpha2List,'r')  
        plt.title('Alpha 2 (no Coef.)',fontsize=14)
        plt.show()
        plt.plot(rvalues,Alpha3List,'r')   
        plt.title('Alpha 3 (no Coef.)',fontsize=14)
        plt.show()
        plt.plot(rvalues,AlphaList,'r',lw=3)   
        plt.title('Alpha Term (no Coef.)',fontsize=14)
        
    def beta(self):         #In the following lines, we analyze the beta terms with each other to see their relevance. We omit common Coef.
              
        Beta1 = lambda r: cos( self.f(r)-(self.b(r)*self.wbar)/(2*self.a) )
        Beta2 = lambda r: - (1/self.q) * 1/(2*self.a) * ( self.b(r)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + self.wbar*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) )
        Beta3 = lambda r:  (1/self.p**2) * 1/(4*self.a**2) * (  (self.b(r)**2-self.wbar**2+2*self.a)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + (2*self.b(r)*self.wbar)*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a))  )
        Beta = lambda r: Beta1(r) + Beta2(r) - Beta3(r)
        
        rvalues= list(arange(-50,50,0.1))    
        
        Beta1Map=map(Beta1, rvalues)
        Beta1List=list(Beta1Map)
        Beta2Map=map(Beta2, rvalues)
        Beta2List=list(Beta2Map)
        Beta3Map=map(Beta3, rvalues)
        Beta3List=list(Beta3Map)
        BetaMap=map(Beta, rvalues)
        BetaList=list(BetaMap)
        
        plt.plot(rvalues,Beta1List,'b')   
        plt.title('Beta 1 (no Coef.)',fontsize=14)
        plt.show()
        plt.plot(rvalues,Beta2List,'b')   
        plt.title('Beta 2 (no Coef. - divided by q)',fontsize=14)
        plt.show()
        plt.plot(rvalues,Beta3List,'b')   
        plt.title('Beta 3 (no Coef. - divided by p^2)',fontsize=14)
        plt.show()
        plt.plot(rvalues,BetaList,'b',lw=3)   
        plt.title('Beta (no Coef.)',fontsize=14)
        plt.show()
    
    def delta(self):        #In the following lines, we analyze the beta terms with each other to see their relevance. We omit common Coef.

        Delta1 = lambda r: sin( self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta) )
        Delta2 = lambda r: (1/self.q) * 1/(2*self.a) * ( self.wbar*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) - self.b(r)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta3 = lambda r: (1/self.p**2) * 1/(4*self.a**2) * (  -(2*self.b(r)*self.wbar)*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) + (2*self.a+self.b(r)**2-self.wbar**2)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta = lambda r: Delta1(r) + Delta2(r) - Delta3(r)
        
        rvalues= list(arange(-50,50,0.1))  
        
        Delta1Map=map(Delta1, rvalues)
        Delta1List=list(Delta1Map)
        Delta2Map=map(Delta2, rvalues)
        Delta2List=list(Delta2Map)
        Delta3Map=map(Delta3, rvalues)
        Delta3List=list(Delta3Map)
        DeltaMap=map(Delta, rvalues)
        DeltaList=list(DeltaMap)
        
        plt.plot(rvalues,Delta1List,'b')   
        plt.title('Delta 1 (no Coef.)',fontsize=12)
        plt.show()
        plt.plot(rvalues,Delta2List,'b')   
        plt.title('Delta 2 (no Coef.)',fontsize=12)
        plt.show()
        plt.plot(rvalues,Delta3List,'b')   
        plt.title('Delta 3 (no Coef.)',fontsize=12)
        plt.show()
        plt.plot(rvalues,DeltaList,'b',lw=3)   
        plt.title('Delta (no Coef.)',fontsize=14)
        plt.show()
        
        
    def NotSimplifiedKernel(self):
        
        CoefA = lambda r: sqrt(pi/self.a)*exp(self.b(r)**2/(4*self.a))
        CoefBD = lambda r: sqrt(pi/self.a)*exp(self.b(r)**2/(4*self.a))*exp(-self.wbar**2/(4*self.a))

        rvalues= list(arange(-100,100,0.1))          

        # Alpha
        Alpha1 = lambda r: 1
        Alpha2 = lambda r: -self.b(r)/(2*self.a*self.q)
        Alpha3 = lambda r: -(2*self.a+self.b(r)**2)/(4*self.a**2*self.p**2)
        Alpha = lambda r: Alpha1(r)+Alpha2(r)+Alpha3(r)
        FinalAlpha = lambda r: Alpha(r)*CoefA(r)
        AlphaMap=map(FinalAlpha, rvalues)
        AlphaList=list(AlphaMap)
        
        plt.plot(rvalues,AlphaList,'r')   
        plt.title('Alpha (not simplified)',fontsize=14)
        plt.show()
        
        #Beta
        Beta1 = lambda r: cos( self.f(r)-(self.b(r)*self.wbar)/(2*self.a) )
        Beta2 = lambda r: - (1/self.q) * 1/(2*self.a) * ( self.b(r)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + self.wbar*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) )
        Beta3 = lambda r:  (1/self.p**2) * 1/(4*self.a**2) * (  (self.b(r)**2-self.wbar**2+2*self.a)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + (2*self.b(r)*self.wbar)*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a))  )
        Beta = lambda r: Beta1(r) + Beta2(r) - Beta3(r)
        FinalBeta = lambda r: Beta(r)*CoefBD(r)
        
        BetaMap=map(FinalBeta, rvalues)
        BetaList=list(BetaMap)
        
        plt.plot(rvalues,BetaList,'r')   
        plt.title('Beta (not simplified)',fontsize=14)
        plt.show()
        
        #Delta
        Delta1 = lambda r: sin( self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta) )
        Delta2 = lambda r: (1/self.q) * 1/(2*self.a) * ( self.wbar*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) - self.b(r)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta3 = lambda r: (1/self.p**2) * 1/(4*self.a**2) * (  -(2*self.b(r)*self.wbar)*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) + (2*self.a+self.b(r)**2-self.wbar**2)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta = lambda r: Delta1(r) + Delta2(r) - Delta3(r)
        FinalDelta = lambda r: Delta(r)*CoefBD(r)
        
        DeltaMap=map(FinalDelta, rvalues)
        DeltaList=list(DeltaMap)
        
        plt.plot(rvalues,DeltaList,'r')   
        plt.title('Delta (not simplified)',fontsize=14)
        plt.show()
        
        #Kernel
        Kernel = lambda r: self.G(r)*(FinalAlpha(r)+self.G1*(FinalBeta(r)-self.Gd(r)*FinalDelta(r)))
        
        KernelMap=map(Kernel, rvalues)
        KernelList=list(KernelMap)
        
        plt.figure(figsize=(20,15))             
        plt.plot(rvalues,KernelList,'r',lw=6)   
        plt.axhline(0, color='black',linestyle='dashed', dashes=(2,10))
        plt.axvline(0, color='black',linestyle='dashed', dashes=(2,10))
        plt.tick_params(labelsize=60)
        plt.ylabel('Non-Simplified Kernel $\Gamma(r)$',fontsize=65)
        plt.xlabel('r (cm)', fontsize=65)
        plt.show()
        
        
    def EqualFieldsApprox(self):
        
        rvalues= list(arange(-100,100,0.01))    
        
        CoefA = lambda r: sqrt(pi/self.a)*exp(self.b(r)**2/(4*self.a))
        CoefBD = lambda r: sqrt(pi/self.a)*exp(self.b(r)**2/(4*self.a))*exp(-self.wbar**2/(4*self.a))
        
        Beta1 = lambda r: cos( self.f(r)-(self.b(r)*self.wbar)/(2*self.a) )
        Beta2 = lambda r: - (1/self.q) * 1/(2*self.a) * ( self.b(r)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + self.wbar*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) )
        Beta3 = lambda r:  (1/self.p**2) * 1/(4*self.a**2) * (  (self.b(r)**2-self.wbar**2+2*self.a)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + (2*self.b(r)*self.wbar)*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a))  )
        Beta = lambda r: Beta1(r) + Beta2(r) - Beta3(r)
        FinalBeta = lambda r: Beta(r)*CoefBD(r)
        BetaMap=map(FinalBeta, rvalues)
        BetaList=list(BetaMap)
                
        Delta1 = lambda r: sin( self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta) )
        Delta2 = lambda r: (1/self.q) * 1/(2*self.a) * ( self.wbar*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) - self.b(r)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta3 = lambda r: (1/self.p**2) * 1/(4*self.a**2) * (  -(2*self.b(r)*self.wbar)*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) + (2*self.a+self.b(r)**2-self.wbar**2)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta = lambda r: Delta1(r) + Delta2(r) - Delta3(r)
        FinalDeltaGd = lambda r: self.Gd(r)*Delta(r)*CoefBD(r)
        FinalDeltaGdMap=map(FinalDeltaGd, rvalues)
        FinalDeltaGdList=list(FinalDeltaGdMap)
        
        both= lambda r: +FinalBeta(r)-FinalDeltaGd(r)
        BothMap=map(both,rvalues)
        BothList=list(BothMap)
        
        plt.figure(figsize=(20,15))             
        plt.plot(rvalues,BothList,'r',lw=5)   
        plt.plot(rvalues,BetaList,'b',lw=1.5)
        plt.plot(rvalues,FinalDeltaGdList,'g',lw=1.5)
        plt.axhline(0, color='black',linestyle='dashed', dashes=(2,5))
        plt.tick_params(labelsize=60)
        plt.xlabel('r (cm)', fontsize=65)
        plt.ylabel('β+Gd·$\delta$', fontsize=65)
        plt.show()
        
        print('We see that our entire, very complex expression for Beta-Gd*Delta, could be reduced to a single trigonometric term.')
        print('With trigonometric and phasor relations we will try to simplify it as much as possible.')
        print()
        print('Our first step will be to separate our Kernel')
        
        if self.Ri!=self.Rj:
            print()
            print()
            print('The sizes of the place fields are different. To obtain a result, please make them equal.')
        
        else:
            
            p = pi*self.wj/self.OmegaTheta
            
            #### FIRST TERM ####
            both1 = lambda r: Beta1(r)-self.Gd(r)*Delta1(r)
            both1Map=map(both1,rvalues)
            both1List=list(both1Map)
            
            k1 = lambda r: 1-self.Gd(r)*sin(p)
            k2 = lambda r: -self.Gd(r)*cos(p)
            Term1 = lambda r: k1(r)*cos(self.f(r)) + k2(r)*sin(self.f(r))
            Term1Map = map(Term1,rvalues)
            Term1List = list(Term1Map)
            
            plt.plot(rvalues,Term1List)
            plt.plot(rvalues,both1List)
            plt.title('Term I - Comparision')
            plt.show()        
            
            #### SECOND TERM ####
            both2 = lambda r: Beta2(r)-self.Gd(r)*Delta2(r)
            both2Map = map(both2,rvalues)
            both2List = list(both2Map)
            
            c1 = lambda r: self.b(r)/(2*self.a)
            c2 = lambda r: self.wbar/(2*self.a)
            k3 = lambda r: ( -c1(r) + c1(r)*self.Gd(r)*sin(p) - c2(r)*self.Gd(r)*cos(p) )
            k4 = lambda r: ( -c2(r) + c1(r)*self.Gd(r)*cos(p) + c2(r)*self.Gd(r)*sin(p) )
            Term2 = lambda r: (1/self.q) * ( k3(r)*cos(self.f(r)) + k4(r)*sin(self.f(r)) )
            Term2Map = map(Term2,rvalues)
            Term2List = list(Term2Map)
            
            plt.plot(rvalues,Term2List)
            plt.plot(rvalues,both2List)
            plt.title('Term II - Comparision')
            plt.show()
            
            #### THIRD TERM ####
            both3 = lambda r: Beta3(r)-self.Gd(r)*Delta3(r)
            both3Map = map(both3,rvalues)
            both3List = list(both3Map)
            
            c3 = lambda r: (self.b(r)**2-self.wbar**2+2*self.a)/(4*self.a**2)
            c4 = lambda r: self.b(r)*self.wbar/(2*self.a**2)
            k5 = lambda r: c3(r)+c4(r)*self.Gd(r)*cos(p)-c3(r)*self.Gd(r)*sin(p)
            k6 = lambda r: c4(r)-c3(r)*self.Gd(r)*cos(p)-c4(r)*self.Gd(r)*sin(p)
            Term3 = lambda r: (1/self.p**2) * ( k5(r) * cos(self.f(r)) + k6(r) * sin(self.f(r)) )
            Term3Map = map(Term3,rvalues)
            Term3List = list(Term3Map)
            
            plt.plot(rvalues,Term3List)
            plt.plot(rvalues,both3List)
            plt.title('Term III - Comparision')
            plt.show()
            
            #### ENTIRE EXPRESSION ####
            
            kc = lambda r: k1(r) + k3(r)/self.q - k5(r)/self.p**2
            ks = lambda r: k2(r) + k4(r)/self.q - k6(r)/self.p**2
            
            SimpBoth = lambda r: CoefBD(r) * (kc(r)*cos(self.f(r)) + ks(r)*sin(self.f(r)))
            SimpBothMap = map(SimpBoth,rvalues)
            SimpBothList = list(SimpBothMap)
            
            plt.plot(rvalues,BothList,'r',lw=1)   
            plt.plot(rvalues,SimpBothList,'b',lw=1)   
            plt.title('Beta+Gd*Delta Terms - Exact and Simplified')
            plt.show()
            
            print()
            print('Now, we will make our expression smooth. For this, we assume equal place fields (Ri=Rj).')
            
            #### SMOOTHING OF THE PHASE ####
            
            #We make the smothing term -Gd*Delta = 0, since the term is already smooth
            #The phase offset added from the sin part, comes from the sines appearing in BetaII and BetaIII
            
            Qc = lambda r: 1 - self.b(r)/(2*self.a*self.q) - (self.b(r)**2-self.wbar**2+2*self.a)/(4*self.a**2*self.p**2)
            Qs = lambda r: self.wbar/(2*self.a*self.q) + self.b(r)*self.wbar/(2*self.a**2*self.p**2)
            
            SmoothSimpBoth = SimpBoth = lambda r: CoefBD(r) * ( Qc(r)*cos(pi/self.Ri*r) - Qs(r)*sin(pi/self.Ri*r) )
            SmoothSimpBothMap = map(SmoothSimpBoth,rvalues)
            SmoothSimpBothList = list(SmoothSimpBothMap)
            
            plt.figure(figsize=(20,15))             
            plt.plot(rvalues,BothList,'r',lw=5)   
            plt.plot(rvalues,SmoothSimpBothList,'b',lw=5) 
            plt.axhline(0, color='black',linestyle='dashed', dashes=(2,5))
            plt.tick_params(labelsize=60)
            plt.ylabel('β+Gd·$\delta$', fontsize=65)
            plt.xlabel('r (cm)', fontsize=65)
            plt.show()
                        
        
#==============================================================================
#             #### Qs and Qc simplification ####
#             
#             Qcmap = map(Qc,rvalues)
#             QcList = list(Qcmap)
#             Qsmap = map(Qs,rvalues)
#             QsList = list(Qsmap)
#             
#             plt.plot(rvalues,QcList,'g',lw=1)
#             plt.plot(rvalues,QsList,'orange',lw=1)
#             plt.title('Amplitudes Qs and Qc')
#             plt.show()
#==============================================================================
            

    def EqualFieldsKernel(self):
        
        rvalues= list(arange(-100,100,0.01))          

        # EXACT TERM - TO COMPARE WITH APPROXIMATION
        
        CoefA = lambda r: sqrt(pi/self.a)*exp(self.b(r)**2/(4*self.a))
        CoefBD = lambda r: sqrt(pi/self.a)*exp(self.b(r)**2/(4*self.a))*exp(-self.wbar**2/(4*self.a))
        
        # Alpha
        Alpha1 = lambda r: 1
        Alpha2 = lambda r: -self.b(r)/(2*self.a*self.q)
        Alpha3 = lambda r: -(2*self.a+self.b(r)**2)/(4*self.a**2*self.p**2)
        Alpha = lambda r: Alpha1(r)+Alpha2(r)+Alpha3(r)
        FinalAlpha = lambda r: Alpha(r)*CoefA(r)
        
        #Beta
        Beta1 = lambda r: cos( self.f(r)-(self.b(r)*self.wbar)/(2*self.a) )
        Beta2 = lambda r: - (1/self.q) * 1/(2*self.a) * ( self.b(r)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + self.wbar*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) )
        Beta3 = lambda r:  (1/self.p**2) * 1/(4*self.a**2) * (  (self.b(r)**2-self.wbar**2+2*self.a)*cos(self.f(r)-(self.b(r)*self.wbar)/(2*self.a)) + (2*self.b(r)*self.wbar)*sin(self.f(r)-(self.b(r)*self.wbar)/(2*self.a))  )
        Beta = lambda r: Beta1(r) + Beta2(r) - Beta3(r)
        FinalBeta = lambda r: Beta(r)*CoefBD(r)
        
        #Delta
        Delta1 = lambda r: sin( self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta) )
        Delta2 = lambda r: (1/self.q) * 1/(2*self.a) * ( self.wbar*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) - self.b(r)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta3 = lambda r: (1/self.p**2) * 1/(4*self.a**2) * (  -(2*self.b(r)*self.wbar)*cos(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta)) + (2*self.a+self.b(r)**2-self.wbar**2)*sin(self.f(r) - (self.b(r)*self.wbar)/(2*self.a) + (pi*self.wj/self.OmegaTheta))  )
        Delta = lambda r: Delta1(r) + Delta2(r) - Delta3(r)
        FinalDelta = lambda r: Delta(r)*CoefBD(r)

        #Kernel
        ExactKernel = lambda r: self.G(r)*(FinalAlpha(r)+self.G1*(FinalBeta(r)-self.Gd(r)*FinalDelta(r)))
        
        ExactKernelMap=map(ExactKernel, rvalues)
        ExactKernelList=list(ExactKernelMap)
        
        ################################################################################################################################################################################
            
        # Alpha
        AlphaSimp = lambda r: FinalAlpha(r)
        AlphaSimpMap=map(AlphaSimp,rvalues)
        AlphaSimpList=list(AlphaSimpMap)
        
        AlphaMap =map(FinalAlpha,rvalues)
        AlphaList=list(AlphaMap)
        
        plt.plot(rvalues,AlphaSimpList,'b')  
        plt.plot(rvalues,AlphaList,'r')
        plt.title('Alpha (simplified vs exact)',fontsize=14)
        plt.show()
        
        #Beta&Delta
        Qc = lambda r: 1 - self.b(r)/(2*self.a*self.q) - (self.b(r)**2-self.wbar**2+2*self.a)/(4*self.a**2*self.p**2)
        Qs = lambda r: self.wbar/(2*self.a*self.q) + self.b(r)*self.wbar/(2*self.a**2*self.p**2)
            
        BetaDeltaSimp = lambda r: CoefBD(r) * ( Qc(r)*cos(pi/self.Ri*r) - Qs(r)*sin(pi/self.Ri*r) )
        BetaDeltaSimpMap = map(BetaDeltaSimp,rvalues)
        BetaDeltaSimpList = list(BetaDeltaSimpMap)
        
        BetaDelta = lambda r: FinalBeta(r)-self.Gd(r)*FinalDelta(r)    #Coefficient already included!
        BetaDeltaMap=map(BetaDelta,rvalues)
        BetaDeltaList=list(BetaDeltaMap)
        
        plt.plot(rvalues,BetaDeltaSimpList,'b')   
        plt.plot(rvalues,BetaDeltaList,'r')
        plt.title('Beta&Delta (simplified vs exact)',fontsize=14)
        plt.show()
        
        #Kernel
        SimpKernel = lambda r: self.G(r)*(AlphaSimp(r)+self.G1*(BetaDeltaSimp(r)))
        SimpKernelMap=map(SimpKernel, rvalues)
        SimpKernelList=list(SimpKernelMap)
        
        plt.figure(figsize=(20,15))
        plt.plot(rvalues,ExactKernelList,'r',lw=5)   
        plt.plot(rvalues,SimpKernelList,'b',lw=5) 
        plt.axhline(0, color='black',linestyle='dashed', dashes=(2,10))
        plt.axvline(0, color='black',linestyle='dashed', dashes=(2,10))
        plt.tick_params(labelsize=60)
        plt.ylabel('Kernel $\ \Gamma (r)$',fontsize=65)
        plt.xlabel('r (cm)', fontsize=65)
        plt.show()
        
    
    def PhaseSmoothing(self):
        
        #dRvalues=list(arange(-0.8*R,0.6*R,0.01))
        rvalues=list(arange(-10,10,0.001))
        
        xi = lambda r: self.wi/self.OmegaTheta * (self.OmegaTheta*r/self.Speed-(self.OmegaTheta*r/self.Speed)%(2*pi))
        xiMap=map(xi,rvalues)
        xiList=list(xiMap)
        
        linexi = lambda r: self.OmegaTheta*r/self.Speed
        linexiMap=map(linexi,rvalues)
        linexiList=list(linexiMap)
        
        moduloxi = lambda r: (self.OmegaTheta*r/self.Speed)%(2*pi)
        moduloxiMap=map(moduloxi,rvalues)
        moduloxiList=list(moduloxiMap)
        
        stair = lambda r: (self.OmegaTheta*r/self.Speed) - (self.OmegaTheta*r/self.Speed)%(2*pi)
        stairMap=map(stair,rvalues)
        stairList=list(stairMap)
        
        resta = lambda r: xi(r)-stair(r)
        restaMap=map(resta,rvalues)
        restaList=list(restaMap)
        
        simplexi = lambda r: pi/self.Ri*r
        simplexiMap=map(simplexi,rvalues)
        simplexiList=list(simplexiMap)        
        
        plt.figure(figsize=(20,15))
        plt.plot(rvalues,xiList,'k',lw=5)
        plt.tick_params(labelsize=60)
        plt.ylabel('ξ',fontsize=65)
        plt.xlabel('r (cm)',fontsize=65)
        plt.show()
        
        plt.figure(figsize=(20,15))
        plt.plot(rvalues,stairList,'k',lw=5)
        plt.tick_params(labelsize=60)
        plt.ylabel('$ξ_θ$',fontsize=65)
        plt.xlabel('r (cm)',fontsize=65)
        plt.show()
        
        plt.figure(figsize=(20,15))
        plt.plot(rvalues,linexiList,'b',lw=5)
        plt.plot(rvalues,moduloxiList,'g',lw=3)
        plt.plot(rvalues,stairList,'k',lw=5)
        plt.tick_params(labelsize=60)
        plt.ylabel('\'Linear\' $ξ_θ$',fontsize=65) 
        plt.xlabel('r (cm)',fontsize=65)
        plt.show()
        
        plt.figure(figsize=(20,15))
        plt.plot(rvalues,xiList,'k',lw=5)
        plt.plot(rvalues,stairList,'r',lw=5)
        plt.tick_params(labelsize=60)
        plt.ylabel('ξ and $ξ_θ$',fontsize=65)
        plt.xlabel('r (cm)',fontsize=65)
        plt.show()
        
        plt.figure(figsize=(20,15))
        plt.plot(rvalues,restaList,'orange',lw=5)
        plt.plot(rvalues,simplexiList,'purple',lw=5)
        plt.tick_params(labelsize=60)
        plt.ylabel('ξ\' and $\psi$',fontsize=65)
        plt.xlabel('r (cm)',fontsize=65)
        plt.legend()
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
       

        



        





        
        ###############################
        ###############################
        ###############################
        
        #   THE MATERIAL IN THE FOLLOWING LINES MIGHT BE HELPFUL FOR FOLLOWING GENERLIZATIONS TO
        #   KERNELS OF PLACE FIELDS WITH DIFFERENT SIZES
        
        ###############################
        ###############################
        ###############################
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def GeneralApprox(self):      ###### Here we test the quality of our trigonometric term simplification
        
        rvalues= list(arange(0,150,0.01))     
        
        #The first obtained term gives error for Ri=Rj (divided by zero). We omit it when place fields are equal.
        if self.Ri!=self.Rj:
            #Term with no simplification: just right after integration over all possible initial OmegaTheta
            ExactTerm = lambda r: (self.OmegaTheta)/(pi*self.Dw)*sin(self.Dw*(2*pi-self.Xr(r))/(2*self.OmegaTheta))*cos(self.f(r))  +  self.OmegaTheta/(self.Dw*pi)*sin(self.Dw*self.Xr(r)/(2*self.OmegaTheta))*cos(self.f(r)+2*pi*self.wj/self.OmegaTheta)
            ExactTermMap=map(ExactTerm,rvalues)
            ExactTermList=list(ExactTermMap)
        
        #We simplify here for Dw/OmegaTheta << 1
        beta= lambda r: cos(self.f(r))              
        delta= lambda r:  -self.Gd(r) * sin(self.f(r)+(pi*self.wj/self.OmegaTheta))
        SimpExactTerm= lambda r: beta(r)+delta(r)
        SimpExactTermMap=map(SimpExactTerm,rvalues)
        SimpExactTermList=list(SimpExactTermMap)
        
        #Now, we use trigonometric identities on the above expression
        FinalTerm = lambda r: (1-self.Xr(r)/pi*sin(pi*self.wj/self.OmegaTheta)**2)*cos(self.f(r)) - (self.Xr(r)/(2*pi)*sin(2*pi*self.wj/self.OmegaTheta))*sin(self.f(r))
        FinalTermMap=map(FinalTerm,rvalues)
        FinalTermList=list(FinalTermMap)        
        
        #Using the fasor addition relation, we transform it to
        a= lambda r: 1-self.Xr(r)/pi*sin(pi*self.wj/self.OmegaTheta)**2
        b= lambda r: self.Xr(r)/(2*pi)*sin(2*pi*self.wj/self.OmegaTheta)
        Amplitude= lambda r : sqrt(a(r)**2+b(r)**2) ### Aproximated to 1!!!
        PhasorTerm= lambda r: Amplitude(r)*cos(self.f(r)+math.atan(b(r)/a(r)))
        PhasorTermMap=map(PhasorTerm,rvalues)
        PhasorTermList=list(PhasorTermMap)
        
        #We take the phasor relation
        SimpPhasorTerm= lambda r: cos(self.f(r)+math.atan(b(r)/a(r)))
        SimpPhasorTermMap=map(SimpPhasorTerm,rvalues)
        SimpPhasorTermList=list(SimpPhasorTermMap)
        
        #Now we plot the smooth PhasorTerm, with phase offset atan2(b(0),a(0))+f(0)
        p=(self.wi+self.wj-2*self.OmegaTheta+self.Dw*(self.sbari**2-self.sbarj**2))/(2*self.Speed) 
        PhasorSmoothTerm= lambda r: cos( p*r + math.atan2(b(0),a(0)) + (self.wi+self.wj)/(2*self.OmegaTheta)*(self.OmegaTheta/self.Speed*self.DR-(self.OmegaTheta/self.Speed*self.DR)%(2*pi)) + self.Dw/(2*self.Speed)*(self.Ri+self.Rj) - pi*self.Dw/(self.OmegaTheta) )
        PhasorSmoothTermMap=map(PhasorSmoothTerm,rvalues)
        PhasorSmoothTermList=list(PhasorSmoothTermMap)
        
        
        #################
        ##### PLOTS #####
        #################
        
        if self.Ri!=self.Rj:
            plt.plot(rvalues,ExactTermList,'r',lw=1,label='Exact Term')
        plt.plot(rvalues,SimpExactTermList,'b',lw=1,label= 'Simplified Term')
        plt.plot(rvalues,FinalTermList,'g',lw=1,label= 'Rewritten Simplified Term')
        plt.legend(loc='lower right', title='Legend')
        plt.title('Exact Term vs Simplified Terms',fontsize=14)
        plt.show()    
        print('3 Terms plotted: the exact term after the integral, a simplified one for Dw/wtheta<<1 and another re-written expresion of this last one.')
        print('These last two Simplified Terms appear overlapped since they are equal, and both are almost identical to the Exact Term.')
        print()
        
        plt.plot(rvalues,FinalTermList,'g',lw=1,label='Rewritten Simplified Term')
        plt.plot(rvalues, PhasorTermList,'y',lw=1, label='Phasor Term')
        plt.plot(rvalues, SimpPhasorTermList,'k',lw=1,label='Phasor Term Amplitude\=1')
        plt.legend(loc='lower right', title='Legend')
        plt.title('Phasor Terms (simp. and not simp.) compared to Simplified Term',fontsize=14)
        plt.show()
        print('We take now the Rewritten Simplified Expression. We write it again, considering it now a Phasor Term (addition). We will have also a third term, where we approximate the amplitude to 1.')
        print('Above can be seen the three of them plotted together, being almost identical.')
        print()
        
        plt.plot(rvalues, SimpPhasorTermList,'k',lw=1,label='Phasor Term Simplified')
        plt.plot(rvalues, PhasorSmoothTermList,'darkmagenta',lw=2,label='Phasor Smooth Term')
        leg=plt.legend(loc='lower right', title='Legend')
        plt.title('Phasor Term - Amplitude\=1 vs Amplitude\=1 Smoothen',fontsize=14)
        plt.show()
        print('Now, we compare our Simplified Phasor Term to the Smooth Phasor Term, and see how good our approximation is.')
        print()
        
        
        
    def PhaseOffsetSimplification(self):
        
        R=50
        dRvalues=list(arange(-0.8*R,0.6*R,0.01))
        
        Ri=R
        Rj= lambda dR: R-dR
        wi=self.OmegaTheta+pi*self.Speed/Ri
        wj=lambda dR: self.OmegaTheta+pi*self.Speed/Rj(dR)
        dw= lambda dR: wi-wj(dR)
        
        a = lambda dR:  1 - (dR/self.Speed*self.OmegaTheta)%(2*pi)/(pi) * sin( pi/self.OmegaTheta*(pi*self.Speed/(R-dR)+self.OmegaTheta) )**2
        aMap=map(a,dRvalues)
        aList=list(aMap)
        
        b = lambda dR: (dR/self.Speed*self.OmegaTheta)%(2*pi)/(2*pi)*sin(2*pi*(pi*self.Speed/(R-dR)+self.OmegaTheta)/self.OmegaTheta)
        bMap=map(b,dRvalues)
        bList=list(bMap)
        
        arct = lambda dR: math.atan2(b(dR),a(dR))
        arctMap=map(arct,dRvalues)
        arctList=list(arctMap)
        
        f1 = lambda dR: (wi+wj(dR))/(2*self.OmegaTheta)*(self.OmegaTheta*dR/self.Speed-(self.OmegaTheta*dR/self.Speed)%(2*pi))
        f1Map=map(f1,dRvalues)
        f1List=list(f1Map)
        
        f2 = lambda dR: dw(dR)/(2*self.Speed)*(Ri+Rj(dR))-pi*dw(dR)/self.OmegaTheta        
        f2Map=map(f2,dRvalues)
        f2List=list(f2Map)
        
        f= lambda dR: f1(dR)+f2(dR)
        fMap=map(f,dRvalues)
        fList=list(fMap)
        
        phase= lambda dR: arct(dR)+f(dR)
        phaseMap=map(phase,dRvalues)
        phaseList=list(phaseMap)
        
        plt.plot(dRvalues,arctList,'y')
        plt.title('arctan(b/a)')
        plt.show()
        plt.plot(dRvalues,f1List,'r',lw=1)
        plt.title('f1')
        plt.show()
        plt.plot(dRvalues,f2List,'g',lw=1)
        plt.title('f2')
        plt.show()
        plt.plot(dRvalues,phaseList,'b',lw=2)
        plt.title('Total Phase-Offset as a function of dR (with R fixed)')
        plt.show()
        print('We see here the three main terms of the phase-offset. The terms f2%2pi and f1%2pi are quite similar. Let\'s plot both together, modulo-2pi.')
        
        Minusf2= lambda dR: -(f2(dR))%(2*pi) 
        Minusf2Map=map(Minusf2,dRvalues)
        Minusf2List=list(Minusf2Map)
        f1modulo= lambda dR: f1(dR)%(2*pi)
        f1moduloMap=map(f1modulo,dRvalues)
        f1moduloList=list(f1moduloMap)
        
        sumf1f2= lambda dR: f1modulo(dR)-Minusf2(dR)
        sumf1f2Map=map(sumf1f2,dRvalues)
        sumf1f2List=list(sumf1f2Map)
        
        plt.plot(dRvalues,Minusf2List,'g', label='- f2',lw=0.8)
        plt.plot(dRvalues,f1moduloList,'r',label='f1',lw=0.8)
        plt.title('Comparing f1 and -f2 (Modulo-2pi)')
        plt.legend(loc='lower left', title='Legend')
        plt.show()
        plt.plot(dRvalues,sumf1f2List,'k',label='Sum f1+f2')
        plt.title('Term f1+f2=f(0) (Modulo-2pi)')
        plt.show()

        print('Above: Here we compare the f1 term to the -f2 term. They are quite close to each other, so f1+f2 will give us a small offset, still relevant.')
        print('Below: The black line in this second plot represents the Phase-Offset included by f1+f2=f(0) term.')
        
        
        ArctModulo= lambda dR: arct(dR)%(2*pi)
        ArctModuloMap=map(ArctModulo,dRvalues)
        ArctModuloList=list(ArctModuloMap)

        plt.plot(dRvalues,ArctModuloList,'y',label='Arctan Modulo 2pi')
        plt.plot(dRvalues,sumf1f2List,'k',label='f(0)')
        plt.title('f(0) and Arctan-Term (Modulo-2pi)')
        plt.legend(loc='upper left', title='Legend')
        plt.show()
        
        print('We see that the Total Phase cannot be approximated by the arctan term alone. We need to consider all the three terms, and simplify them, if we are able to.')
        print('We will plot now the total phase without step: making negative the phase for negative values of dR, substracting 2pi to it.')
        
        
        NegativedRValues=list(arange(-0.8*R,-0.01,0.01))
        PositivedRValues=list(arange(0,+0.6*R,0.01))
        
        TotalPhase= lambda dR: ((f1modulo(dR)-Minusf2(dR))+ArctModulo(dR))%(2*pi)
        TotalPhaseMap=map(TotalPhase,dRvalues)
        TotalPhaseList=list(TotalPhaseMap)
        
        NegativeTotalPhase= lambda dR: TotalPhase(dR)-(2*pi)
        NegativeTotalPhaseMap=map(NegativeTotalPhase,NegativedRValues)
        NegativeTotalPhaseList=list(NegativeTotalPhaseMap)
        PositiveTotalPhase= lambda dR: TotalPhase(dR)
        PositiveTotalPhaseMap=map(PositiveTotalPhase,PositivedRValues)
        PositiveTotalPhaseList=list(PositiveTotalPhaseMap)
        
        SumdRvalues= NegativedRValues + PositivedRValues
        SumTotalPhase= NegativeTotalPhaseList+PositiveTotalPhaseList
        
        plt.plot(SumdRvalues,SumTotalPhase)
        plt.title('Total Phase-Offset - dR dependence')
        plt.show()
        
        print('Above we can see the phase offset dependent on the difference in size between fields dR=Ri-Rj, given the size R=Ri of one of the fields.')
        
       
        
        
        
        
        
        