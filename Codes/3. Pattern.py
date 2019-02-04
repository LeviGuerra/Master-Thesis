# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:56:40 2018

@author: Multivac
"""

import pylab
import random
from numpy import *
from math import *
import numpy
import math
import numpy as np
import scipy 
import scipy.fftpack


import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.spatial.distance import pdist, squareform
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("white")


from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
from scipy.fftpack import fftshift
from scipy.fftpack import ifftshift
# numpy.set_printoptions(threshold=4)                                         # Change of settings: print entire array
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})    # Change of settings: print only 3 decimals


class Pattern:    
    
    def __init__(self):
        
        #Parameters of the Window
        self.m=1 
        self.p=0.002 
        self.q=-0.005
        self.W0=0.1
        self.step=1.5
        
        #Parameters of animal movement and cells - RECALL THAT ALL CELLS ARE IDENTICAL!!!
        self.Time = 400                                                         # Time in seconds of the Learning Process
        
        self.Speed=25
        self.OmegaTheta=2*pi*8                                                  # Angular frequency of the theta oscillation (experimental value of 8Hz)
        self.R=200                                                               # Radius of the place field 1
        self.P=50                                                               # Peak value of Place Field
        self.DeltaOmega=pi*self.Speed/self.R                                    # Frecuency difference between cell 1 and theta
        self.w=self.DeltaOmega+self.OmegaTheta                                  # Angular frequency of the firing rate of cell 1
        self.s=self.R/(sqrt(2*np.log(10)))                                      # Std.Dev. of Cell 1 - It is fixed by the size of the place field
        
        #Other definitions
        self.sbar=self.s                                                        # sbar is equal to s for identical neurons
        self.wbar=self.w                                                        # wbar is equal to w for identical neurons
        self.Dw=0                                                               # Place Fields are equal, Dw=0
        self.DR=0
        
        #Parameters of the Kernel
        G0 = lambda r: sqrt(pi)/4*self.s/self.Speed*self.P**2*exp(-r**2/(4*self.sbar**2))
        K = self.W0*(2*pi*self.m**2*self.p**2)**(-1/2)
        self.G = lambda r: K*G0(r)
        self.G1 = 1/2*exp(-self.s**2*self.Dw**2/(4*self.Speed**2))
        
        self.a = ((self.Speed*self.m*self.p)**2+2*self.sbar**2)/(4*(self.sbar*self.m*self.p)**2)
        self.b = lambda r: self.Speed*r/(2*self.sbar**2)
        
        #Kernel
        coefalpha = lambda r: sqrt(pi/self.a) * exp(self.b(r)**2/(4*self.a))
        coefbeta = lambda r: sqrt(pi/self.a) * exp(self.b(r)**2/(4*self.a)) * exp(-self.wbar**2/(4*self.a))
        
        self.Extreme=input('Extreme case? If so, write "Antisymmetric" or "Symmetric". Otherwise, write "N": ')
        
        if self.Extreme=='N':
            kc = lambda r: 1 - self.b(r)/(2*self.a*self.q) - (self.b(r)**2-self.wbar**2+2*self.a)/(4*self.a**2*self.p**2)
            ks = lambda r: self.wbar/(2*self.a*self.q) + self.b(r)*self.wbar/(2*self.a**2*self.p**2)
        
            self.alpha = lambda r: coefalpha(r) * ( 1 - self.b(r)/(2*self.a*self.q) - (2*self.a+self.b(r)**2)/(4*self.a**2*self.p**2) )
            self.betasimp = lambda r: coefbeta(r) * ( kc(r)*cos(pi/self.R*r) - ks(r)*sin(pi/self.R*r) )
            self.Kernel = lambda r: self.G(r) * ( self.alpha(r) + self.G1*self.betasimp(r) )
        
        elif self.Extreme=='Symmetric':
            kc = lambda r: 1 - (self.b(r)**2-self.wbar**2+2*self.a)/(4*self.a**2*self.p**2)
            ks = lambda r: self.b(r)*self.wbar/(2*self.a**2*self.p**2)
        
            self.alpha = lambda r: coefalpha(r) * ( 1 - (2*self.a+self.b(r)**2)/(4*self.a**2*self.p**2) )
            self.betasimp = lambda r: coefbeta(r) * ( kc(r)*cos(pi/self.R*r) - ks(r)*sin(pi/self.R*r) )
            self.Kernel = lambda r: self.G(r) * ( self.alpha(r) + self.G1*self.betasimp(r) )
        
        elif self.Extreme=='Antisymmetric':
            kc = lambda r: - self.b(r)/(2*self.a*self.q)
            ks = lambda r: self.wbar/(2*self.a*self.q) 
        
            self.alpha = lambda r: coefalpha(r) * ( - self.b(r)/(2*self.a*self.q) )
            self.betasimp = lambda r: coefbeta(r) * ( kc(r)*cos(pi/self.R*r) - ks(r)*sin(pi/self.R*r) )
            self.Kernel = lambda r: self.G(r) * ( self.alpha(r) + self.G1*self.betasimp(r) )
        
        #Place Fields Distribution 
        self.PFspacing = 2                                                      # Spacing between consecutive Place Fields in cm
        self.Length = 600                                                       # Length of the Path
        self.PFpositions = arange(self.PFspacing,self.Length+self.PFspacing,self.PFspacing)     
        
        #Learning - Matrix form
        self.Gamma = np.vectorize(self.Kernel)
        
        n = len(self.PFpositions)                                               # Size of the positions array - The Kernel in the matrix form will be nxn
        Mdist = np.zeros((n,n))
        
        Periodicity= input('Spatial periodicity (Y) or not (N)?: ') 
        
        if Periodicity=='Y':
            for i in range(0,n):
                for j in range(0,n):
                    
                    d1=self.PFpositions[j]-self.PFpositions[i]
                       
                    if self.PFpositions[i]>self.PFpositions[j]:
                        d2=self.Length-abs((self.PFpositions[j]-self.PFpositions[i]))
                    else:
                        d2=-(self.Length-abs((self.PFpositions[j]-self.PFpositions[i])))
                    
                    if abs(d1)<=abs(d2):
                        Mdist[i][j] = d1
                    else:
                        Mdist[i][j] = d2
            self.Mdist=Mdist
        
        elif Periodicity=='N':
            for i in range(0,n):
                for j in range(0,n):
                    d1=self.PFpositions[j]-self.PFpositions[i]
                    Mdist[i][j] = d1
            self.Mdist=Mdist
        else:
            print('Please, introduce only N or Y')
        


    def PlaceFieldsDistribution(self):
        
        print()
        print('The place fields positions are',self.PFpositions,'listed in a',type(self.PFpositions))
        print('The number of place fields is', len(self.PFpositions))
        
        # Plot segment of length + Create for loop to plot all the place field circles
        
        segment=arange(0,self.Length+1,1)
        
        plt.figure(figsize=(20,15))
        plt.plot(segment,zeros(size(segment)),lw=3)
        for i in self.PFpositions:
            field=plt.Circle((i, 0), self.R, color='r', fill=False, lw=1.5)
            plt.gcf().gca().add_artist(field)
        
        plt.tick_params(labelsize=50)
        plt.xlim(-30-3,self.Length+35)   
        plt.title('Place Fields Distribution over Linear Track (cm)', fontsize=50)
        plt.ylim(-self.R*2,self.R*2)
        plt.show()
        
        print()
        print()
        print('The distances matrix looks like:')
        
        plt.figure(figsize=(15,15))
        plt.imshow(self.Mdist,cmap='viridis')                                   # viridis, inferno, magma
        cb=plt.colorbar()
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=65)
        cb.set_label('(cm)', rotation=270, fontsize=65)
        plt.tick_params(labelsize=65)
        plt.title('Matrix of Distances',fontsize=65)
        plt.xlabel('Place Cell Index',fontsize=65)
        plt.ylabel('Place Cell Index',fontsize=65)
        plt.grid()
        
        plt.show()
        
        print()
        print('To test if the matrix of distances Mdist is properly defined, meaning, whether or not it is antisymmetric (it should be, from the definition of distances), we also calculate the symmetric part Mdist+Mdist^T, which should give back a matrix of zeros:')
        print()
        print((self.Mdist+self.Mdist.T)/2)
        print('The sum of all this matrix elements is:', ((self.Mdist+self.Mdist.T)/2).sum())
        print()
        print('If we plot the matrix:')
        plt.imshow(self.Mdist+self.Mdist.T,cmap='jet')
        plt.colorbar()
        plt.show()
        
        #Better in Kernel!!
# =============================================================================
# 
#     def Window(self): 
#         
#         print()
#         print('The window parameters \'m\' (variance param.) is:',self.m)
#         print('The window parameters \'p\' (quadratic param.) is:',self.p)
#         print('The window parameters \'q\' (linear param.) is:',self.q)
#         print('The window parameters \'W0\' (window factor) is:',self.W0)
#         print()
#         
#         pol = lambda s: ((s/self.q))  
#         win = lambda s: self.W0*(2*pi*self.m**2*self.p**2)**(-1/2)*pol(s)*exp(-(s**2)/(2*self.p**2*self.m**2))
# 
#         interval = list(arange(-0.05,0.05,0.0001))
#         result= map(win, interval)
#         W=list(result)
#         
#         plt.plot(interval,W)
#         plt.xlabel('$τ$ (s)', fontsize=20)
#         plt.ylabel('W($τ$)', fontsize=20)
#         plt.show()
#         
#         print(win(-0.0025))
#         print(win(0.0025))
# =============================================================================
        
    
    
    def KernelPlot(self):
        
        rvalues = list(arange(-100,100,0.01))
        KernelMap = map(self.Kernel, rvalues)
        KernelList = list(KernelMap)
        
        plt.plot(rvalues,KernelList,'r',lw=1)   
        plt.title('Kernel',fontsize=18)
        plt.show()
        
        print('To test symmetry we check the values of the Kernel for:')
        print('Kernel(0)=',self.Kernel(0))
        print('Kernel(0.1)=',self.Kernel(0.1))
        print('Kernel(-0.1)=',self.Kernel(-0.1))
        print('Kernel(1)=',self.Kernel(1))
        print('Kernel(-1)=',self.Kernel(-1))
        print('Kernel(8)=',self.Kernel(8))
        print('Kernel(-8)=',self.Kernel(-8))
        print('Kernel(50)=',self.Kernel(50))
        print('Kernel(-50)=',self.Kernel(-50))
    
    
    def KernelValues(self,Distance):
        
        print()
        print('Distance =',Distance)
        print()
        print('The value of the parameter a is:', self.a)
        print('The value of the parameter b is:', self.b(Distance))
        print()
        print('The value of the factor G(r) for the given distance is:', self.G(Distance))
        print('The value of the term alpha for the given distance is:', self.alpha(Distance))
        print('The value of the term beta for the given distance is:', self.betasimp(Distance))
        print()
        print('The value of the Kernel for the given distance is:', self.Kernel(Distance))
        
        
        
    def FourierAnalysis(self):
                
#==============================================================================
#         dt=0.001
#         rvalues=arange(-1000,1000,dt)             #We enlarge the interval, since the bigger it is, the higher resolution we will have
#         KernelMap = map(self.Kernel, rvalues)
#         KernelList = list(KernelMap)
#         plt.figure(figsize=(20,5))
#         plt.plot(rvalues,KernelList,'r',lw=1) 
#         plt.show()
#         
#         #Frequency Sampling
#         N=len(rvalues)
#         ind=arange(1,int(N/2+1))     # Sampling of real space
#         allfreqs=fftfreq(N)*(2*pi)/dt          # Sampling of frequency space
#         realfreqs=fftfreq(N)[ind]*(2*pi)/dt    # Frequencies in which we are interested (we omit redundancies due to complex conjugates)
#         
#         #We now put the fourier transform coefficients
#         FFTKernel = scipy.fftpack.fft(KernelList)
#         plt.figure(figsize=(20,5))
#         plt.plot(allfreqs, FFTKernel)
#         plt.xlim(-0.2,0.2)
#         plt.show()
#                 
#         #Now, the Imaginary Part
#         FFTKernelImg = scipy.fftpack.fft(KernelList).imag
#         plt.figure(figsize=(20,5))
#         plt.plot(allfreqs,FFTKernelImg)
#         plt.xlim(-0.2,0.2)
#         plt.show()
# 
#         #Spectral density
#         FFTKernel = scipy.fftpack.fft(KernelList)
#         sd= abs(FFTKernel[ind])**2 + abs(FFTKernel[-ind])**2
#         plt.figure(figsize=(20,5))
#         plt.plot(realfreqs,sd)
#         plt.xlim(-0.2,0.2)
#         
#         #We choose the maximum frequency
#         maxvalue=np.where(sd==max(sd))
#         print(maxvalue)
#         maxfreq=realfreqs[maxvalue]
#         print(maxfreq)
#         plt.axvline(maxfreq,c='k')
#         plt.show()
#         
#         #Now we correct to the real positions of the selected frequencies in the entire array
#         print('But this is the frequency considering the interval of relevant frequencies, (1,N/2+1). We have to correct it!')
#         print(ind[maxvalue])
#         FinalFreq=realfreqs[ind[maxvalue]]
#         print(FinalFreq)
#         
#         #Now we re-built the Fourier Transform
#         FFTFreq=zeros(len(FFTKernel))
#         FFTFreq[ind[maxvalue]]=FFTKernel[ind[maxvalue]]
#         FFTFreq[-ind[maxvalue]]=FFTKernel[-ind[maxvalue]]
#         InvFFTFreq=ifft(FFTFreq)
#         
#         print('The real component of the maximum frequency is:')
#         print(FFTKernel[ind[maxvalue]])
#         print('And the imaginary component will be:')
#         print(FFTKernel[ind[maxvalue]].imag)
#         
#         exponent = lambda r: 0.01*cos(0.10995574*r)
#         exponentMap = map(exponent,rvalues)
#         exponentList = list(exponentMap)
#         
#         plt.figure(figsize=(20,10))
#         plt.plot(rvalues,InvFFTFreq,'r--',lw=3)
#         plt.plot(rvalues,KernelList,'k',lw=3)
#         plt.plot(rvalues,exponentList)        
#         plt.show()
#==============================================================================
        
        print('We will calculate the measure of overlap between cosines of varying frequency and the kernel, and find the frequency for a maximum overlap. In other words, we will find the leading frequency of the symmetric part.')
        
        numpy.set_printoptions(threshold=4)
        
        N=arange(-150,150,0.1)
        FourierTransf=[]
        FTKernel=0
        dx=0.0001
        kint=arange(0,0.05,dx)
        for k in kint:
            print(k)
            for n in N:    
                FTKernel += np.cos(2*pi*k*n)*self.Kernel(n)*dx
            FourierTransf.append(FTKernel)  
            FTKernel=0

        maxk=where(FourierTransf==max(FourierTransf))
        kmax=kint[maxk]
        print(kmax)
        print(max(FourierTransf))
        
        f=plt.figure(figsize=(20,15))
        plt.plot(kint,FourierTransf,c='k',lw=3)
        plt.axvline(kint[maxk],c='r', lw=5)
        plt.tick_params(labelsize=50)
        plt.xlabel('$k \ (cm^{-1})$', fontsize=55)
        plt.ylabel('$\hat{Γ}_{sym}$', fontsize=55)
        f.savefig("R=30.pdf", bbox_inches='tight')
        plt.show()
            
        print('Now we will find the imaginary part of that maximum k-value, i.e. the antisymmetric part')
        
        FTKernel=0
        for n in N:    
            FTKernel += np.sin(2*pi*kmax*n)*self.Kernel(n)*dx
        
        print('The value of the imaginary part is',FTKernel)
        
        N=arange(-150,150,0.1)
        FourierTransf=[]
        FTKernel=0
        dx=0.0001
        kint=arange(0,0.05,dx)
        for k in kint:
            print(k)
            for n in N:    
                FTKernel += np.sin(2*pi*k*n)*self.Kernel(n)*dx
            FourierTransf.append(FTKernel)  
            FTKernel=0
            
        plt.figure(figsize=(20,15))
        plt.plot(kint,FourierTransf,c='k',lw=3)
        plt.axvline(kmax,c='r', lw=5)
        plt.tick_params(labelsize=50)
        plt.xlabel('$k \ (cm^{-1})$', fontsize=55)
        plt.ylabel('$\hat{Γ}_{anti}$', fontsize=55)
        plt.show()
        
                
    def FourierAnalysisP(self):
               
        G0 = lambda r: sqrt(pi)/4*self.s/self.Speed*self.P**2*exp(-r**2/(4*self.sbar**2))
        K = lambda p: self.W0*(2*pi*self.m**2*p**2)**(-1/2)
        G = lambda r,p: K(p)*G0(r)
        G1 = 1/2*exp(-self.s**2*self.Dw**2/(4*self.Speed**2))
        
        a = lambda p: ((self.Speed*self.m*p)**2+2*self.sbar**2)/(4*(self.sbar*self.m*p)**2)
        b = lambda r: self.Speed*r/(2*self.sbar**2)
        
        #Kernel
        coefalpha = lambda r, p: sqrt(pi/a(p)) * exp(b(r)**2/(4*a(p)))
        coefbeta = lambda r, p: sqrt(pi/a(p)) * exp(b(r)**2/(4*a(p))) * exp(-self.wbar**2/(4*a(p)))
        
        
        kc = lambda r, p: 1 - (b(r)**2-self.wbar**2+2*a(p))/(4*a(p)**2*p**2)
        ks = lambda r, p: b(r)*self.wbar/(2*a(p)**2*p**2)
        
        alpha = lambda r,p: coefalpha(r,p) * ( 1 - (2*a(p)+b(r)**2)/(4*a(p)**2*p**2) )
        betasimp = lambda r,p: coefbeta(r,p) * ( kc(r,p)*cos(pi/self.R*r) - ks(r,p)*sin(pi/self.R*r) )
        Kernel = lambda r,p: G(r,p) * ( alpha(r,p) + G1*betasimp(r,p) )
        
        numpy.set_printoptions(threshold=4)
        pint1=arange(0.00001,0.02,0.05)
        pint2=arange(0.02,0.085,0.0005)
       
        pint=concatenate((pint1,pint2))

        MaxList=[]
        
        for p in pint:
            N=arange(-150,150,0.1)
            FourierTransf=[]
            FTKernel=0
            dx=0.0001
            kint=arange(0,0.02,dx)
            for k in kint:
                for n in N:    
                    FTKernel += np.cos(2*pi*k*n)*Kernel(n,p)*dx
                FourierTransf.append(FTKernel)
                FTKernel=0
        
            maxk=where(FourierTransf==max(FourierTransf))
            kmax=kint[maxk]
            MaxList.append(kmax)
            print(MaxList)
        
        print('')
        print(MaxList)
        
        plist=list(pint)
        MaxList=list(MaxList)
        plt.figure(figsize=(20,15))
        plt.plot(pint,MaxList,'r',lw=5)
        plt.tick_params(labelsize=55)
        plt.xlabel('$ρ \ (s)$', fontsize=55)
        plt.ylabel('$k_{max} \ (cm^{-1})$', fontsize=55)
        plt.show()        
                
        
    def FourierAnalysisQ(self):
        
        G0 = lambda r: sqrt(pi)/4*self.s/self.Speed*self.P**2*exp(-r**2/(4*self.sbar**2))
        K = self.W0*(2*pi*self.m**2*self.p**2)**(-1/2)
        G = lambda r: K*G0(r)
        G1 = 1/2*exp(-self.s**2*self.Dw**2/(4*self.Speed**2))
        
        a = ((self.Speed*self.m*self.p)**2+2*self.sbar**2)/(4*(self.sbar*self.m*self.p)**2)
        b = lambda r: self.Speed*r/(2*self.sbar**2)
        
        #Kernel
        coefalpha = lambda r: sqrt(pi/a) * exp(b(r)**2/(4*a))
        coefbeta = lambda r: sqrt(pi/a) * exp(b(r)**2/(4*a)) * exp(-self.wbar**2/(4*a))
        
        kc = lambda r, q: 1 - b(r)/(2*a*q) - (b(r)**2-self.wbar**2+2*a)/(4*a**2*self.p**2)
        ks = lambda r, q: self.wbar/(2*a*q) + b(r)*self.wbar/(2*a**2*self.p**2)
        
        alpha = lambda r,q: coefalpha(r) * ( 1 - b(r)/(2*a*q) - (2*a+b(r)**2)/(4*a**2*self.p**2) )
        betasimp = lambda r, q: coefbeta(r) * ( kc(r,q)*cos(pi/self.R*r) - ks(r,q)*sin(pi/self.R*r) )
        Kernel = lambda r, q: G(r) * ( alpha(r,q) + G1*betasimp(r,q) )
        
        pint=arange(0.00001,0.08,0.005)

        MaxList=[]
        
        for p in pint:
            N=arange(-100,100,0.005)
            FourierTransf=[]
            FTKernel=0
            dx=0.001
            kint=arange(0,0.02,dx)
            for k in kint:
                for n in N:    
                    FTKernel += np.cos(2*pi*k*n)*Kernel(n,p)*dx
                FourierTransf.append(FTKernel)
                FTKernel=0
                
            
            maxk=where(FourierTransf==max(FourierTransf))
            kmax=kint[maxk]
            MaxList.append(kmax)
            print(MaxList)
        
        print('')
        print(MaxList)
        
        plist=list(pint)
        MaxList=list(MaxList)
        plt.figure(figsize=(20,15))
        plt.plot(pint,MaxList,'r',lw=5)
        plt.tick_params(labelsize=55)
        plt.xlabel('$q \ (s)$', fontsize=55)
        plt.ylabel('$k_{max} \ (cm^{-1})$', fontsize=55)
        plt.show()        
               
        
        
        
    def Pattern(self):
        
        Inhibition = input('Inhibition Network? Yes=Y, No=N: ')

        print()
        print()
        print('The kernel we are using is:')
        rvalues = list(arange(-100,100,0.01))
        print(len(rvalues))
        KernelMap = map(self.Kernel, rvalues)
        KernelList = list(KernelMap)
        
        #Plot Kernel
        plt.plot(rvalues,KernelList,'r',lw=1)   
        plt.ylabel('Kernel',fontsize=30)
        plt.xlabel('x (cm)', fontsize=30)
        plt.show()
        
        MatrixKernel = self.Gamma(self.Mdist)
        
        print()
        print('The distance between two consecutive place fields is',self.PFspacing,'cm. We have in total',len(self.PFpositions),'place fields.')
        print()
        print('And the respective Kernel in the matrix form will be:')
        
        #Plot Kernel Matrix
        plt.figure(figsize=(6,6))
        plt.imshow(MatrixKernel,cmap='jet')                   # viridis, inferno, magma
        plt.tick_params(labelsize=23)     
        cb=plt.colorbar()
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=23)
        plt.title('$\Gamma_{matrix}$', fontsize=30)
        plt.show()
        #Further details on the Kernel Matrix
        
        print()
        print('We plot the Symmetric part of the Matrix as well:')
        plt.imshow((MatrixKernel+MatrixKernel.T)/2,cmap='jet')
        plt.tick_params(labelsize=23)
        cb=plt.colorbar()
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=23)
        plt.title('Symm. Part', fontsize=25)
        plt.show()
        
        print()
        print('Also we plot the Antisymmetric part of the Matrix as well:')
        plt.imshow((MatrixKernel-MatrixKernel.T)/2,cmap='jet')
        plt.tick_params(labelsize=23)
        cb=plt.colorbar()
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=23)
        plt.title('Anti. Part', fontsize=25)
        plt.show()
        
              
        # Now we write the time evolution of the weights 
        
        if Inhibition=='N':
            
            Average = input('Average weights over time? Yes=Y, No=N: ')
   
            print()
            print()
            print('We generate random initial weights:')
            
            J0 = 1 + 0.1*random.rand(len(self.PFpositions))                     # We generate a vector J of random synaptic weights
            #J0 = J0/sqrt(sum(J0**2))                                           # We normalize them

            plt.figure(figsize=(20,5))
            plt.scatter(self.PFpositions,J0)
            plt.plot(self.PFpositions,J0)
            plt.tick_params(labelsize=30)
            plt.xlim(0,+self.Length)   
            plt.ylim(0)
            plt.xlabel('Place Cells (blue dots) over Linear Track (cm)', fontsize=30)
            plt.ylabel('Initial Synaptic Weight', fontsize=30)
            plt.show()
            

            t=0
            dt=0.001
            J=J0
            Jvec = [J0]
            
            #String of weights
            X=[J[0]]
            Y=[J[25]]
            Z=[J[50]]
            
            # String of averaged weights
            A=[]
            B=[]
            C=[]
            
            
            if Average == 'N':            #No Average
                while t<self.Time:
                    t=dt+t
                    J = J + dt * ( dot(MatrixKernel,J) - J**2 ) #   - dot(J,J) )
                    J[J<0]=0                                                    # Positivity constrain
                    #J[J>1]=1
                    
                    X.append(J[0])                                              # The intention is to plot this X against time, and see if the weights are stable on time            
                    Y.append(J[25])
                    Z.append(J[50])                                             # The intention is to plot this X against time, and see if the weights are stable on time

                print('After learning, the pattern looks like this:')

                plt.figure(figsize=(20,5))
                plt.scatter(self.PFpositions,J)
                plt.plot(self.PFpositions,J)
                plt.tick_params(labelsize=30)
                plt.xlim(0,+self.Length)
                #plt.ylim(0)
                plt.xlabel('Place Cells (blue dots) over Linear Track (cm)', fontsize=30)
                plt.ylabel('Final Synaptic Weight', fontsize=30)
                plt.show()
                
                        
                print()
                print()
                print('We want to see the time evolution of several individual cell weights along the path. We selected 2 of them:')
                
                lx=int(len(X)/8)
                ly=int(len(Y)/8)
                lz=int(len(Z)/8)
                X=X[2*lx:3*lx]
                Y=Y[2*ly:3*ly]
                Z=Z[2*lz:3*lz]
                timex = arange(2*lx*dt,3*lx*dt,dt)                               #Appropiate spacing for the x-axis to be Time
                
               # timex = arange(0,len(X)*dt,dt)
                
                #Temporal Evolution
                plt.figure(figsize=(20,5))
                #Neuron no. 0
                plt.plot(timex,X,label='Cell x=0',lw=2, color='darkgreen')
                #Neuron no. 25
                plt.plot(timex,Y,label='Cell x=50',lw=2, color='darkred')
                #Neuron no. 50
                plt.plot(timex,Z,label='Cell x=100',lw=2, color='darkorange')
                plt.tick_params(labelsize=30)
                plt.xlabel('Time (s)', fontsize=30)
                plt.ylabel('Synaptic Weight', fontsize=30)       
                plt.legend(frameon=True, prop={'size': 30})
                plt.show()
            
            if Average == 'Y':            #Averaged
                while t<self.Time:
                    
                    step=self.step
                    
                    t=t+dt
                    
                    J = J + dt * ( dot(MatrixKernel,J) - J**2 )
                    Jvec.append(J)
                    J[J<0]=0                                                    # Positivity constrain
                    
                    X.append(J[0])
                    Y.append(J[25])
                    Z.append(J[50])
                    
                    
                    if t % step <= dt :                                         # Loop to store J after a certain number of steps - CELL 0
                        
                        Javg = np.average(np.array(Jvec), axis=0)
        
                        A.append(Javg[0])                                       # The intention is to plot this X against time, and see if the weights are stable on time    
                        B.append(Javg[25])                                      # The intention is to plot this X against time, and see if the weights are stable on time
                        C.append(Javg[50])
                        
                        J=Javg
                        Jvec=[]

                print('After learning, the pattern looks like this:')
        
                plt.figure(figsize=(20,5))
                plt.scatter(self.PFpositions,J)
                plt.plot(self.PFpositions,J)
                plt.tick_params(labelsize=30)
                plt.xlim(0,+self.Length)
                #plt.ylim(0)
                plt.xlabel('Place Cells (blue dots) over Linear Track (cm)', fontsize=30)
                plt.ylabel('Averaged Synaptic Weights', fontsize=30)
                plt.show()
                        
                print()
                print()
                print('We want to see the time evolution of several individual cell weights along the path. We selected 2 of them:')
                
                la=int(len(A)/8)
                lb=int(len(B)/8)
                lc=int(len(C)/8)
                A=A[2*la:3*la]
                B=B[2*la:3*lb]
                C=C[2*la:3*lc]
                               
                stepx = arange(2*la*step,3*la*step,step)                           #Appropiate spacing for the x-axis to be Time
                
                #Temporal Evolution
                plt.figure(figsize=(20,5))
                plt.plot(stepx,A,label='Cell x=0 avg',lw=3, color='darkgreen')
                plt.tick_params(labelsize=30)
                plt.xlabel('Time (s)', fontsize=30)
                plt.ylabel('Synaptic Weights', fontsize=30)       
                plt.plot(stepx,B,label='Cell x=50 avg',lw=3, color='darkred')
                plt.plot(stepx,C,label='Cell x=100 avg',lw=3, color='darkorange')
                plt.legend(frameon=True, prop={'size': 30})
                plt.show()
                
                
                
        elif Inhibition=='Y':
            
            dP=len(self.PFpositions)                                            # Number of Place Cells 
            dG=100                                                              # Number of Grid Cells that will inhibit each other     
            
            J=1/dP*( np.ones((dG,dP)) +  0.1*np.random.randn(dG,dP))            # Initial Random Weight Matrix - From Place Cells to Grid Cell
            #M=-1/dG*( np.ones((dG,dG)) + 0.1*np.random.randn(dG,dG) )          # Inhibition Matrix - No Sparsed
            
            M0=scipy.sparse.random(dG,dG,.1)     # Inhibition Matrix - Sparsed (all 3 lines)
            M0=M0!=0
            M=-10*(M0 + 0.1*np.random.randn(dG,dG))
            
            plt.figure(figsize=(20,5))
            plt.scatter(self.PFpositions,J[0])
            plt.plot(self.PFpositions,J[0])
            plt.xlim(0,+self.Length)   
            plt.ylim(0)
            plt.title('Initial Weight Distribution (Randomized)')
            plt.xlabel('Place Cells (blue dots) over Linear Track (cm)')
            plt.ylabel('Synaptic Weight')
            plt.show()
            
            
            
            J=np.matrix(J)
            M=np.matrix(M)
            
            MI=(np.matrix(np.identity(dG))-M)
            MI=MI.I
            
            print('')
            print('Now we plot the matrix MI:')
            plt.imshow(MI,cmap='viridis')                             # viridis, inferno, magma
            plt.colorbar()
            plt.show()
            
            t=0
            dt=0.001
            
            X=[J[0,0]]
            Y=[J[0,50]]
            
            #if Average=='N':
            while t<self.Time:
                
                t=dt+t
                J = J + dt * ( MI*J*MatrixKernel - np.matrix((J.A)**2 ) )   # .A is just transforming matrix into array, to be able to perform **2
                
                #J=np.matrix((J.A)*(J.A>0))
                J[J<0]=0                                                    # Positivity constrain
                #J[J>1]=0
                
                X.append(J[0,0])                                            # The intention is to plot this X against time, and see if the weights are stable on time
                Y.append(J[0,50])                                           # The intention is to plot this X against time, and see if the weights are stable on time
            
            J=array(J)
            
            plt.figure(figsize=(20,5))
            plt.scatter(self.PFpositions,J[0])
            plt.plot(self.PFpositions,J[0])
            plt.xlim(0,+self.Length)
            plt.tick_params(labelsize=30)
            #plt.ylim(0)
            plt.xlabel('Place Cells (blue dots) over Linear Track (cm)', fontsize=30)
            plt.ylabel('Final Synaptic Weight', fontsize=30)
            plt.show()
            
            plt.figure(figsize=(20,5))
            plt.scatter(self.PFpositions,J[23])
            plt.plot(self.PFpositions,J[23])
            plt.xlim(0,+self.Length)
            plt.tick_params(labelsize=30)
            #plt.ylim(0)
            plt.xlabel('Place Cells (blue dots) over Linear Track (cm)', fontsize=30)
            plt.ylabel('Final Synaptic Weight', fontsize=30)
            plt.show()
            
            plt.figure(figsize=(20,5))
            plt.scatter(self.PFpositions,J[57])
            plt.plot(self.PFpositions,J[57])
            plt.xlim(0,+self.Length)
            plt.tick_params(labelsize=30)
            #plt.ylim(0)
            plt.xlabel('Place Cells (blue dots) over Linear Track (cm)', fontsize=30)
            plt.ylabel('Final Synaptic Weight', fontsize=30)
            plt.show()
            
            plt.figure(figsize=(20,5))
            plt.scatter(self.PFpositions,J[81])
            plt.plot(self.PFpositions,J[81])
            plt.xlim(0,+self.Length)
            plt.tick_params(labelsize=30)
            #plt.ylim(0)
            plt.xlabel('Place Cells (blue dots) over Linear Track (cm)', fontsize=30)
            plt.ylabel('Final Synaptic Weight', fontsize=30)
            plt.show()
            
            timex = arange(0,len(X)*dt,dt)                               #Appropiate spacing for the x-axis to be Time
            
            #Neuron 0
            plt.figure(figsize=(20,5))
            plt.plot(timex,X,label='Cell 0 - Avg')
            plt.title('Temporal Evolution of the Synaptic Weights of Cells 0')
            #plt.ylim(0)
            plt.xlabel('Time (s)')
            plt.ylabel('Synaptic Weight')
            plt.legend(frameon=True)
            plt.show()
            
            #Neuron 50
            plt.figure(figsize=(20,5))
            plt.plot(timex,Y,label='Cell 50 - Avg')
            plt.title('Temporal Evolution of the Synaptic Weights of Cells 50')
            #plt.ylim(0)
            plt.xlabel('Time (s)')
            plt.ylabel('Synaptic Weight')
            plt.legend(frameon=True)
            plt.show()
            
#==============================================================================
#             Y.append(J[13,50])
#             
#             plt.figure(figsize=(20,5))
#             plt.scatter(self.PFpositions,J[13])
#             plt.plot(self.PFpositions,J[13])
#             plt.xlim(0,+self.Length)
#             #plt.ylim(0)
#             plt.title('Final Pattern')
#             plt.xlabel('Place Cells over Linear Path (Blue Dots)')
#             plt.ylabel
#             
#             #Neuron 50
#             plt.figure(figsize=(20,5))
#             plt.plot(timex,Y,label='Cell 50 - Avg')
#             plt.title('Temporal Evolution of the Synaptic Weights of Cells 50')
#             #plt.ylim(0)
#             plt.xlabel('Time (s)')
#             plt.ylabel('Synaptic Weight')
#             plt.legend(frameon=True)
#             plt.show()
#             
#             Y.append(J[24,50])
#             
#             plt.figure(figsize=(20,5))
#             plt.scatter(self.PFpositions,J[24])
#             plt.plot(self.PFpositions,J[24])
#             plt.xlim(0,+self.Length)
#             #plt.ylim(0)
#             plt.title('Final Pattern')
#             plt.xlabel('Place Cells over Linear Path (Blue Dots)')
#             plt.ylabel
#             
#             #Neuron 50
#             plt.figure(figsize=(20,5))
#             plt.plot(timex,Y,label='Cell 50 - Avg')
#             plt.title('Temporal Evolution of the Synaptic Weights of Cells 50')
#             #plt.ylim(0)
#             plt.xlabel('Time (s)')
#             plt.ylabel('Synaptic Weight')
#             plt.legend(frameon=True)
#             plt.show()
#             
#             Y.append(J[67,50])
#             
#             plt.figure(figsize=(20,5))
#             plt.scatter(self.PFpositions,J[67])
#             plt.plot(self.PFpositions,J[67])
#             plt.xlim(0,+self.Length)
#             #plt.ylim(0)
#             plt.title('Final Pattern')
#             plt.xlabel('Place Cells over Linear Path (Blue Dots)')
#             plt.ylabel
#             
#             #Neuron 50
#             plt.figure(figsize=(20,5))
#             plt.plot(timex,Y,label='Cell 50 - Avg')
#             plt.title('Temporal Evolution of the Synaptic Weights of Cells 50')
#             #plt.ylim(0)
#             plt.xlabel('Time (s)')
#             plt.ylabel('Synaptic Weight')
#             plt.legend(frameon=True)
#             plt.show()
#==============================================================================
            
            
            
            
        else:
            print('In the input, please insert only Y or N')
    
    
    
    
    
    
    
    
    
    
    
    
    
#==============================================================================
#     
#         print()
#         print()
#         print()
#         print('After learning, the pattern looks like this:')
#         
#         plt.figure(figsize=(20,5))
#         plt.scatter(self.PFpositions,J)
#         plt.plot(self.PFpositions,J)
#         plt.xlim(0,+self.Length)   
#         plt.title('Final Pattern')
#         plt.xlabel('Place Cells over Linear Path (Blue Dots)')
#         plt.ylabel('Synaptic Weight')
#         plt.show()
#         
#         print()
#         print()
#         print('We want to see the time evolution of several individual cell weights along the path. We selected 3 of them:')
#         
#         
#         timex = arange(0,self.Time,self.Time/len(A))                    #Appropiate spacing for the x-axis to be Time
#         stepx = arange(0,self.Time,self.Time/len(X))                    #Appropiate spacing for the x-axis to be Time
#         
#         #Neuron 0
#         plt.figure(figsize=(20,5))
#         plt.plot(timex,A,label='Cell 0 - Fast')
#         plt.plot(stepx,X,label='Cell 0 - Avg')
#         plt.title('Fast Temporal Evolution of the Synaptic Weights of Cells 0')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Synaptic Weight')
#         plt.xlim(0,50)
#         plt.legend(frameon=True)
#         plt.show()
#         
#         #Neuron 50
#         plt.figure(figsize=(20,5))
#         plt.plot(timex,B,label='Cell 50 - Fast')
#         plt.plot(stepx,Y,label='Cell 50 - Avg')
#         plt.title('Fast Temporal Evolution of the Synaptic Weights of Cells 50')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Synaptic Weight')
#         plt.xlim(0,50)
#         plt.legend(frameon=True)
#         plt.show()
#     
#     
#     
#         
#         
#         
#==============================================================================
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#==============================================================================
#==============================================================================
#==============================================================================
# # #                 # Now we write the time evolution of the weights 
# # #         
# # #         J0 = random.rand(len(self.PFpositions))                 # We generate a vector J of random synaptic weights
# # #         J0 = J0/sqrt(sum(J0**2))                                # To make the weights smaller, we divided them by the square root of the sum of the squared
# # #         
# # #         print()
# # #         print()
# # #         print('We generate random initial weights:')
# # #         
# # # #==============================================================================
# # # #         plt.figure(figsize=(10,5))
# # # #         plt.scatter(self.PFpositions,J0)
# # # #         plt.plot(self.PFpositions,J0)
# # # #         plt.xlim(0,+self.Length)   
# # # #         plt.ylim(0,1)
# # # #         plt.title('Initial Weight Distribution (Randomized)')
# # # #         plt.xlabel('Place Cells over Linear Path (Blue Dots)')
# # # #         plt.ylabel('Synaptic Weight')
# # # #         plt.show()
# # # #==============================================================================
# # # 
# # #         t=0
# # #         dt=0.001
# # #         J=J0
# # #         
# # #         print(J)
# # #         
# # #         
# # #         X=[]
# # #         Y=[]
# # #         Z=[]
# # #         
# # #         while t<self.Time:
# # #             t=dt+t
# # #             J = J + dt * ( dot(MatrixKernel,J) - J**2 ) 
# # #             J[J<0]=0                                            # Positivity constrain
# # #             
# # #             if t % 0.01 <= 0.001 :                                 # Loop to store J after a certain number of steps - CELL 0
# # #                 X.append(J[0])                                     # The intention is to plot this X against time, and see if the weights are stable on time
# # #     
# # #             if t % 0.01 <= 0.001 :                                 # Loop to store J after a certain number of steps - CELL 50
# # #                 Y.append(J[50])                                     # The intention is to plot this X against time, and see if the weights are stable on time
# # #     
# # #             if t % 0.01 <= 0.001 :                                 # Loop to store J after a certain number of steps- CELL 120
# # #                 Z.append(J[120])                                     # The intention is to plot this X against time, and see if the weights are stable on time
# # # 
# # #     
# # #         print()
# # #         print()
# # #         print()
# # #         print('After learning, the pattern looks like this:')
# # #         
# # #         plt.figure(figsize=(20,5))
# # #         plt.scatter(self.PFpositions,J)
# # #         plt.plot(self.PFpositions,J)
# # #         plt.xlim(0,+self.Length)   
# # #         plt.title('Final Pattern')
# # #         plt.xlabel('Place Cells over Linear Path (Blue Dots)')
# # #         plt.ylabel('Synaptic Weight')
# # #         plt.show()
# # #         print(J)
# # #         
# # #         print()
# # #         print()
# # #         print('We want to see the time evolution of several individual cell weights along the path. We selected 3 of them:')
# # #         
# # #         
# # #         plt.figure(figsize=(20,5))
# # #         plt.plot(X,label='Cell 0')
# # #         plt.plot(Y,label='Cell 50')
# # #         plt.plot(Z,label='Cell 120')
# # #         
# # #         plt.title('Temporal Evolution of the Synaptic Weights of Cells 0, 50 and 120')
# # #         plt.xlabel('Time Steps (step=0.01s)')
# # #         plt.ylabel('Synaptic Weight')
# # #         #plt.xlim(0,5000)
# # #         plt.legend(frameon=True)
# # #         plt.show()
# # #         
# # #         print('There is a total number of steps of', len(X))
# # #     
# # #         
# # #         print(np.linalg.eig(MatrixKernel)[0])    
# # #     
# # #         print(len(rvalues))
#==============================================================================
#==============================================================================
#==============================================================================
        
        
        
        
        
        
        
        
        
        
        
        