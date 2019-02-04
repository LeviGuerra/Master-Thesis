# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:41:32 2018

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

import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.spatial.distance import pdist, squareform
import seaborn as sns; sns.set(color_codes=True)

from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq


N = 10001
Nf = 3
t = arange(N,dtype=float)
Ts = random.rand(Nf)*200 + 100
fs = 1/Ts
print('The real unknown frequencies are:', fs)

amp = random.rand(Nf)*200+100
phi = random.rand(Nf)*2*pi

h=zeros(N)

for j in range(len(fs)):
    h += amp[j]*np.sin(2*pi*t*fs[j]+phi[j])

hn=h+random.randn(N)*3*h+random.randn(N)*700
plt.scatter(t,hn,s=3)
plt.show()

#Frequency Sampling
ind=arange(1,int(N/2+1))     # Sampling of real space
allfreqs=fftfreq(N)          # Sampling of frequency space
realfreqs=fftfreq(N)[ind]    # Frequencies in which we are interested (we omit redundancies due to complex conjugates)

#We now put the fourier transform coefficients
Hn = scipy.fftpack.fft(hn)
plt.plot(allfreqs,Hn)
plt.show()

#Power spectral density
Hn = scipy.fftpack.fft(hn)
psd= abs(Hn[ind])**2 + abs(Hn[-ind])**2
plt.plot(realfreqs,psd)
plt.show()

#We paint it closer
plt.plot(realfreqs,psd)
plt.xlim(0,0.02)

#We select now the principal frequencies of the psd
indcut=np.where(psd>0.3e12)                 # We filter relevant frequencies
SelectedFreqs=realfreqs[indcut]             # Not corrected positions of main frequencies - We putted away zero-freq
for i in arange(len(SelectedFreqs)):
    plt.axvline(SelectedFreqs[i],c='k')
plt.show()

#Now we correct to the real positions of the selected frequencies in the entire array
print(ind[indcut])                          #Corrected positions of main frequencies - We selected frequencies over psd, which is plotted over ind (not allfreq)
FinalFreqs=realfreqs[ind[indcut]]
print(FinalFreqs)

#Now we re-built the simplified Fourier Transform
Hncut=zeros(len(Hn))
Hncut[ind[indcut]]=Hn[ind[indcut]]
Hncut[-ind[indcut]]=Hn[-ind[indcut]]
hncut=ifft(Hncut)

plt.figure(figsize=(30,15))
plt.scatter(t,hn,s=3,label='Real Data (Noisy)')
plt.plot(t,hncut,'k--',lw=3,label='Estimated Signal')
plt.plot(t,h,'b',lw=3,label='Real Signal')
plt.legend()
plt.plot()



















