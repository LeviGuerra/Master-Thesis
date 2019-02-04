# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:12:50 2017

@author: Multivac
"""

get_ipython().magic('matplotlib inline')
import random
from numpy import *
from math import *
import decimal
import numpy
import math
import numpy as np
import scipy 
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("white")
#sns.set_style("whitegrid")


# In this first python code behind my master thesis we will calculate the correlation between two Place Cells.
# The analytical result is compared to numerical simulations of an animal walking rightwards on a linear track. 
# The code can be divided in three parts:
# 1. The different variables and parameters needed are defined
# 2. Different functions with self-explanatory names are presented. They are intermediate steps, plots... that help to understand the results.
# 3. The plot of the Correlation Average comparison is included.


class Correlation:    
        
    def __init__(self,T,L):
        
        self.Speed=25                                   # Speed of the rat inside the maze (cm/s)
        self.Time = T*60                                # Total Time of the walk in seconds
        self.Length = L                                 # Diameter of the maze in cm
           
        self.Distance=2                                # Distance between fields centers in cm
        self.OmegaTheta=2*pi*8                          # Angular frequency of the theta oscillation (experimental value of 8Hz)
        self.Ri=30                                      # Radius of the place field 1
        self.Rj=30                                      # Radius of the place field 2
        self.Pi=50                                      # Peak value of Place Field 1
        self.Pj=50                                      # Peak value of Place Field 2
                
        self.DeltaOmegai=pi*self.Speed/self.Ri          # Frecuency difference between cell 1 and theta
        self.DeltaOmegaj=pi*self.Speed/self.Rj          # Frecuency difference between cell 2 and theta
        self.Omegai=self.DeltaOmegai+self.OmegaTheta    # Angular frequency of the firing rate of cell 1
        self.Omegaj=self.DeltaOmegaj+self.OmegaTheta    # Angular frecuency of the firing rate of cell 2
        self.si=self.Ri/(sqrt(2*np.log(10)))            # Std.Dev. of Cell 1 - It is fixed by the size of the place field
        self.sj=self.Rj/(sqrt(2*np.log(10)))            # Std.Dev. of Cell 2 - It is fixed by the size of the place field
        
        self.dt=0.001                                   # Time resolution (seconds)
        self.LagRadius=2                                # Interval of time in which we calculate the Correlation for different time lags (seconds). 
        
        # Correction for big LagRadius
        RunTime=self.Length/self.Speed                  # Time length of a single run
        LagRadiusTime=2*self.LagRadius                  # Time length of LagRadius. We add x2 because Lag is considered in + and - directions.
        if RunTime<LagRadiusTime:                       # If LagRadius is bigger than the Length of each run (in time units):
            self.LagRadius=self.Length/(2*self.Speed)   # if so, we fix it to the reasonable maximum: equal to the time length of a single run (read notes for more details)
            print('The time lag limits introduced were bigger than the single run time-length. They has been redefined to:', self.LagRadius)
        
        # Path Generator
        Time=0
        XX=[]
        
        while abs(Time-(self.Time))>(self.dt)**2:       # When time of simulation reaches desired time, with a difference much smaller than dt, finish simulation.
            
            if Time==0:
                X=-self.Length/2                                # Starting point
            else:                                               # If not starting, then add step
                X=X+self.dt*self.Speed
            
            if X>(self.Length/2-self.dt*self.Speed):            # When limit reached:
                X=-self.Length/2

            Time=Time+self.dt
            XX.append(X)

        self.Path = array(XX)                                     # Array with all the locations
        
        
        # Generator of theta phases: we randomize the theta phases at the entrance of place fields for each run
        ThetaList=[]
        NumberRuns=self.Time*self.Speed/(self.Length)+1              # Number of runs; i.e. number of random theta values for place field entries.
                                            
        for i in arange(0,NumberRuns):
            ThetaPhase= random.uniform(0,2*pi) 
            ThetaList.append(ThetaPhase)
        
        # IMPORTANT: Even if for each run the theta phase at entrance is random, the theta phase at the second field has to be related to the first one on each run
        DistFieldsEntrance=(self.Ri+self.Distance-self.Rj)           # First of all, we calculate the distance between the two place field entrances.
        TimeDistFields=DistFieldsEntrance/self.Speed                 # Now, we change if to time units
        PhaseDiffFields=TimeDistFields*self.OmegaTheta               # Finally, we estimate the theta phase difference
        self.ThetaGenerator1=ThetaList    
        self.ThetaGenerator2=[(x+PhaseDiffFields) for x in ThetaList]


################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################


    def PathSize(self):                                 # Gives back the Path Size in number of Steps dt
        print(self.Path)
        print(size(self.Path))
        
    
    def Spikes(self,Sigma,Peak,Center,Omega,DeltaOmega,PhiTheta):
        
        dt=self.dt                                          
        Radius=pi*self.Speed/DeltaOmega                     # Effective radius of the place field
        
        Threshold=Peak*exp(-Radius**2/(2*Sigma**2))         # Value lower Threshold = Outside place field
        
        GaussEnvelopeFun = lambda x:  Peak * exp( -((x-Center)**2)/(2*Sigma**2) )                   # Gaussian envelope function
        GaussEnvelope=[GaussEnvelopeFun(self.Path[i]) for i in arange(0,size(self.Path[:])) ]       # Array of evaluating GaussEnvelopeFun in Path
                                                               
        # Inhomogeneous Poisson point process spike generator 
        Index=0
        SpikeIndex=[]
        ProbIntegrator=0
        InsideIndex=0
        
        ProbSpike=-log(1-np.random.rand())      
        PhaseFlag=0
        i=0
        
        while Index<size(self.Path[:]):                                                                 # While Index of step inside Path:
                if GaussEnvelope[Index]>Threshold:                                                      # If inside place field:
                    
                    if PhaseFlag==0:                                                                    # If Entering field:
                        ThetaPhase=PhiTheta[i]                                                          # Theta Phase giving by the list defined above
                        i+=1                                                                            # Next element of list: i+1
                        Phase=((ThetaPhase)%(2*pi) - 2*pi)*Omega/self.OmegaTheta                        # Phase of the firing rate entering the place field
                        PhaseFlag=1                                                                     # When PhaseFlag=1 - Already inside field
                        
                    ProbIntegrator=ProbIntegrator+GaussEnvelope[Index]*cos(.5*(Omega*InsideIndex*dt+Phase))**2*dt   
                    InsideIndex=InsideIndex+1
                                                                                                                                  
                    if ProbIntegrator>ProbSpike:
                        SpikeIndex.append(Index)
                        ProbIntegrator=0
                        ProbSpike=-log(1-np.random.rand())
                else:
                    PhaseFlag=0
                    InsideIndex=0
                                                                                                                                          
                Index=Index+1
        i=0
        return SpikeIndex


    #Plots an histogram of a certain number of the random numbers used to generate the Spikes    
    def RandomDist(self):                   
        k = [-log(1-np.random.rand()) for i in arange(0,10000)]

        bins = numpy.linspace(0,20, 10000)
        plt.figure(figsize=(15,10))
        pyplot.hist(k, bins, alpha=0.75, label='ProbSpike - Random number distribution for Spike Generation', color=sns.xkcd_rgb["coral pink"])
        pyplot.legend(loc='upper right')
        pyplot.show()
        print('Here we see a certain histogram for the random numbers used to generate the Spikes')
        
        
    def ThetaDist(self):                        #Plots the real distribution of initial theta values each time the run is restarted. 
                                                #The value at the place field entry is just a fixed phase difference. We use this to justify why we generate random thetas in the Spike generator.                                       
        T=self.Time                             # Minutes running
        Runs=T*self.Speed/(self.Length)         # Number of runs
        RunTime=self.Length/(self.Speed)        # Time spent on a single run
        
        print(' ')
        print('The running time is',T,'s, and the run-length',self.Length,'cm.')        
        print('The speed is',self.Speed,'m/s, and hence the time per run is',RunTime,'s')
        print('The number of runs is:',Runs)
        print(' ')
        print('Now we will chop the theta wave at the end of each run and measure the phase with which the run is restarted. A distribution will be plotted.')
        print(' ')
        
        Period=1/8                              # Periodicity of the Theta cycles per second    
        
        ThetaList=[]
        
        Dt=(RunTime)%Period
        ThetaPhase=0
        
        for i in arange(0,Runs):
            ThetaPhase=(ThetaPhase+Dt*self.OmegaTheta)%2*pi
            ThetaList.append(ThetaPhase)
        
        print(ThetaList)
        
        bins = numpy.linspace(0,2*pi,500)
        plt.figure(figsize=(10,10))
        pyplot.hist(ThetaList, bins, alpha=0.75, label='Number of runs restarted with this theta phase..', color=sns.xkcd_rgb["coral pink"])
        pyplot.legend(loc='upper right')
        pyplot.show()
                                                       

    def SpikeIndex1(self):
        return self.Spikes(self.si,self.Pi,-(self.Distance)/2,self.Omegai,self.DeltaOmegai,self.ThetaGenerator1)
    def SpikeIndex2(self):
        return self.Spikes(self.sj,self.Pj,(self.Distance)/2,self.Omegaj,self.DeltaOmegaj,self.ThetaGenerator2) 
    
    
    def PathPlot(self):     
        
        L=self.Length
        
        SpikeIndex1=self.SpikeIndex1()
        SpikeIndex2=self.SpikeIndex2()
        
        circlel=plt.Circle((-self.Distance/2, 0), self.Ri, color='r', fill=False, lw=1)
        circler=plt.Circle((+self.Distance/2, 0), self.Rj, color='b', fill=False, lw=1)
        
        print()
        print('Number of steps is:', size(self.Path))
        print()
        print('The distance between place fields centers is:', str(self.Distance))
        print('Number of Spikes from cell 1 with radius', str(self.Ri),'and frequency', str(self.Omegai), 'is', size(SpikeIndex1))
        print('Number of Spikes from cell 2 with radius', str(self.Rj),'and frequency', str(self.Omegaj),'is', size(SpikeIndex2))
        print('Value of sigma 1 is', str(self.si))
        print('Value of sigma 2 is', str(self.sj))
        
        #Plots coming in the following lines:
        
        plt.figure(figsize=(15,10))
        plt.plot(self.Path[:],zeros(size(self.Path)),alpha=1,c='gray',lw=1,zorder=1)
        plt.scatter(self.Path[SpikeIndex1],zeros(size(SpikeIndex1)),s=pi*3**2,c=sns.xkcd_rgb["coral pink"],alpha=1,zorder=3)            
        plt.scatter(self.Path[SpikeIndex2],zeros(size(SpikeIndex2)),s=pi*3**2,alpha=1,zorder=2)
        
        plt.gcf().gca().add_artist(circlel)         #Get current figure, get current axis, add CIRCLE LEFT
        plt.gcf().gca().add_artist(circler)         #Get current figure, get current axis, add CIRCLE RIGHT
        
        plt.xlim(-L/2-L/16,L/2+L/16)   
        plt.ylim((-L/2-L/16)*2/3,(L/2+L/16)*2/3)
        
        ########### We add an histogram here too ###########
        
        i=0
        X=[]                                                # List with the positions of spikes 
        while i<len(SpikeIndex1):                           # If there are still spikes left
            x=self.Path[SpikeIndex1][i]                     # Calculate the position at which take place
            X.append(x)                                     # Add it to the array of positions of spikes
            i=i+1                                           # Move one position in Spikes list
        
        j=0
        Y=[]
        while j<len(SpikeIndex2):                           # If there are still spikes left from Place cell 2
            y=self.Path[SpikeIndex2][j]                     # Calculate the position at which take place
            Y.append(y)                                     # Add it to the array of positions of spikes
            j=j+1                                           # Move one position in Spikes list
        
       
        bins = numpy.linspace(-L/2,L/2, 2000)
        pyplot.hist(X, bins, alpha=0.9, label='Neuron 1', color=sns.xkcd_rgb["coral pink"],zorder=4,lw=0)
        pyplot.hist(Y, bins, alpha=0.9, label='Neuron 2', color='b', zorder=5,lw=0)
        pyplot.legend(loc='upper right',fontsize=25)
        plt.tick_params(labelsize=30)
        plt.xlabel('$x$ (cm)', fontsize=30)
        plt.ylabel('Number of Spikes per bin', fontsize=30)
        pyplot.show()
        
        #In the following, what we do is to take the two lists of generated spikes and calculate the size of the
        #intersections between them for different time-lag values. What this does, is to give the correlation
        #for different time lags, since if we add +1 to every element in the list SpikeIndexA, we are
        # actually moving temporarilly one step dt the whole set of values. 
        #
        #On top of this plot, we add the analytical expression of the correlation.
        #
        #The shape of the correlation is symmetric back and forth movements,
        #and it will be peaked positively (negatively) for rightward (leftward) directed paths
   

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

    
    def CorrelationAvg(self):                 #Exact Correlation Function Averaging for all onset PhiTheta
        
        Lag=self.LagRadius
        dt=self.dt

        ########## ANALYTICAL CORRELATION ##########      

        si=self.si
        sj=self.sj
        Pi=self.Pi
        Pj=self.Pj        

        s=sqrt(2*(si**2*sj**2)/(si**2+sj**2))
        sbar=sqrt((si**2+sj**2)/2)
        sbari=sqrt(si**2/(si**2+sj**2))
        sbarj=sqrt(sj**2/(si**2+sj**2))
        wtheta=self.OmegaTheta
        wi=self.Omegai                        
        wj=self.Omegaj                        
        wbar=sbari**2*wi+sbarj**2*wj
        Dw=wi-wj                              #Difference between wi and wj

        Ri=self.Ri                            
        Rj=self.Rj                            
        
        v=self.Speed
        r=self.Distance
        
        diR = lambda x_c,gamma: sqrt(Ri**2 - (x_c - sbari**2*r*sin(gamma))**2 ) + sbari**2*r*cos(gamma)
        djR = lambda x_c,gamma: sqrt(Rj**2 - (x_c + sbarj**2*r*sin(gamma))**2 ) - sbarj**2*r*cos(gamma)
        #diL = lambda x_c,gamma: sqrt(Ri**2 - (x_c + sbari**2*r*sin(gamma))**2 ) + sbari**2*r*cos(gamma)
        #djL = lambda x_c,gamma: sqrt(Rj**2 - (x_c - sbarj**2*r*sin(gamma))**2 ) - sbarj**2*r*cos(gamma)
    
        Coeff = lambda x_c,x,gamma : sqrt(pi)*s/(4*v)*Pi*Pj*exp(-x_c**2/(s**2))*exp(-v**2*x**2/(4*sbar**2))*exp(-r**2/(4*sbar**2))*exp(-v*r*x*cos(gamma)/(2*sbar**2))
        
        Exact1DRight = lambda x: Coeff(0,x,0)*(1+.5*exp(-(s**2)*(Dw)**2/(4*v**2))
        *(cos(wbar*x+(wi+wj)/(2*wtheta)*(wtheta/v*(diR(0,0)-djR(0,0))-(wtheta/v*(diR(0,0)-djR(0,0)))%(2*pi))+Dw/(2*v)*(diR(0,0)+djR(0,0))-pi*Dw/wtheta)
          -((wtheta*(diR(0,0)-djR(0,0))/self.Speed)%(2*pi))/pi*sin(pi*wj/wtheta)*sin(wbar*x+(wi+wj)/(2*wtheta)*(wtheta/v*(diR(0,0)-djR(0,0))-(wtheta/v*(diR(0,0)-djR(0,0)))%(2*pi))+Dw/(2*v)*(diR(0,0)+djR(0,0))-pi*Dw/wtheta+pi*wj/wtheta)))
        
        ExactCorrR=[Exact1DRight(x) for x in arange(-Lag,Lag+dt,dt)]                                    #List of points between -+Lag with spacing dt

        plt.figure(figsize=(20,15))
        #plt.title('Place Fields Pair Correlation', fontsize=20)
        plt.tick_params(labelsize=55)
        plt.xlabel('$τ$ (s)', fontsize=65)
        plt.ylabel('Correlation Rate $C_{ij}$ (Hz)', fontsize=65)

        plt.plot(arange(-Lag,Lag+dt,dt),ExactCorrR,lw=3,c=sns.xkcd_rgb["amber"],zorder=2)       
        

        ########## SIMULATED CORRELATION ##########
        
        #For simulated correlation, it is necessary to work with integrers; hence, we change time units so dt is integrer, and then return to seconds at the end:
        def number_of_decimals(x):  
            count = 0  
            residue = x -int(x)  
            if residue != 0:  
                multiplier = 1  
                while not (x*multiplier).is_integer():  
                    count += 1  
                    multiplier = 10 * multiplier  
                return count
        k=number_of_decimals(dt)
    
        Lag=Lag*(10**k)
        dt=dt*(10**k)      
        
        
        SpikeIndex1=self.SpikeIndex1()
        SpikeIndex2=self.SpikeIndex2()
        
        print(' ')        
        print('Number of steps is:', size(self.Path))
        print(' ')
        print('The distance between place fields centers is:', str(self.Distance))
        print('Number of Spikes from cell 1 with radius', str(self.Ri),'and frequency', str(self.Omegai), 'is', size(SpikeIndex1))
        print('Number of Spikes from cell 2 with radius', str(self.Rj),'and frequency', str(self.Omegaj),'is', size(SpikeIndex2))
        print('Value of sigma 1 is', str(self.si))
        print('Value of sigma 2 is', str(self.sj))
        
        #Normalization
        NumberRuns=(self.Length/(self.Speed*self.Time))
        N=NumberRuns/(self.dt)                # Normalization of the correlation
        
        #Correlation and plot
        Corr=[size(list(set(SpikeIndex1).intersection(SpikeIndex2+i)))*N for i in arange(-Lag/dt,(Lag+dt)/dt,1)]        #Calculate Correlation by counting coincidences. The interval is -+Lag/dt because we add time steps, not time directly.
        CorrConv=scipy.ndimage.gaussian_filter(Corr, sigma=1)           #Smoothed simulated correlation
        plt.axvline(0, color='black',linestyle='dashed', dashes=(2,10))
        plt.plot(arange(-Lag/10**k,(Lag+dt)/10**k,dt/10**k),CorrConv,c=sns.xkcd_rgb["taupe"],zorder=1)                              #Plot the correlation. For that we have to consider that now the interval length is 2*Lag+dt, and we had (2*Lag+dt)/dt legth with spacing dt. So, by cross multiplication, the new spacing is dt' = dt*(2*Lag+dt)/((2*Lag+dt)/dt) = dt*dt
        #plt.plot(arange(-round(Lag/dt)*dt/10**k,round((Lag+dt)/dt)*dt/10**k,dt/10**k),CorrConv,c=sns.xkcd_rgb["taupe"],zorder=1)                              #Plot the correlation. For that we have to consider that now the interval length is 2*Lag+dt, and we had (2*Lag+dt)/dt legth with spacing dt. So, by cross multiplication, the new spacing is dt' = dt*(2*Lag+dt)/((2*Lag+dt)/dt) = dt*dt

        # IMPORTANT REMARK 1: Why do we make the plot with steps dt**2?
        # We are plotting between -Lag and +Lag. But we add steps, not time, so  interval is given between -Lag/dt and +Lag/t (in steps of 1, because each time step is already dt in time)
        # When we plot, we plot in time, so the interval is -Lag to +Lag in steps dt.        
        #
        # IMPORTANT REMARK 2: Why do we use round and int?
        # Recall that we have an arange with limits Lag/dt. We need integrer numbers, and this can be float.
        # Hence, we apply round to make it integrer. But, we can have the problem that 
        
        print(' ')
        print('The analytical firing rate average is:', str(mean(ExactCorrR)), 'The simulated firing rate average is', str(mean(Corr)),'.',' The corresponding ratio is:',str(mean(ExactCorrR)/mean(Corr)))
        
        plt.show()
        
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################    
    
  