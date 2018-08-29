# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:54:22 2018

@author: MILICA
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

lM=[10**-3,10**-2]
llam=[0.1,0.2,0.3]

lA=[]
lP=[]

for i in range (len(lM)):
    M=lM[i]
    for j in range (len(llam)):
        Mlam=M*llam[j]
        if (Mlam>=10**-5) and (Mlam<=10**-3):
            
            Mext=M 
            R=1.1                                     
            p=Mext/(4/3*R**3*np.pi)                   
            lam=llam[j]                                  
            q=1                                        
            r0=1
            rtacka0=0
            fi0=0                                    
            e=0                                       
            L0=np.sqrt(r0+(Mext/R**3)*r0**4-e*r0**3)     
            fitacka0=L0/r0**2                                 
            d=1
            n=4                                       
            if (n==4):
                d=2   
            w_res=0.5*np.sqrt(1/r0**3+4*Mext/R**3)    
            w=2*w_res/n
            lr=[r0]
            lrtacka=[rtacka0]
            lfi=[fi0]
            lfitacka=[fitacka0]
            T=30000*d
            dt=0.005
            lt=np.arange(0,T,dt)
            
            for i in range(len(lt)-1):
                
                k1_rtacka=(lr[i]*lfitacka[i]**2-1/lr[i]**2-q*Mext*lr[i]/R**3+4*np.pi*lam*p*lr[i]*np.cos(2*w*lt[i]))*dt 
                k1_fitacka=(-2*lrtacka[i]*lfitacka[i])/lr[i]*dt
                k1_r=lrtacka[i]*dt
                k1_fi=lfitacka[i]*dt
                k2_rtacka=((lr[i]+k1_r/2)*(lfitacka[i]+k1_fitacka/2)**2-1/(lr[i]+k1_r/2)**2-q*Mext*(lr[i]+k1_r/2)/R**3+4*np.pi*lam*p*(lr[i]+k1_r/2)*np.cos(2*w*(lt[i]+dt/2)))*dt 
                k2_fitacka=(-2*(lrtacka[i]+k1_rtacka/2)*(lfitacka[i]+k1_fitacka/2))/(lr[i]+k1_r/2)*dt           
                k2_r=(lrtacka[i]+ k1_rtacka/2)*dt
                k2_fi=(lfitacka[i]+k1_fitacka/2)*dt            
                k3_rtacka=((lr[i]+k2_r/2)*(lfitacka[i]+k2_fitacka/2)**2-1/(lr[i]+k2_r/2)**2-q*Mext*(lr[i]+k2_r/2)/R**3+4*np.pi*lam*p*(lr[i]+k2_r/2)*np.cos(2*w*(lt[i]+dt/2)))*dt 
                k3_fitacka=(-2*(lrtacka[i]+k2_rtacka/2)*(lfitacka[i]+k2_fitacka/2))/(lr[i]+k2_r/2)*dt           
                k3_r=(lrtacka[i]+ k2_rtacka/2)*dt
                k3_fi=(lfitacka[i]+k2_fitacka/2)*dt  
                k4_rtacka=((lr[i]+k3_r)*(lfitacka[i]+k3_fitacka)**2-1/(lr[i]+k3_r)**2-q*Mext*(lr[i]+k3_r)/R**3+4*np.pi*lam*p*(lr[i]+k3_r)*np.cos(2*w*(lt[i]+dt)))*dt 
                k4_fitacka=(-2*(lrtacka[i]+k3_rtacka)*(lfitacka[i]+k3_fitacka))/(lr[i]+k3_r)*dt         
                k4_r=(lrtacka[i]+ k3_rtacka)*dt
                k4_fi=(lfitacka[i]+k3_fitacka)*dt           
                r=lr[i]+(k1_r+2*k2_r+2*k3_r+k4_r)/6
                lr.append(r)    
                rtacka=lrtacka[i]+(k1_rtacka+2*k2_rtacka+2*k3_rtacka+k4_rtacka)/6
                lrtacka.append(rtacka)               
                fi=lfi[i]+(k1_fi+2*k2_fi+2*k3_fi+k4_fi)/6
                lfi.append(fi)  
                fitacka=lfitacka[i]+(k1_fitacka+2*k2_fitacka+2*k3_fitacka+k4_fitacka)/6
                lfitacka.append(fitacka)
                
            fs=len(lt)
            cutoff=50
            def FilteredSignal(lr,fs,cutoff):
                B,A=butter(1,cutoff/(fs/2),btype='low')
                filtered_signal=filtfilt(B,A,lr,axis=0)
                return filtered_signal
            analyticSignal=hilbert(lr)
            amplitudeEvelope = np.abs(analyticSignal)
            filteredSignal=FilteredSignal(amplitudeEvelope, fs, cutoff)
            
            lrt=[]
            for i in range(len(lt)):
                lrt.append((lt[i],filteredSignal[i]))
            lrtmax=[]
            for i in range (len(lrt)-2):
                if (lrt[i][1]<=lrt[i+1][1]) and (lrt[i+1][1]>=lrt[i+2][1]):
                    lrtmax.append(lrt[i+1])
            lrtmax2=[]
            for i in range (len(lrtmax)-2):
                if (lrtmax[i][1]<=lrtmax[i+1][1]) and (lrtmax[i+1][1]>=lrtmax[i+2][1]):
                    lrtmax2.append(lrtmax[i+1])
                    
            P=2*lrtmax2[0][0]
            A=max(lr)
            lA.append((Mlam,A))
            lP.append((Mlam,P))

x1,y1=zip(*lA)
plt.plot(x1,y1)
print(lA)
plt.show()
x2,y2=zip(*lP)
plt.plot(x2,y2)
print(lP)
plt.show()