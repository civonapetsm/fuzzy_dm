import numpy as np
import matplotlib.pyplot as plt

Mext=10**-2                               #masa homogene pozadine
R=1.1                                     #radijus homogene pozadine
p=Mext/(4/3*R**3*np.pi)                   #gustina homogene pozadine
lam=0.3                                   #udeo ULA u homogenoj pozadini (ukljuèivanje/iskljuèivanje oscilujuæe komponente)
q=1                                       #ukljuèivanje/iskljuèivanje homogene pozadine

r0=1                                      #radijus orbite tela u poèetnom trenutku
rtacka0=0                                 #prvi izvod radijusa u poèetnom trenutku                             
fi0=0                                     #fi koordinata u poèetnom trenutku
e=0                                       #vrednost prvog izvoda efektivnog potencijala
L0=np.sqrt(r0+(Mext/R**3)*r0**4-e*r0**3)  #redukovani moment impulsa u poèetnom trenutku     
fitacka0=L0/r0**2                         #fi taèka u poèetnom trenutku  

d=1
n=2                                       
m=0

if (n==4):
    d=2

w_res=0.5*np.sqrt(1/r0**3+4*Mext/R**3)    #omega rezonantno
w=2*w_res/n+m                             #omega tamne materije

lr=[r0]
lrtacka=[rtacka0]
lfi=[fi0]
lfitacka=[fitacka0]
lx=[]
ly=[]

lL=[]
ldL=[]
lE=[]
ldE=[]

T=500*d
dt=0.005
lt=np.arange(0,T,dt)

for i in range(len(lt)):
    
    #resavamo 2 diferencijalne jednacine RK4 metodom:
    #r2tacka-r*fitacka**2=-1/r**2-*Mext*r/R**3+4*np.pi*lam*p*r*cos(2*w*t)
    #r*fi2tacka+2*rtacka*fitacka=0
    
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
      
    #racunamo nove vrednosti za r,rtacka,fi i fitacka i prikljucujemo ih odgovarajucim listama
    
    r=lr[i]+(k1_r+2*k2_r+2*k3_r+k4_r)/6
    lr.append(r)
    
    rtacka=lrtacka[i]+(k1_rtacka+2*k2_rtacka+2*k3_rtacka+k4_rtacka)/6
    lrtacka.append(rtacka)
               
    fi=lfi[i]+(k1_fi+2*k2_fi+2*k3_fi+k4_fi)/6
    lfi.append(fi)
    
    fitacka=lfitacka[i]+(k1_fitacka+2*k2_fitacka+2*k3_fitacka+k4_fitacka)/6
    lfitacka.append(fitacka)
    
    #iyracunavamo odgovarajucu x i y koordinatu da bi smo mogle da plotujemo orbite
    
    x=r*np.cos(fi)
    lx.append(x)
    y=r*np.sin(fi)
    ly.append(y)
    
    #takodje u svakom koraku racunamo vrednost momenta impulsa i energije
    
    L=r**2*fitacka
    lL.append(L)
      
    E=0.5*rtacka**2+L**2/(2*r**2)-1/r+q*Mext*(r**2-3*R**2)/(2*R**3)
    lE.append(E)


plt.plot(lx,ly)
plt.show()
plt.plot(lr)
plt.show()

for i in range(len(lE)):
    
    #racunamo relativnu promenu momenta impulsa i energije
    
    dL=lL[i]/lL[0]
    ldL.append(dL)
    
    dE=lE[i]/lE[0]
    ldE.append(dE)

plt.plot(ldE)
plt.show()

plt.plot(ldL)
plt.show()