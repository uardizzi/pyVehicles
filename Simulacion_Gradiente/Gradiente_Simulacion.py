# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 01:38:19 2021

@author: tutir
"""

# Recordar modificar en funcion de lo que llame en el pdf.
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import random
import time

# Cosas para graficar.
limite = 25
precision = 10
espaciado = limite*2*precision + 1
X = np.linspace(-limite, limite, espaciado)
Y = np.linspace(-limite, limite, espaciado)
Xgrid, Ygrid = np.meshgrid(X, Y)

# Parametros
D = 2
angulo = np.linspace(0, 2*np.pi, 1000) 
epsilon = 10
Dimension = 2
robots = 5;

# Puntos de donde parte mi centro (esto posteriormente lo paso desde su codigo)
x1 = 20
x2 = 20
ck = np.array([x1,x2])             #Ck en cada momento sera el centro.

def function(x1,x2):
    # Pesos de la funcion (mirar hoja a parte que tengo como a y b) 
    H = np.zeros((Dimension, Dimension))
    H[0,0]=1/500
    H[1,1]=1/500
    # Funcion propiamente dicha
    f = np.exp(-(H[0,0]*(x1-10)**2 + H[1,1]*(x2-10)**2))  
    # Derivadas para comparar con el caso de que realmente exista el gradiente. 
    # G = np.zeros((Dimension, 1))
    # G[1,0]= -2*H[1,1]*x2*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2)) #Primera derivada.
    # G[0,0]= -2*H[0,0]*x1*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2)) #Segunda derivada.
    return f    

# Z = math.e**-(H[0,0]*(Xgrid * Xgrid) + H[1,1]*(Ygrid * Ygrid))
zz = np.zeros((espaciado,espaciado),dtype='d')
for i in range(espaciado):
  for j in range(espaciado):
    zz[i,j] = function(Xgrid[i,j],Ygrid[i,j])   

def placerobots(x1,x2,robots):
    xo = np.array([x1,x2]) 
    N = robots
    a = 2*np.pi/N
    ro = np.array([x1 + D,x2])
    rtotal = np.zeros((N,2),dtype='d')
    ro = ro - xo
    for k in range(N):
        if k == N-1:
            ri = np.array([0,0])
        else:
            a = 2*np.pi*(k+1)/(N-1)
            cphi = round(np.cos(a),8)
            sphi = round(np.sin(a),8)
            Rotacion = np.array([[cphi, -sphi] , [sphi, cphi]])
            ri = np.dot(Rotacion,np.transpose(ro)) 
        rtotal[k,0] = ri[0] + xo[0]
        rtotal[k,1] = ri[1] + xo[1]
    return rtotal
    
  
def computegradient(x1,x2,robots,positions):
    xo = np.array([x1,x2]) 
    N = robots-1
    constante = 2/(N*D**2)
    gradest = np.zeros((N,2),dtype='d')
    for k in range(N):
        fradios = function(positions[k,0],positions[k,1]) 
        gradest[k,0] = fradios*(positions[k,0]-x1)
        gradest[k,1] = fradios*(positions[k,1]-x2)
    gradestfin = constante*sum(gradest)
    return gradestfin

def computehesian(x1,x2,robots,positions):
    xo = np.array([x1,x2]) 
    N = robots
    constante = 16/(N*D**4)
    hesiano = np.zeros((2*N,2),dtype='d')
    for k in range(N): 
        fradios = function(positions[k,0],positions[k,1]) 
        f_center = function(xo[0],xo[1])
        f_total = fradios - f_center
        hesiano[2*k,0] = f_total*(positions[k,0]-x1)*(positions[k,0]-x1)
        hesiano[2*k,1] = f_total*(positions[k,1]-x2)*(positions[k,0]-x1)
        hesiano[2*k+1,0] = f_total*(positions[k,0]-x1)*(positions[k,1]-x2)
        hesiano[2*k+1,1] = f_total*(positions[k,1]-x2)*(positions[k,1]-x2)
    hesian_estimated = constante*sum(hesiano)
    return hesian_estimated


positions = np.zeros((robots,2),dtype='d')
positions = placerobots(x1,x2,robots)
fun = 0
fig, ax = plt.subplots(figsize=(10,9))
while(fun < 0.99999):
    plt.pause(0.5) # Espera.
    plt.cla()
    cp = ax.contour(Xgrid, Ygrid, zz, 10, cmap = 'RdGy') # Contorno.
    ax.clabel(cp, inline=True, fontsize=12)
    ax.set_title('Contour Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    h = ck[0]   # Circulito
    k = ck[1]
    Xcir = D * np.cos(angulo) + h
    Ycir = D * np.sin(angulo) + k
    ax.plot(Xcir,Ycir, color='r')
    gradestfin=computegradient(ck[0],ck[1],robots,positions) # Gradiente.
    hesiano=computehesian(ck[0],ck[1],robots,positions) # Realmente no es el hesiano es la Ksigma que despues se despeja de la ecuacion matricial (me falta eso).
    ax.plot(positions[:robots-1,0],positions[:robots-1,1],'o')  #Robots-1 para graficar con el gradiente (omite el central)
    plt.quiver(ck[0],ck[1],gradestfin[0],gradestfin[1],color='r')
    positions = positions + epsilon*gradestfin
    ck = ck + epsilon*gradestfin # Avance.
    fun = function(ck[0],ck[1])
    print("soy fun: ",fun)
    print("gradestfin: ",gradestfin)
    print("hesiano: ",hesiano)
    print("soy ck: ",ck)
    if sum(abs(gradestfin)) <= 10**-4 : # Condicion de parada.
        break


#Prueba para ver que me sale la misma grafica que en matlab, simplemente es la grafica en tres dimensiones.

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)



 