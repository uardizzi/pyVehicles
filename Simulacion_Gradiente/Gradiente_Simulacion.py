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
epsilon = 0.1    #Poner epsilon mas pequeño si utiliza el metodo NR, si es GA un pelo mas grande.
Dimension = 2
robots = 5;

# Puntos de donde parte mi centro (esto posteriormente lo paso desde su codigo)
x1 = 0
x2 = 0
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

def computek_f(x1,x2,robots,positions):
    xo = np.array([x1,x2]) 
    N = robots
    constante = 16/(N*D**4)
    k_f = np.zeros((N,4),dtype='d')
    hesiano = np.zeros((N,4),dtype='d')
    for k in range(N): 
        fradios = function(positions[k,0],positions[k,1]) 
        f_center = function(xo[0],xo[1])
        f_total = fradios - f_center
        k_f[k,0] = f_total*(positions[k,0]-x1)*(positions[k,0]-x1)
        k_f[k,1] = f_total*(positions[k,1]-x2)*(positions[k,0]-x1) #Son las xy cruzadas.
        k_f[k,2] = f_total*(positions[k,0]-x1)*(positions[k,1]-x2) #Son las xy cruzadas.
        k_f[k,3] = f_total*(positions[k,1]-x2)*(positions[k,1]-x2)
    k_f_estimated = constante*sum(k_f)
    return k_f_estimated 

def computehesiano(k_f_estimated):
    hesiano = np.zeros((2,2),dtype='d')
    hesiano[0,0] = (-k_f_estimated[3]+3*k_f_estimated[0])/8
    hesiano[0,1] =  (k_f_estimated[2]+3*k_f_estimated[1])/8 #Son las xy cruzadas.
    hesiano[1,0] =  (k_f_estimated[1]+3*k_f_estimated[2])/8#Son las xy cruzadas.
    hesiano[1,1] = -(k_f_estimated[0]-3*k_f_estimated[3])/8
    return hesiano


positions = np.zeros((robots,2),dtype='d')
positions = placerobots(x1,x2,robots)
fun = 0
fig, ax = plt.subplots(figsize=(10,9))
while(fun < 0.99999):
    plt.pause(1) # Espera.
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
    k_f=computek_f(ck[0],ck[1],robots,positions) # Realmente no es el k_f es la Ksigma que despues se despeja de la ecuacion matricial (me falta eso).
    hesiano=computehesiano(k_f)
    plt.quiver(ck[0],ck[1],gradestfin[0],gradestfin[1],color='r')
    ax.plot(positions[:,0],positions[:,1],'o')  #Estas lineas ejecutan el metodo NR.
    positions = positions - epsilon*np.dot(np.linalg.inv(hesiano),gradestfin)
    ck = ck - epsilon*np.dot(np.linalg.inv(hesiano),gradestfin)
    # ax.plot(positions[:robots-1,0],positions[:robots-1,1],'o')  #Esta tres lineas ejecutan el metodo sin hessiano (GA).
    # positions = positions + epsilon*gradestfin
    # ck = ck + epsilon*gradestfin
    fun = function(ck[0],ck[1])
    print("soy fun: ",fun)
    print("gradestfin: ",gradestfin)
    print("hessiano: ",hesiano)
    print("soy ck: ",ck)
    if sum(abs(gradestfin)) <= 10**-4 : # Condicion de parada.
        break


#Prueba para ver que me sale la misma grafica que en matlab, simplemente es la grafica en tres dimensiones.

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)



 