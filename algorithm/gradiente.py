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
import pygame

# Cosas para graficar.
# limite = 100
# precision = 10
# espaciado = limite*2*precision + 1
# X = np.linspace(-limite, limite, espaciado)
# Y = np.linspace(-limite, limite, espaciado)
# Xgrid, Ygrid = np.meshgrid(X, Y)

# #Implement]acion de la gausiana.
# # Z = math.e**-(H[0,0]*(Xgrid * Xgrid) + H[1,1]*(Ygrid * Ygrid))
# zz = np.zeros((espaciado,espaciado),dtype='d')

# for i in range(espaciado):
#   for j in range(espaciado):
#     zz[i,j] = function(Xgrid[i,j],Ygrid[i,j]) 
    
# # Parametros
# # D = 0.5
# angulo = np.linspace(0, 2*np.pi, 1000) 
# epsilon = 20
# Dimension = 2

# # Puntos de donde parte mi centro.
# # x1 = random.uniform(-limite,limite)
# # x2 = random.uniform(-limite,limite)
# # ck = np.array([x1,x2])             #Ck en cada momento sera el centro.

# fig, ax = plt.subplots(figsize=(10,9))
# for k in range(10):
#     plt.pause(0.5) # Espera.
#     plt.cla()
#     cp = ax.contour(Xgrid, Ygrid, zz, 10, cmap = 'RdGy') # Contorno.
#     ax.clabel(cp, inline=True, fontsize=12)
#     ax.set_title('Contour Plot')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     h = ck[0]   # Circulito
#     k = ck[1]
#     Xcir = D * np.cos(angulo) + h
#     Ycir = D * np.sin(angulo) + k
#     ax.plot(Xcir,Ycir, color='r')
#     [rtotal,gradestfin]=computegradient(ck[0],ck[1]) # Gradiente.
#     ax.plot(rtotal[:,0],rtotal[:,1],'o')
#     plt.quiver(ck[0],ck[1],gradestfin[0],gradestfin[1],color='r')
#     rtotal = rtotal + epsilon*gradestfin
#     ck = ck + epsilon*gradestfin # Avance.
#     if sum(abs(gradestfin)) <= 10**-4 : # Condicion de parada.
#         break


def function(x1,x2):
    # Pesos de la funcion (mirar hoja a parte que tengo como a y b) 
    Dimension = 2;
    H = np.zeros((Dimension, Dimension))
    H[0,0]=1/500
    H[1,1]=1/500
    # Funcion propiamente dicha
    f = np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2))  
    # Derivadas para comparar con el caso de que realmente exista el gradiente. 
    G = np.zeros((Dimension, 1))
    G[1,0]= -2*H[1,1]*x2*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2)) #Primera derivada.
    G[0,0]= -2*H[0,0]*x1*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2)) #Segunda derivada.
    return f  
    
def computegradient(x1,x2,positions,D):
    # xo = np.array([x1,x2]) 
    N = 4
    # a = 2*np.pi/N
    # ro = np.array([x1 + D,x2])
    # print(positions)
    constante = 2/(N*D**2)
    rtotal = np.zeros((N,2),dtype='d')
    gradest = np.zeros((N,2),dtype='d')
    # ro = ro - xo
    for k in range(int(N)):
        # a = 2*np.pi*(k+1)/N
        # ri[0]=positions[2*k]
        # ri[1]=positions[2*k+1]
        # cphi = round(np.cos(a),8)
        # sphi = round(np.sin(a),8)
        # Rotacion = np.array([[cphi, -sphi] , [sphi, cphi]])
        # ri = np.dot(Rotacion,np.transpose(ro)) 
        # fradios = np.exp(-(H[0,0]*(ri[0]+xo[0])**2 + H[1,1]*(ri[1]+xo[1])**2))  
        fradios = function(positions[2*k],positions[2*k+1]) 
        # print("soy fradios:",fradios)
        gradest[k,0] = fradios*(positions[2*k]-x1)
        gradest[k,1] = fradios*(positions[2*k+1]-x2)
        # print("soy gradest:",gradest)
        rtotal[k,0] = positions[2*k] + x1
        rtotal[k,1] = positions[2*k+1] + x2
    gradestfin = constante*sum(gradest)
    # print("soy rotal:",rtotal)
    # print("soy gradestfin:",gradestfin)
    
    return gradestfin

