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

def function(x1,x2):
    # Pesos de la funcion (mirar hoja a parte que tengo como a y b) 
    Dimension = 2;
    H = np.zeros((Dimension, Dimension))
    H[0,0]=1/(5000**2) #elevo al cuadrado para hacer más plana la gausiana 
    H[1,1]=1/(5000**2) # y ver que pasa
    # Funcion propiamente dicha
    f = np.exp(-(H[0,0]*(x1)**2 + H[1,1]*(x2)**2))  
    # Derivadas para comparar con el caso de que realmente exista el gradiente. 
    G = np.zeros((Dimension, 1))
    G[1,0]= -2*H[1,1]*x2*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2)) #Primera derivada.
    G[0,0]= -2*H[0,0]*x1*np.exp(-(H[0,0]*x1**2 + H[1,1]*x2**2)) #Segunda derivada.
    return f  
    
def computegradient(x1,x2,positions,D,center,num_of_agents):
    N = num_of_agents
    # print(positions)
    constante = 2/(N*D**2)
    rtotal = np.zeros((N,2),dtype='d')
    gradest = np.zeros((N,2),dtype='d')
    # ro = ro - xo
    for k in range(int(N)):
        # Evalua positionx -centerx, positiony - centery.
        fradios = function(positions[k,0]-center[0],positions[k,1]-center[1])  
        print("soy fradios:",fradios)
        # print("soy x1:",x1)
        # print("soy x1:",x2)
        # print("positions:",positions[k,0])
        # print("positions  2  :",positions[k,1])
        gradest[k,0] = fradios*(positions[k,0]-x1)
        gradest[k,1] = fradios*(positions[k,1]-x2)
        # print("soy gradest:",gradest)
    gradestfin = constante*sum(gradest)
    # print("soy rotal:",rtotal)
    # print("soy gradestfin:",gradestfin)
    return gradestfin