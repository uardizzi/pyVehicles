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

def function(x1,x2,th,desv,flag):
    s=np.zeros((2,1),dtype='d')
    Hnew=np.zeros((2,2),dtype='d')
    s[0]= desv[0]
    s[1]= desv[1]
    x0 = x1
    y0 = x2
    cth = np.cos(th)
    sth = np.sin(th)
    R =np.array([x0,y0])
    Rot = np.array([[cth,sth],[-sth,cth]]) #matriz de rotación
    Q = np.array([[1/(2*s[0]**2),0],[0,1/(2*s[1]**2)]])
    Rrt = np.dot(Rot,R)
    H = Rrt*Q*Rrt
    Hnew[0,0] = H[0,0]
    Hnew[0,1] = H[0,1]
    Hnew[1,0] = H[1,0]
    Hnew[1,1] = H[1,1]
    Hfin = np.sum(H)
    f = np.exp(-np.sum(H))
    Dimension = 2;
    x1 = x0
    x2 = y0
    G = np.zeros((Dimension, 1),dtype='d')
    comp_x = np.zeros((2,2),dtype='d')
    comp_x[0,0] = x1
    comp_x[0,1] = x1
    comp_x[1,0] = x1
    comp_x[1,1] = x1
    comp_y = np.zeros((2,2),dtype='d')
    comp_y[0,0] = x2
    comp_y[0,1] = x2
    comp_y[1,0] = x2
    comp_y[1,1] = x2
    Res_x = np.divide(Hnew, comp_x,out=np.zeros_like(Hnew), where=comp_x!=0)
    Res_y = np.divide(Hnew, comp_y,out=np.zeros_like(Hnew), where=comp_y!=0)
    G[1,0]= (-2*Res_y[0,0]-Res_x[0,1]-Res_x[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) #Primera derivada.
    G[0,0]= (-2*Res_x[0,0]-Res_y[0,1]-Res_y[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) 
    if flag==0:
        return f
    else:
        return G  
    
def gausianillas(x,y,c,s,th,p):
    """
    Devuelve el valor de una gausiana definida en dos dimesiones x e y.
    Cada dimension tiene su propia desviación. La gausisana puede tener su
    centro en cualquier valor arbitrario y estar orientada 
    también de modo arbitrario
    x e y pueden ser:
    un par de puntos en los que calcular el valor V que toma la gausiana
    Un par de matrices generadas por meshgrid para crear una malla de puntos
    En este caso V es una matriz de la misma dimension que x e y con los
    valores que toma v en los  nodos de la malla
    c =[cx,cy]. representa el centro de la gausiana debe ser un array de 
    dos elementos
    s = [sx,sy] desviación en la dirección xy, en lo que podríamos llamar
    'ejes gaussiana'
    th angulo de los 'ejes gaussiana' respecto a los ejes tierra.
    p modulo de la gausiana (valor que toma en el centro)
    
    Return -> V (mirar definición en el código de la función línea 41)
    """

    x0 = x-c[0]
    y0 = y-c[1]
    cth = np.cos(th)
    sth = np.sin(th)
    R =np.array([x0,y0])
    Rot = np.array([[cth,sth],[-sth,cth]]) #matriz de rotación
    Q = np.array([[1/(2*s[0]**2),0],[0,1/(2*s[1]**2)]])
    if x0.ndim > 1:       
        Rrt = np.dot(Rot,R.transpose(1,0,2))        
    else:
        Rrt = np.dot(Rot,R)
   
    #V = p*np.exp(-(Rrt[0]**2/(2*s[0]**2)+Rrt[1]**2/(2*s[1]**2)))
    H = np.sum(Rrt*np.dot(Q,Rrt.transpose(1,0,2)),axis=0)
    V =p*np.exp(-H)
    return V


    
    
def computegradient(x1,x2,positions,D,center,num_of_agents,desv):
    N = num_of_agents
    # print(positions)
    constante = 2/(N*D**2)
    rtotal = np.zeros((N,2),dtype='d')
    gradest = np.zeros((N,2),dtype='d')
    # ro = ro - xo
    for k in range(int(N)):
        # Evalua positionx -centerx, positiony - centery.
        fradios = function(positions[k,0]-center[0],positions[k,1]-center[1],0,desv,0)  
        # print("soy fradios:",fradios)
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