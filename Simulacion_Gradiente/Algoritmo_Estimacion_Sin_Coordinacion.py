# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:55:26 2021

@author: tutir
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 01:38:19 2021

@author: tutir
"""

# La finalidad es ir comentando y descomentando segun interes, la capacidad que 
# tiene es sobretodo evaluar unicamente el avance de la formacion sin la implementacion
# del algoritmo de coordinacion.

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import random
import time
from time import sleep
import pygame
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors
import numpy as np
from numpy import linalg as la


# Para evitar que salgan a cada rato warnings.
import warnings
warnings.filterwarnings('ignore')

# Necesario para las graficas
X = np.arange(-1200.,2400.,10)
Y = np.arange(-1200.,2400.,10)
Xgrid, Ygrid = np.meshgrid(X, Y)

# Parametros

# Fijos
angulo = np.linspace(0, 2*np.pi, 1000) 
Dimension = 2
counter = 0;
error = np.zeros((2,1),dtype='d')
positions = np.zeros((robots,2),dtype='d')
 
# Gaussiana
p = 1
rota = 0
c = np.array([600,600])
# Una sola
desv = np.array([1000/np.sqrt(2),1000/np.sqrt(2)])
# Multi
# desv = np.array([1000/np.sqrt(2),500/np.sqrt(2)])

# Algoritmos
D = 60
epsilon = 20
robots = 3    #Es la N en el documento.
fun = 0

# Puntos de donde parte mi centro.
x1 = 0
x2 = 0
ck = np.array([x1,x2])             #Ck en cada momento sera el centro.

# Para poder implementar el plano se utilizo esta función.
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
        Rrt = np.dot(R,Rot.transpose(1,0,2))        
    else:
        Rrt = np.dot(R,Rot)
   
    #V = p*np.exp(-(Rrt[0]**2/(2*s[0]**2)+Rrt[1]**2/(2*s[1]**2)))
    H = np.sum(Rrt*np.dot(Q,Rrt.transpose(1,0,2)),axis=0)
    V =p*np.exp(-H)
    return V

# Para actualizar el paso, estaba dando error multiplicar las matrices asi que
# se hizo una solucion "hack" con indexaciones
def dis_gaussiana(x1,x2,th,desv,p,flag):
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
    Rrt = np.dot(R,Rot)
    H = Rrt*Q*Rrt
    Hnew[0,0] = H[0,0]
    Hnew[0,1] = H[0,1]
    Hnew[1,0] = H[1,0]
    Hnew[1,1] = H[1,1]
    Hfin = np.sum(H)
    f = p*np.exp(-np.sum(H))
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
    G[1,0]= (-2*Res_y[1,1]-Res_x[0,1]-Res_x[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) #Primera derivada.
    G[0,0]= (-2*Res_x[0,0]-Res_y[0,1]-Res_y[1,0])*np.exp(-(Hnew[0,0] + Hnew[1,1])) 
    # print("meh: ",G[1,0])
    # print("H: ",Hnew)
    # print("H: ",Res_y)
    if flag==0:
        return f
    else:
        return G  

# Matrices necesarias para la grafica 3D.
zz = np.zeros((len(X),len(Y)),dtype='d')
zz1 = np.zeros((len(X),len(Y)),dtype='d')
zz2 = np.zeros((len(X),len(Y)),dtype='d')
zz3 = np.zeros((len(X),len(Y)),dtype='d')
for i in range(len(X)):
  for j in range(len(Y)):
      # Un solo maximo
      zz1[i,j] = dis_gaussiana(Xgrid[i,j]-c[0],Ygrid[i,j]-c[1],rota,desv,p,0) 
      # Multiples gaussianas.
    # zz1[i,j] = dis_gaussiana(Xgrid[i,j]-c[0],Ygrid[i,j]-c[1],rota-np.pi/6,desv,p,0)
    # zz2[i,j] = dis_gaussiana(Xgrid[i,j],Ygrid[i,j]-1200,rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0) 
    # zz3[i,j] = dis_gaussiana(Xgrid[i,j]-1200,Ygrid[i,j],rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0) 
    
zz = zz1
#Prueba para ver que me sale la misma grafica que en matlab, simplemente es la grafica en tres dimensiones.
# Z = gausianillas(Xgrid, Ygrid,c,desv,0,1)

# Disposicion de los vehiculos en torno a la formacion circular (No se aplica el
# algoritmo de coordinacion en este caso).
def placerobots(x1,x2,robots):
    co = np.array([x1,x2]) 
    N = robots
    a = 2*np.pi/N
    ro = np.array([x1 + D,x2])
    rtotal = np.zeros((N,2),dtype='d')
    ro = ro - co
    for k in range(N):
        a = 2*np.pi*(k+1)/(N)
        cphi = round(np.cos(a),8)
        sphi = round(np.sin(a),8)
        Rotacion = np.array([[cphi, -sphi] , [sphi, cphi]])
        ri = np.dot(Rotacion,np.transpose(ro)) 
        rtotal[k,0] = ri[0] + co[0]
        rtotal[k,1] = ri[1] + co[1]
    return rtotal
    
  
def computegradient(x1,x2,robots,positions,desv,c,rota,p):
    co = np.array([x1,x2]) 
    N = robots
    constante = 2/(N*D**2)
    grad_est_i = np.zeros((N,2),dtype='d')
    for k in range(N):
        # Una sola gaussiana.
         f1 = dis_gaussiana(positions[k,0]-c[0],positions[k,1]-c[1],rota,desv,p,0) 
        # Multifuentes.
        # f1 = dis_gaussiana(positions[k,0]-c[0],positions[k,1]-c[1],rota-np.pi/6,desv,p,0) 
        # f2 = dis_gaussiana(positions[k,0],positions[k,1]-1200,rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0) 
        # f3 = dis_gaussiana(positions[k,0]-1200,positions[k,1],rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0)
        f_c = f1
        grad_est_i[k,0] = f_c*(positions[k,0]-x1)
        grad_est_i[k,1] = f_c*(positions[k,1]-x2)
    grad_est = constante*sum(grad_est_i)
    return grad_est

positions = placerobots(x1,x2,robots)

# Gaussiana vista desde arriba.
fig, ax = plt.subplots(figsize=(7,5))
plt.figure(1)
plt.axis('equal')
cp = ax.contour(Xgrid, Ygrid, zz, 30, cmap = 'RdGy') # Contorno.
ax.clabel(cp, inline=True, fontsize=12)
ax.set_title('Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')

counter = 0
runsim = True
while(runsim):
    
    # Dibujo del circulo
    # plt.figure(1)
    # h = ck[0]   # Circulito
    # k = ck[1]
    # Xcir = D * np.cos(angulo) + h
    # Ycir = D * np.sin(angulo) + k
    # ax.plot(Xcir,Ycir, color='r')
    
    # Calculo del gradiente estimado
     grad_est=computegradient(ck[0],ck[1],robots,positions,desv,c,rota,p) 
     
    # Dibujar los robots en el circulo y el gradiente estimado
    # ax.plot(positions[:robots,0],positions[:robots,1],'o')
    # plt.quiver(ck[0],ck[1],grad_est[0],grad_est[1],color='r')
    
    # Avance de los vehiculos mediante ascenso de gradiente
    positions = positions + epsilon*grad_est
    
    # El gradiente real
    G = dis_gaussiana(ck[0]-c[0],ck[1]-c[1],rota,desv,p,1)
    
    # Avance del centro de la formacion mediante el ascenso de gradiente.
    ck = ck + epsilon*grad_est
    
    # Valor de la funcion cuando es una sola.
    fun1 = dis_gaussiana(ck[0]-c[0],ck[1]-c[1],rota,desv,p,0)
    
    # Valor cuando son multiples 
    # fun1 = dis_gaussiana(ck[0]-c[0],ck[1]-c[1],rota-np.pi/6,desv,p,0)
    # fun2 = dis_gaussiana(ck[0],ck[1]-1200,rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0)
    # fun3 = dis_gaussiana(ck[0]-1200,ck[1],rota,[300/np.sqrt(2),300/np.sqrt(2)],0.9*p,0)
    
    # Comprobar avance y el valor de la funcion.
    fun = fun1
    # print("soy fun: ",fun)

    # Esto de abajo son pruebas para obtener las figuras 2.4 y 2.5. El valor del
    # modulo se vio incrementado dado que se realizo en este caso un estudio cualitativo.
    # en el caso del avance dicho modulo no se ve modificado.
    # plt.quiver(ck[0],ck[1],k*G[0,0],k*G[1,0],color='b',scale=1/120000000*D, units='xy')
    # Q = plt.quiver(X, Y, U, V, units="width")
    # plt.quiver(ck[0],ck[1],k*grad_est[0],k*grad_est[1],color='r',scale=1/80000000*D, units='xy')
    # # ax.axis([-D-0.5*D,D+0.5*D,-D-0.5*D,D+0.5*D]) 
    # ax.legend([r'$D$',r'$N$',r'$\nabla{f(x)}$',r'$\widehat{\nabla}{f(x)}$'])
 
    # Error sin coordinacion.
    # if counter%20==0:     
    #     fig = plt.figure(5)
    #     ax = fig.add_subplot(111)
    #     error[0] = grad_est[0]-G[0,0]
    #     error[1] = grad_est[1]-G[1,0]
    #     plt.plot(counter,10000*la.norm(error),'.g',markersize=1)
    #     ax.set_ylabel(r'${||}\widehat{\nabla}{f-}\nabla{f}{||}$')
    #     ax.set_xlabel('N (iteraciones)')
    elif fun >= 0.999:
        break
    counter = counter + 1  
    
     
# Funcion utilizada en 3D.
 
# fig = plt.figure(2)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Plot the surface.
# surf = ax.plot_surface(Xgrid, Ygrid, zz, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x,y)')