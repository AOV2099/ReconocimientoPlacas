# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:13:59 2023

@author: A_209
"""

import cv2
import numpy as np
from scipy import ndimage
import os 

#img = cv2.imread("placa1.jpg")
#img = cv2.imread("placa1.jpg")



def detectarPlaca(img):
    
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    u,_ = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)
    mascara = np.uint8(255*(I>u))
    output = cv2.connectedComponentsWithStats(mascara, 4, cv2.CC_STAT_AREA)
    
    cantidadObj = output[0]
    labels = output[1]
    stats = output[2]
    
    maskObj = []
    maskConv = []
    diferenciaArea = []
    
    for i in range (1, cantidadObj):#el convexo devuielve una imagen rectangular del area que le pasamos
        if stats[i,4] > stats[:,4].mean():
            mascara=ndimage.binary_fill_holes(labels==i)
            mascara = np.uint8(255*mascara)
            maskObj.append(mascara)
            #calculo del covexhull
            contours,_ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            hull = cv2.convexHull(cnt)
            puntosConvex=hull[:,0,:]
            m,n= mascara.shape
            ar=np.zeros((m,n))
            mascaraConvex=np.uint8(255*cv2.fillConvexPoly(ar, puntosConvex,1))
            maskConv.append(mascaraConvex)
            #cmparacion area de CONVEXHULL con el objeto
            areaObj = np.sum(mascara)/255
            areaConv = np.sum(mascaraConvex)/255
            diferenciaArea.append(np.abs(areaObj -areaConv))#la diferencia de areas es la más pequeña del array
              
    maskPlaca = maskConv[np.argmin(diferenciaArea)]
  
    
    # correccion perspectiva
    
    vertices=cv2.goodFeaturesToTrack(maskPlaca,4,0.01,10)
    x=vertices[:,0,0]
    y=vertices[:,0,1]
    vertices=vertices[:,0,:]
    xo=np.sort(x)
    yo=np.sort(y)
    
    xn=np.zeros((1,4))
    yn=np.zeros((1,4))
    n=(np.max(xo)-np.min(xo))
    m=(np.max(yo)-np.min(yo))
    
    xn=(x==xo[2])*n+(x==xo[3])*n
    
    yn=(y==yo[2])*m+(y==yo[3])*m
    verticesN=np.zeros((4,2))
    verticesN[:,0]=xn
    verticesN[:,1]=yn
    
    vertices=np.int64(vertices)
    verticesN=np.int64(verticesN)
    
    h,_=cv2.findHomography(vertices,verticesN)
    
    placa=cv2.warpPerspective(img,h,(np.max(verticesN[:,0]),(np.max(verticesN[:,1]))))
    return placa

# img = cv2.imread("placa3.jpg")
# placa = detectarPlaca(img)

# carpeta_borrar = "borrar"
# if not os.path.exists(carpeta_borrar):
#     os.makedirs(carpeta_borrar)

# ruta_archivo = os.path.join(carpeta_borrar, "placa_borrar.jpg")
# cv2.imwrite(ruta_archivo, placa)
# print(ruta_archivo)
# cv2.imwrite(carpeta_borrar, placa)







