# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:56:42 2023

@author: A_209
"""

import cv2
import numpy as np
from scipy import ndimage
from deteccionPlaca import detectarPlaca
from reconocimientoCaracteres import clasificadorCaracteres, get_hog, escalar
from utils import saveImage

img = cv2.imread("placa1.jpg")
placa = detectarPlaca(img)

I = cv2.cvtColor(placa, cv2.COLOR_RGB2GRAY)

u, _ = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)
mascara = np.uint8(255 * (I<u))

output = cv2.connectedComponentsWithStats(mascara, 4, cv2.CC_STAT_AREA)
cantidadObj = output[0]
labels= output[1]
stats = output[2]

for i in range(1, cantidadObj):
    if stats[i,4]<stats[:,4].mean()/10:
        labels=labels-i*(labels==i)
        
mascara=np.uint8(255*(labels>0))
kernel=np.ones((3,3),np.uint8)
mascara=np.uint8(255*ndimage.binary_fill_holes(cv2.dilate(mascara,kernel)))

contours,_=cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
caracteres=[]
orden=[]
placa2=placa.copy()

for cnt in contours:
    #contiene los caracteres de la placa y podríamos hace run imshow(caracteres[3])
    x,y,w,h=cv2.boundingRect(cnt)
    caracteres.append(placa[y:y+h,x:x+w,:])
    orden.append(x)
    cv2.rectangle(placa2,(x,y),(x+w,y+h),(0,0,255),1)
    
caracteresOrdenados = [x for _,x in sorted(zip(orden,caracteres))]

# saveImage("bounds", placa2)

palabrasKnn=""
palabrasSVM=""
hog=get_hog()
knn,SVM=clasificadorCaracteres()

posiblesEtiq=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

posiblesEtiq=np.array(posiblesEtiq)

caracteresPlaca=[]

for i in caracteresOrdenados:
    m,n,_=i.shape
    escalado=escalar(i,m,n)
    caracteresPlaca.append(escalado)
    caracteristicas = np.array(hog.compute(escalado))
    caracteristicas = caracteristicas.reshape(1, -1)  # Asegurarte de que tenga la forma correcta
    palabrasKnn += posiblesEtiq[knn.predict(caracteristicas)][0][0]
    palabrasSVM += posiblesEtiq[SVM.predict(caracteristicas)][0][0]

print("El clasificador Knn dice: "+palabrasKnn)    
print("El clasificador SVM dice: "+palabrasSVM)    


m,n,_= img.shape
#cv2.putText(img,"La placa es: "+palabrasSVM,(10,300),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,255,255),1)


# saveImage("carro", img)
# saveImage("placa", placa2)

# cv2.imshow("carro",img)
# cv2.imshow("placa",placa2)














