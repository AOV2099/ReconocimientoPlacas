# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:00:19 2023

@author: A_209
"""

import os 
import cv2

def saveImage(filename, img):

    carpeta_borrar = "borrar"
    if not os.path.exists(carpeta_borrar):
        os.makedirs(carpeta_borrar)

    ruta_archivo = os.path.join(carpeta_borrar, filename + ".jpg")
    cv2.imwrite(ruta_archivo, img)
    print(ruta_archivo)
    
