# -*- coding: utf-8 -*-
"""
Created on Tue May 13 22:41:56 2014

@author: Pedro
"""

import numpy as np
import cv2

def findPart(n):
    return np.cumsum(1.0/n*np.ones(n-1))
    
def quantize(signal, partitions):
    indices = []
    for datum in signal:
        index = sum(~(partitions>=datum))
        indices.append(index)
    return np.array(indices)
 
        
def compCSD(img,colorNum):

    tab = {'256': np.array([(1,32),(4,8),(16,4),(16,4),(16,4)]),
           '128': np.array([(1,16),(4,4),(8,4),(8,4),(8,4)]),
            '64': np.array([(1,8),(4,4),(4,4),(8,2),(8,1)]),
            '32': np.array([(1,8),(4,4),(4,1),(4,1)])}
    
    csdQuantPart = np.array([6, 20, 60, 110])
    regionLevels = np.array([1, 25, 20, 35, 35, 140]);
    regionLevelsC = np.cumsum(regionLevels)
    csdHistQuant = np.array([0.000000001, 0.037, 0.08, 0.195, 0.32])
    
    #colorNum = 128
    tabP = np.concatenate((np.zeros(1),np.cumsum(np.prod(tab[str(colorNum)],axis=1))))
        
    heightI, widthI, channelsI = img.shape
    
    p = np.floor(np.log2(np.sqrt(heightI*widthI))-7.5)
    if p:
        scale = np.power(2,p)
        imgS = cv2.resize(img,None,fx=1.0/scale,fy=1.0/scale)
    else:
        imgS = img

    heightS, widthS, channelsS = imgS.shape
    
    # RGB to HMMD conversion
    imgHSV = cv2.cvtColor(imgS,cv2.COLOR_BGR2HSV)
    Hue_plane = imgHSV[:,:,0]
    
    Max_plane = cv2.max(imgS[:,:,0],cv2.max(imgS[:,:,1],imgS[:,:,2]))
    
    Min_plane = cv2.min(imgS[:,:,0],cv2.min(imgS[:,:,1],imgS[:,:,2]))
    
    Diff_plane = Max_plane - Min_plane
    
    Sum_plane = (Max_plane.astype(float)+Min_plane)/2.0
    
    D_quant = quantize(Diff_plane.ravel(),csdQuantPart)
    H_plane_R = Hue_plane.ravel()
    S_plane_R = Sum_plane.ravel()
    H_quant = np.zeros(H_plane_R.shape)
    S_quant = np.zeros(S_plane_R.shape)
    for q in range(5):
        ind = np.nonzero(D_quant==q)
        H_part = findPart(tab[str(colorNum)][q,0])*255
        H_quant[ind] = quantize(H_plane_R[ind],H_part)
        S_part = findPart(tab[str(colorNum)][q,1])*255
        S_quant[ind] = quantize(S_plane_R[ind],S_part)
    
    C = tabP[D_quant] + H_quant*tab[str(colorNum)][D_quant,1] + S_quant;
    imgC = np.uint8(C.reshape(heightS,widthS))
    
    CSD_hist = np.zeros(colorNum)
    for i in range(heightS-8):
        for j in range(widthS-8):
            bins = np.unique(imgC[i:i+8,j:j+8])
            CSD_hist[bins]=CSD_hist[bins]+1
    
    CSD_hist = CSD_hist/np.sum(CSD_hist)
    #print CSD_hist
    
    CSD_hist_quant = np.zeros(colorNum)
    CSD_hist_aux = quantize(CSD_hist,csdHistQuant)
    for q in np.arange(1,5):
        indQ = np.nonzero(CSD_hist_aux==q)
        if indQ:
            quantQ = np.cumsum((csdHistQuant[q]-csdHistQuant[q-1])/regionLevels[q]*np.ones(regionLevels[q]-1))+csdHistQuant[q-1]
            CSD_hist_quant[indQ] = quantize(CSD_hist[indQ],quantQ) + regionLevelsC[q-1]
    return CSD_hist_quant
