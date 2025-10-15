import numpy as np
import cv2 as cv

def preprocess(img,method,freq,range1): 
    
    cvt_img = cv.cvtColor(img,method) 
    gray_img = cv.cvtColor(cvt_img,cv.COLOR_BGR2GRAY)
    
    
    hist, bin = np.histogram(gray_img,bins=256,range=(0,256))
    
    mask_condition1 = np.where(hist > freq)[0] 

    mask_condition2 = []

    for v in mask_condition1:
        mask_condition2.extend(range(v - range1, v + range1))  

    mask_condition2 = np.unique(mask_condition2)
    mask_condition2 = mask_condition2[(mask_condition2 >= 0) & (mask_condition2 <= 255)]
    
    mask = np.isin(gray_img,mask_condition2) 
    
    mask_img = np.where(mask,0,255).astype(np.uint8)
    
    return mask_img