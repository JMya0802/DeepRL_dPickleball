import numpy as np
import cv2 as cv

def preprocess(img,method,freq,range1): 
    
    """
    Parameters:
    img -> the pickleball screen page
    method -> what the color you want to change: default as cv.COLOR_BGR2LAB is better, other parameter include: cv.COLOR_BGR2HSV and cv.COLOR_BGR2YCrCb
    freq -> the pixel value appear freq times
    range1 -> take the range value (since the original observation is compressed all the pixel have some difference)
    
    
    cvt_img convert the img to your method you use and then convert it to the gray image
    using histogram to calculate the number of pixel 
    filter > freq times pixels using numpy where
    take the neigbour pixel value the range is the pixel values are selected +- range1
    using np.unique to filter repeat pixel value
    using np.isin to mask the value are contain in observation
    convert the mask to dark otherwise convert to white

    
    """
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