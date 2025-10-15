from collections import deque, namedtuple

import numpy as np
import cv2 as cv

import torch
import torch.nn as nn

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


class FrameStack:
    def __init__(self,frame,max_len=5):
        
        self.frame = preprocess(frame)
        self.deque = deque(maxlen=max_len)
    
    def stack_frame(self):
        self.deque.append(torch.tensor(self.frame,dtype=torch.float16))
        
        while len(self.deque) < self.deque.maxlen:
            self.deque.append(self.deque[-1])
        
        frame_stack = torch.stack(list(self.deque))
        
        return frame_stack
    


class ReplayBuffer:
    def __init__(self):
        self.Transitions = namedtuple("Episodes",("state","next_state","action","reward","done"))
        self.memory = deque()

    def push(self,*args):
        self.memory.append(self.Transitions(*args))

    def empty(self):
        self.memory.clear()
        

class Simple_Random_Agent:
    def __init__(self):
        pass
        
    def play(self):
        return [np.random.randint(0,2),np.random.randint(0,2),np.random.randint(0,2)]
    


class ActorModel(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.conv1 = nn.
        