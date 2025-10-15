from collections import deque
import torch


class FrameStack:
    def __init__(self,max_len=5):
        self.deque = deque(maxlen=max_len)
    
    def stack_fram(self,frame):
        self.deque.append(torch.tensor(frame,dtype=torch.float16))
        
        while len(self.deque) < self.deque.maxlen:
            self.deque.append(self.deque[-1])
        
        frame_stack = torch.stack(list(self.deque))
        
        return frame_stack
    
    