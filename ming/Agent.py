import torch 
import random
import torch.nn as nn

class Simple_Random_Agent:
    def __init__(self):
        pass
        
    def play(self):
        return [random.randint(0,2),random.randint(0,2),random.randint(0,2)]


class Actor(nn.Module):
    pass