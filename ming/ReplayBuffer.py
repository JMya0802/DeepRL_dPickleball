from collections import namedtuple,deque

class ReplayBuffer:
    def __init__(self):
        self.Transitions = namedtuple("Episodes",("state","next_state","action","reward","done"))
        self.memory = deque()

    def push(self,*args):
        self.memory.append(self.Transitions(*args))

    def empty(self):
        self.memory.clear()