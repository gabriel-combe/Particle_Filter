from State import State
from numpy.random import randn

class NoisySensor(object):
    def __init__(self, std_noise: float =1.) -> None:
        self.std = std_noise

    # Add gaussian noise to ground truth position
    def sense(self, pos: State) -> State:
        return pos + State(x = randn() * self.std, y = randn() * self.std)