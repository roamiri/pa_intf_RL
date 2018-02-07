import numpy as np


class Agent:

    def __init__(self, row, col):
        self.p_index = 0
        self.s_index = 0
        self.Q = np.zeros((row, col))
        self.power = 0.0
        self.next_s_index = 0
        self.state = 0

    def set_power(self, pp):
        self.power = pp

    def set_s_index(self, idx):
        self.s_index = idx

    def set_next_s_index(self, idx):
        self.next_s_index = idx
