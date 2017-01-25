import numpy as np

class MDP:
    def __init__(self, T, S, R, A):
        # Transition probabilities
        # Form: (start_state, action, end_state)
        self.T = np.array(T)
        # State space
        # Integer number of states
        self.S = np.array(S)
        # Reward space
        # Form: vector, rewards for each state
        self.R = np.array(R)
        # Action space
        # integer, number of possible actions
        self.A = np.array(A)
