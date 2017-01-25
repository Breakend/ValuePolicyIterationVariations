import numpy as np


class ValueIteration:

    def __init__(self, mdp):
        self.mdp = mdp

    def run(self, theta = 0.001):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)

        iteration = 0
        while True:
            iteration += 1
            delta = 0
            for s in range(self.mdp.S):
                v = V[s]
                V[s] = max([self.mdp.T[s,a,:].dot(self.mdp.R + gamma * v) for a in range(self.mdp.A)])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        print("Converged in %d iterations" % iteration)

        pi = {}
        for s in range(self.mdp.S):
            v = V[s]
            possibilities = [self.mdp.T[s,a,:].dot(self.mdp.R + gamma * v) for a in range(self.mdp.A)]
            pi[s] = max(enumerate(possibilities), key=itemgetter(1))

        return pi



class GaussSeidelValueIteration(ValueIteration):

    def run(self, theta = 0.001):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)

        iteration = 0
        while True:
            iteration += 1
            delta = 0
            for s in range(self.mdp.S):
                v = V[s]
                V[s] = max([self.mdp.T[s,a,:].dot(self.mdp.R + gamma * v) for a in range(self.mdp.A)])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        print("Converged in %d iterations" % iteration)

        pi = {}
        for s in range(self.mdp.S):
            v = V[s]
            possibilities = [self.mdp.T[s,a,:].dot(self.mdp.R + gamma * v) for a in range(self.mdp.A)]
            pi[s] = max(enumerate(possibilities), key=itemgetter(1))

        return pi
