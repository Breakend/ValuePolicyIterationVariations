import numpy as np

class ValueIteration:

    def __init__(self, mdp, gauss_seidel=False):
        self.mdp = mdp
        self.gauss_seidel = gauss_seidel

    def run(self, theta = 0.001):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)

        iteration = 0
        sweeps = 0
        while True:
            delta = 0
            if self.gauss_seidel:
                # as per slides http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2016/04/02-MarkovDecisionProcess.pdf
                # simply allow updates to the current state-value space
                Vold = V
            else:
                Vold = V.copy()

            for s in range(self.mdp.S):
                iteration += 1
                V[s] = max([self.mdp.T[s,a,:].dot(self.mdp.R + gamma * Vold[s]) for a in range(self.mdp.A)])
                # Sutton, p.90 2nd edition draft (Jan. 2017)
                delta = max(delta, abs(Vold[s] - V[s]))
            sweeps += 1
            if delta < theta:
                break

        print("Converged in %d iterations (%d sweeps)" % (iteration, sweeps))

        pi = get_policy(V)

        return pi

    def get_policy(V):
        pi = {}
        for s in range(self.mdp.S):
            possibilities = [self.mdp.T[s,a,:].dot(self.mdp.R + gamma * V[s]) for a in range(self.mdp.A)]
            pi[s] = max(enumerate(possibilities), key=itemgetter(1))

        return pi

class GaussSeidelValueIteration(ValueIteration):

    def __init__(self, mdp):
        super().__init__(mdp, gauss_seidel=True)

class JacobiValueIteration(ValueIteration):

    def run(self, theta = 0.001):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)

        iteration = 0
        sweeps = 0
        while True:
            delta = 0
            if self.gauss_seidel:
                # as per slides http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2016/04/02-MarkovDecisionProcess.pdf
                # simply allow updates to the current state-value space
                Vold = V
            else:
                Vold = V.copy()

            for s in range(self.mdp.S):
                iteration += 1
                #TODO: is this right?
                # As in https://tspace.library.utoronto.ca/bitstream/1807/24381/6/Shlakhter_Oleksandr_201003_PhD_thesis.pdf
                masked_transition = np.ma.array(self.mdp.T[s,a,:], mask=False)
                masked_transition.mask[s] = True
                V[s] = max([masked_transition.dot(self.mdp.R + gamma * Vold[s]) / (1 - gamma * T[s][a][s]) for a in range(self.mdp.A)])
                # Sutton, p.90 2nd edition draft (Jan. 2017)
                delta = max(delta, abs(Vold[s] - V[s]))
            sweeps += 1
            if delta < theta:
                break

        print("Converged in %d iterations (%d sweeps)" % (iteration, sweeps))

        pi = get_policy(V)

        return pi

class GaussSeidelJacobiValueIteration(JacobiValueIteration):
    def __init__(self, mdp):
        super().__init__(mdp, gauss_seidel=True)

class PrioritizedSweepingValueIteration(ValueIteration):

    def run(self, theta=0.001, max_iterations=1000):
        # as per slides http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2016/04/02-MarkovDecisionProcess.pdf
        # and http://www.jmlr.org/papers/volume6/wingate05a/wingate05a.pdf
        V = np.zeros(self.mdp.S)
        H = np.zeros(self.mdp.S)
        iteration = 0
        while True:
            iterations += 1
            v = V[s]
            s = np.argmax(H)
            V[s] = max([self.mdp.T[s,a,:].dot(self.mdp.R + gamma * v) for a in range(self.mdp.A)])
            H[s] = abs(v - V[s])
            delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        print("Converged in %d iterations" % (iteration))

        pi = get_policy(V)

        return pi
