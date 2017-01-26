import numpy as np
from operator import itemgetter

class ValueIteration(object):

    def __init__(self, mdp, gauss_seidel=False):
        self.mdp = mdp
        self.gauss_seidel = gauss_seidel

    def run(self, theta = 0.001, gamma=.9, optimal_value=None):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)
        vs = []
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
                if optimal_value is not None:
                    vs.append(np.linalg.norm(V - optimal_value))
                iteration += 1
                v = Vold[s]
                # import pdb; pdb.set_trace()
                # V[s] = max([self.mdp.R[s] + gamma * self.mdp.T[s,a,:].dot( Vold) for a in range(self.mdp.A)])
                V[s] = max([sum(self.mdp.T[s,a,k] *(self.mdp.R[k] + gamma * Vold[k]) for k in range(self.mdp.S)) for a in range(self.mdp.A)])
                # Sutton, p.90 2nd edition draft (Jan. 2017)
                # import pdb; pdb.set_trace()
                delta = max(delta, abs(v - V[s]))
                # print(delta)
            sweeps += 1
            if delta < theta:
                break

        print("Converged in %d iterations (%d sweeps)" % (iteration, sweeps))

        pi = self.get_policy(V)

        return pi, V, vs

    def get_policy(self, V, gamma=0.9):
        pi = {}
        for s in range(self.mdp.S):
            possibilities = [sum(self.mdp.T[s,a,k] *(self.mdp.R[k] + gamma * V[k]) for k in range(self.mdp.S)) for a in range(self.mdp.A)]
            # possibilities = [self.mdp.T[s,a,:].dot(self.mdp.R + gamma * V[s]) for a in range(self.mdp.A)]
            # import pdb; pdb.set_trace()
            pi[s] = max(enumerate(possibilities), key=itemgetter(1))[0]

        return pi

class GaussSeidelValueIteration(ValueIteration):

    def __init__(self, mdp):
        super(GaussSeidelValueIteration, self).__init__(mdp, gauss_seidel=True)

class JacobiValueIteration(ValueIteration):

    def run(self, theta = 0.01, gamma=.9, optimal_value=None):
        # initialize array V arbitrarily
        # print("Jacobian")
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)
        vs = []

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
                if optimal_value is not None:
                    vs.append(np.linalg.norm(V - optimal_value))
                v = Vold[s]
                #TODO: is this right?
                # As in https://tspace.library.utoronto.ca/bitstream/1807/24381/6/Shlakhter_Oleksandr_201003_PhD_thesis.pdf
                possibilities = []

                for a in range(self.mdp.A):
                    # masked_transition = np.ma.array(self.mdp.T[s,a,:], mask=False)
                    # masked_transition.mask[s] = True
                    possibilities.append(sum(self.mdp.T[s,a,k] * (self.mdp.R[k] + gamma * Vold[k]) for k in range(self.mdp.S) if k != s) /  (1. - gamma * self.mdp.T[s][a][s]))
                    # possibilities.append(self.mdp.R[s] + gamma * np.ma.dot(masked_transition, Vold)  / (1. - gamma * self.mdp.T[s][a][s]) )
                V[s] = max(possibilities)

                # Sutton, p.90 2nd edition draft (Jan. 2017)
                delta = max(delta, abs(v - V[s]))
            sweeps += 1
            if delta < theta:
                break

        print("Converged in %d iterations (%d sweeps)" % (iteration, sweeps))

        pi = self.get_policy(V)

        return pi, V, vs

class GaussSeidelJacobiValueIteration(JacobiValueIteration):
    def __init__(self, mdp):
        super(GaussSeidelJacobiValueIteration, self).__init__(mdp, gauss_seidel=True)

class PrioritizedSweepingValueIteration(ValueIteration):

    def run(self, theta=0.001, gamma=.9, max_iterations= 3000, optimal_value = None):
        # as per slides http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2016/04/02-MarkovDecisionProcess.pdf
        # and http://www.jmlr.org/papers/volume6/wingate05a/wingate05a.pdf
        V = np.zeros(self.mdp.S)
        H = np.zeros(self.mdp.S)
        vs = []
        iterations = 0
        while iterations < max_iterations:
            #TODO: this is wrong right now, see http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node98.html
            # and also see http://aritter.github.io/courses/slides/mdp.pdf
            delta = 0
            iterations += 1
            if optimal_value is not None:
                vs.append(np.linalg.norm(V - optimal_value))
            s = np.argmax(H)
            Vold = V.copy()
            V[s] = max([self.mdp.R[s] + gamma * self.mdp.T[s,a,:].dot(Vold) for a in range(self.mdp.A)])
            H[s] = abs(Vold - V)
            import pdb; pdb.set_trace()
            delta = max(delta, abs(Vold[s] - V[s]))
            if delta < theta:
                break

        print("Converged in %d iterations" % (iterations))

        pi = self.get_policy(V)

        return pi, V, vs
