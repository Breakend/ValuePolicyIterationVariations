

class ValueIteration:

    def __init__(self, mdp):
        self.mdp = mdp

    def run(self, theta = 0.001):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = { s : 0.0 for s in self.mdp.states }

        # Repeat
        # TODO: turn this into vector form ??
        iteration = 0
        while True:
            iteration += 1
            delta = 0
            for s in self.mdp.states:
                v = V[s]
                action_possibilities = []
                for a in self.mdp.possible_actions(s):
                    action_possibilities.append(self.mdp.R(s,a) + sum([p *self.mdp.gamma*V[s_prime]) for (p, s_prime) in self.mdp.T(s,a)]))
                V[s] = max(action_possibilities)

                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        print("Converged in %d iterations" % iteration)

        pi = {}
        for s in self.mdp.states:
            action_possibilities = []
            for a in self.mdp.possible_actions(s):
                action_possibilities.append((a, self.mdp.R(s,a) + sum([p * (self.mdp.gamma*V[s_prime]) for (p, s_prime) in self.mdp.T(s,a)])))
            pi[s] = argmax(action_possibilities)

        return pi

class GaussSeidel:

    def __init__(self, mdp):
        self.mdp = mdp

    def run(self, theta = 0.001):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = { s : 0.0 for s in self.mdp.states }

        # Repeat
        # TODO: turn this into vector form ??
        iteration = 0
        while True:
            iteration += 1
            delta = 0
            V_n1 = V.copy()
            for s in self.mdp.states:
                v = V[s]
                action_possibilities = []
                for a in self.mdp.possible_actions(s):
                    action_possibilities.append(self.mdp.R(s,a) + sum([p *self.mdp.gamma*V[s_prime]) for (p, s_prime) in self.mdp.T(s,a)]))
                V[s] = max(action_possibilities)

                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        print("Converged in %d iterations" % iteration)

        pi = {}
        for s in self.mdp.states:
            action_possibilities = []
            for a in self.mdp.possible_actions(s):
                action_possibilities.append((a, self.mdp.R(s,a) + sum([p * (self.mdp.gamma*V[s_prime]) for (p, s_prime) in self.mdp.T(s,a)])))
            pi[s] = argmax(action_possibilities)

        return pi

def argmax(iterable):
    return max(iterable, key=lambda x: x[1])[0]
