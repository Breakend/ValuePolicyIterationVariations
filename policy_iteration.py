import numpy as np
from operator import itemgetter

def policy_iteration(mdp, gamma = 0.9, optimal_value = None):
    # Initialization
    V = [0]*mdp.S
    pol = [1]*mdp.S
    old_V = V
    vs = []
    n_iter = 0
    while True:
        # Policy evaluation
        while True:
            delta = 0.0
            old_V = V
            for s in range(mdp.S):
                v = V[s]
                V[s] = sum(mdp.T[s, pol[s], k] * (mdp.R[k] + gamma * V[k]) for k in range(mdp.S))
                delta = max(delta, abs(v-V[s]))

            if(delta < 0.001):
                break
        # Policy improvement
        policy_stable = True
        for s in range(mdp.S):
            old_action = pol[s]
            possibilities = [sum(mdp.T[s,a,k] *(mdp.R[k] + gamma * V[k]) for k in range(mdp.S)) for a in range(mdp.A)]
            pol[s] = max(enumerate(possibilities), key=itemgetter(1))[0]
            if(old_action != pol[s]):
                policy_stable = False
        if policy_stable:
            print "Converged in %d iterations" % (n_iter)
            return V, pol, n_iter, vs
        vs.append(np.linalg.norm(V - optimal_value))
        n_iter += 1

def policy_iteration_by_inversion(mdp, gamma = 0.9, optimal_value = None):
    # Initialization
    V = [0]*mdp.S
    pol = [1]*mdp.S
    vs = []
    old_V = V
    n_iter = 0
    while True:
        # Policy evaluation
        T_new = np.zeros((mdp.S, mdp.S))

        for s in range(mdp.S):
            T_new[s,:] = mdp.T[s, pol[s], :]

        V = np.array(np.linalg.inv(np.eye(mdp.S) - gamma*T_new).dot(mdp.R))

        vs.append(np.linalg.norm(V - optimal_value))

        # Policy improvement
        policy_stable = True
        for s in range(mdp.S):
            old_action = pol[s]
            possibilities = [sum(mdp.T[s,a,k] *(mdp.R[k] + gamma * V[k]) for k in range(mdp.S)) for a in range(mdp.A)]
            pol[s] = max(enumerate(possibilities), key=itemgetter(1))[0]
            if(old_action != pol[s]):
                policy_stable = False
        if policy_stable:
            print "Converged in %d iterations" % (n_iter)
            return V, pol, n_iter, vs
        n_iter += 1


def modified_policy_iteration(mdp, epsilon = 0.001, gamma = 0.9, m = 20, optimal_value = None):
    # Initialization
    V = [0]*mdp.S
    pol = [1]*mdp.S
    vs = []
    n_iter = 0

    while True:
        u = []
        u.append([])

        # Policy improvement
        policy_stable = True
        for s in range(mdp.S):
            old_action = pol[s]
            possibilities = [sum(mdp.T[s,a,k] *(mdp.R[k] + gamma * V[k]) for k in range(mdp.S)) for a in range(mdp.A)]
            pol[s] = max(enumerate(possibilities), key=itemgetter(1))[0]
            u[0].append(max(possibilities))

        u.append([])
        # Policy evaluation
        i = 0
        if np.linalg.norm(np.asarray(u[0]) - np.asarray(V)) < epsilon/(2*gamma):
            print "Converged in %d iterations" % (n_iter)
            return V, pol, n_iter, vs

        while True:
            if (i == m):
                V = u[i]
                n_iter+=1
                break
            else:
                for s in range(mdp.S):
                    u[i+1].append(sum(mdp.T[s, pol[s], k] * (mdp.R[k] + gamma * u[i][k]) for k in range(mdp.S)))
                    u.append([])

            if np.linalg.norm(np.asarray(u[i+1]) - np.asarray(u[i])) < epsilon/(2*gamma):
                V = u[i]
                n_iter+=1
                break
            else:
                i+=1

        vs.append(np.linalg.norm(V - optimal_value))
        n_iter += 1
