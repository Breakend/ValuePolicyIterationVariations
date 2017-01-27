import numpy as np
from operator import itemgetter

def policy_iteration(mdp, gamma = 0.9, epsilon = 0.01, optimal_value = None):
    # Initialization
    V = [0]*mdp.S
    pol = [0]*mdp.S
    old_V = V
    vs = []
    n_iter = 0
    v_iter = 0
    while True:
        # Policy evaluation
        while True:
            delta = 0.0
            old_V = V
            for s in range(mdp.S):
                v = V[s]
                V[s] = sum(mdp.T[s, pol[s], k] * (mdp.R[k] + gamma * V[k]) for k in range(mdp.S))
                delta = max(delta, abs(v-V[s]))
                vs.append(np.linalg.norm(V - optimal_value))
                v_iter+=1
            if(delta < epsilon):
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
            print "Converged in %d iterations, %d evaluations" % (n_iter, v_iter)
            return V, pol, n_iter, vs
        n_iter += 1

def policy_iteration_by_inversion(mdp, gamma = 0.9, optimal_value = None):
    # Initialization
    V = [0]*mdp.S
    pol = [1]*mdp.S
    vs = []
    old_V = V
    n_iter = 0
    v_iter = 0
    while True:
        # Policy evaluation
        T_new = np.zeros((mdp.S, mdp.S))

        for s in range(mdp.S):
            T_new[s,:] = mdp.T[s, pol[s], :]

        V = np.array(np.linalg.inv(np.eye(mdp.S) - gamma*T_new).dot(mdp.R))
        v_iter+=1
        vs.append(np.linalg.norm(V - optimal_value))

        # Policy improvement
        policy_stable = True
        for s in range(mdp.S):
            old_action = pol[s]
            possibilities = [sum(mdp.T[s,a,k] *(mdp.R[k] + gamma * V[k]) for k in range(mdp.S)) for a in range(mdp.A)]
            pol[s] = max(enumerate(possibilities), key=itemgetter(1))[0]
            if(old_action != pol[s]):
                policy_stable = False
        n_iter += 1
        if policy_stable:
            print "Converged in %d iterations, %d evaluations" % (n_iter, v_iter)
            return V, pol, n_iter, vs

def modified_policy_iteration(mdp, gamma = 0.9, epsilon = 1.0, m = 0, optimal_value=None):
    # Initialization
    V = [0]*mdp.S
    pol = [1]*mdp.S
    old_V = V
    vs = []
    n_iter = 0
    v_iter = 0
    while True:
        # Policy improvement
        policy_stable = True
        for s in range(mdp.S):
            old_action = pol[s]
            possibilities = [sum(mdp.T[s,a,k] *(mdp.R[k] + gamma * V[k]) for k in range(mdp.S)) for a in range(mdp.A)]
            pol[s] = max(enumerate(possibilities), key=itemgetter(1))[0]
            if(old_action != pol[s]):
                policy_stable = False

        if policy_stable:
            print "Converged in %d iterations, %d evaluations" % (n_iter, v_iter)
            return V, pol, n_iter, vs

        i = 0
        # Policy evaluation
        while i<m:
            delta = 0.0
            old_V = V
            for s in range(mdp.S):
                v = V[s]
                V[s] = sum(mdp.T[s, pol[s], k] *
                       (mdp.R[k] + gamma * V[k])
                       for k in range(mdp.S))
                delta = max(delta, abs(v-V[s]))
                vs.append(np.linalg.norm(V - optimal_value))
                v_iter += 1
            i += 1

            if delta < epsilon/(2*gamma):
                break
        n_iter += 1
