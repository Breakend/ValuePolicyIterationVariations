"""
Taken from https://github.com/aimacode/aima-python/blob/master/mdp.py

TODO: swap this out for our own MDP class

Markov Decision Processes (Chapter 17)
First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration
and policy_iteration algorithms."""

from grid import orientations, turn_right, turn_left

import operator
import random
import numpy as np


class MDP:

    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs.  We also keep track of the possible states,
    terminal states, and actions for each state. [page 646]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = {}

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (probability, result-state) pairs."""
        raise NotImplementedError

    def possible_actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):

    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = tuple(map(operator.add, state, direction))
        return state1 if state1 in self.states else state


sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None,  -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])

