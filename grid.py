# OK, the following are not as widely useful utilities as some of the other
# functions here, but they do show up wherever we have 2D grids: Wumpus and
# Vacuum worlds, TicTacToe and Checkers, and markov decision Processes.
# __________________________________________________________________________
import math

orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, -1)


def turn_left(heading):
    return turn_heading(heading, +1)


def distance(a, b):
    """The distance between two (x, y) points."""
    return math.hypot((a[0] - b[0]), (a[1] - b[1]))


def distance2(a, b):
    "The square of the distance between two (x, y) points."
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


