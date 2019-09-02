# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    c = problem.getStartState()
    exploredState = []
    exploredState.append(start)
    states = util.Stack()
    stateTuple = (start, [])
    states.push(stateTuple)
    while not states.isEmpty() and not problem.isGoalState(c):
        state, actions = states.pop()
        exploredState.append(state)
        successor = problem.getSuccessors(state)
        for i in successor:
            coordinates = i[0]
            if not coordinates in exploredState:
                c = i[0]
                direction = i[1]
                states.push((coordinates, actions + [direction]))
    return actions + [direction]
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    exploredState = []
    exploredState.append(start)
    states = util.Queue()
    stateTuple = (start, [])
    states.push(stateTuple)
    while not states.isEmpty():
        state, action = states.pop()
        if problem.isGoalState(state):
            return action
        successor = problem.getSuccessors(state)
        for i in successor:
            coordiates = i[0]
            if not coordiates in exploredState:
                direction = i[1]
                exploredState.append(coordiates)
                states.push((coordiates, action + [direction]))
    return action
    util.raiseNotDefined()



def iterativeDeepeningSearch(problem):
    """
    Start with depth = 0.
    Search the deepest nodes in the search tree first up to a given depth.
    If solution not found, increment depth limit and start over.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    state = problem.getStartState()
    cost = {}
    actions = []
    cost[state] = 0
    limit = 0
    answer = idsRecursive(state, cost, actions, problem, limit, 0)
    limit = 1
    while not answer:
        state = problem.getStartState()
        cost = {}
        actions = []
        cost[state] = 0
        limit = limit + 1
        answer = idsRecursive(state, cost, actions, problem, limit, 0)
    return actions


def idsRecursive(state, cost, actions, problem, limit, depth):
    if limit < depth:
        return False
    if problem.isGoalState(state):
        return True
    for successor in problem.getSuccessors(state):
        next_state, action, step_cost = successor[0], successor[1], successor[2]
        if (next_state not in cost) or (cost[state] + step_cost < cost[next_state]):
            cost[next_state] = cost[state] + step_cost
            actions.append(action)
            if idsRecursive(next_state, cost, actions, problem, limit, depth + 1):
                return True
            actions.pop()
    return False


def getActionList(node):
    """
    Get the sequence of actions from the
    beginning of the problem to the current node
    """
    actions = []
    while node["prev"]:
        actions.append(node["action"])
        node = node["prev"]
    return actions[::-1]

def search(problem, init, expand):
    """
    Perform a search algorithm with shared logic,
    it calls init function to initialise data structure for open list,
    and calls expand function to expand the current node and update the
    data structure accordingly
    """
    closed = set()
    opened = init()
    while not opened.isEmpty():
        node = opened.pop()
        if problem.isGoalState(node["state"]):
            return getActionList(node)
        if node["state"] not in closed:
            closed.add(node["state"])
            expand(node, opened)
    return None


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    exploredState = []
    states = util.PriorityQueue()
    states.push((start, []), 0)
    while not states.isEmpty():
        state, actions = states.pop()
        if problem.isGoalState(state):
            return actions
        if state not in exploredState:
            successors = problem.getSuccessors(state)
            for succ in successors:
                coordinates = succ[0]
                if coordinates not in exploredState:
                    directions = succ[1]
                    newCost = actions + [directions]
                    states.push((coordinates, actions + [directions]), problem.getCostOfActions(newCost))
        exploredState.append(state)
    return actions
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def waStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    start = problem.getStartState()
    exploredState = []
    states = util.PriorityQueue()
    states.push((start, []), nullHeuristic(start, problem))
    # ncost = 0
    while not states.isEmpty():
        state, actions = states.pop()
        # print(state)
        if problem.isGoalState(state):
            return actions
        if state not in exploredState:
            successor = problem.getSuccessors(state)
            for suc in successor:
                coordinates = suc[0]
                if coordinates not in exploredState:
                    directions = suc[1]
                    naction = actions + [directions]
                    c = problem.getCostOfActions(naction)
                    h1 = heuristic(coordinates, problem)
                    h2 = heuristic(coordinates, problem)
                    ncost = c + h1 + h2
                    states.push((coordinates, actions + [directions]), ncost)
        exploredState.append(state)
    return actions

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    start = problem.getStartState()
    exploredState = []
    states = util.PriorityQueue()
    states.push((start, []), nullHeuristic(start, problem))
    nCost = 0
    while not states.isEmpty():
        state, actions = states.pop()
        if problem.isGoalState(state):
            return actions
        if state not in exploredState:
            successors = problem.getSuccessors(state)
            for succ in successors:
                coordinates = succ[0]
                if coordinates not in exploredState:
                    directions = succ[1]
                    nActions = actions + [directions]
                    nCost = problem.getCostOfActions(nActions) + heuristic(coordinates, problem)
                    states.push((coordinates, actions + [directions]), nCost)
        exploredState.append(state)
    return actions
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
wastar = waStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch