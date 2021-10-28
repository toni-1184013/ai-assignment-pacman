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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    mystack = util.Stack()
    startNode = (problem.getStartState(), '', 0, [])
    mystack.push(startNode)
    visited = set()
    while mystack :
        node = mystack.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                mystack.push(newNode)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    '''
    Run A* search algorithm based on type of problem specified and heuristic.

    Parameters:
        problem : type of problem to be solved (position/food search problem)

    Returns:
        actions(list) : list of action for optimum solution   
    '''

    #COMP90054 Task 1, Implement your A Star search algorithm here
    """Search the node that has the lowest combined cost and heuristic first."""
    mypq = util.PriorityQueue()
    startNode = (problem.getStartState(), '', 0, [], heuristic(problem.getStartState(), problem))
    mypq.update(startNode, 0+heuristic(problem.getStartState(), problem))
    visited = set()
    while not mypq.isEmpty() :
        node = mypq.pop()
        state, action, cost, path, heur = node
       
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost+succCost , path + [(state, action)], heuristic(succState, problem))
                mypq.update(newNode, cost+succCost+heuristic(succState, problem) )
    actions = [action[1] for action in path]
    del actions[0]
    return actions

        
def recursivebfs(problem, heuristic=nullHeuristic) :
    '''
    Run recursive best first search algorithm based on type of problem specified and heuristic.

    Parameters:
        problem : type of problem to be solved (position/food search problem)

    Returns:
        actions(list) : list of action for optimum solution   
    '''

    #COMP90054 Task 2, Implement your Recursive Best First Search algorithm here
    startNode = (problem.getStartState(), '', 0, [], heuristic(problem.getStartState(), problem),heuristic(problem.getStartState(), problem))
    #print("start node: ",problem.getStartState(),"-",heuristic(problem.getStartState(), problem))
    actions, f = rbfs(problem, startNode, heuristic)
    return actions


def rbfs(problem, node, heuristic, fLimit=999999999) :
    '''
    Recursive function for expanding search node.

    Parameters:
        problem : type of problem to be solved (position/food search problem)
        node    : node to be expanded
        heuristic   : selected heuristic (distance or cost)
        fLimit      : maximum value of heuristic

    Returns:
        actions(list) : list of action for optimum solution
    '''

    state, action, cost, path, heur, f = node

    #print("current node: ",state," ",action," ",cost," ",path," ",heur," ",f)
    #print("current node: ",state,"- f: ",f,"vs flimit: ",fLimit)
    #print("========================================================")

    if problem.isGoalState(state) :
        #print("GOAL!!")
        path = path + [(state, action)]
        actions = [action[1] for action in path]
        del actions[0]
        print(actions)
        return [actions, f]

    succNodes = problem.expand(state)

    if len(succNodes) <= 0 :
        return [False, 999999999]

    successors = util.PriorityQueue()
    for succNode in succNodes :
        succState, succAction, succCost = succNode
        newF = max(cost + succCost + heuristic(succState, problem), f)
        #print(newF,"vs", f)
        newNode = (succState, succAction, cost+succCost , path + [(state, action)], heuristic(succState, problem), newF)
        successors.update(newNode, newF )
        #print("expand: ",succState,"-",succAction,"g=",cost,"d=",succCost,"heur=",heuristic(succState, problem),",f=",newF)
    #print("========================================================")
    while True:

        best = successors.pop()
        bestF = best[5]

        #print("prev curr state recur: ",best[0] , " f limit: ",fLimit,"f after: ",bestF)
        if bestF > fLimit:
            #print("return!")
            successors.update(best, bestF)
            return [False, bestF]

        if (not successors.isEmpty() ) :
            alternative = successors.pop()
            alternativeF = alternative[5]
            updatedAlternative = (alternative[0],alternative[1],alternative[2],alternative[3],alternative[4],alternativeF)
            successors.update(updatedAlternative, alternativeF)
        else:
            alternativeF = 999999999
        
        result, bestF = rbfs(problem, best, heuristic, min(fLimit,alternativeF))
        #print("curr state recur: ",best[0] , " f before: ",best[5],"f after: ",bestF)
        updatedBest = (best[0],best[1],best[2],best[3],best[4],bestF)
        successors.update(updatedBest, bestF)


        if (result):
            return [result, bestF]

    return []

    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
rebfs = recursivebfs
