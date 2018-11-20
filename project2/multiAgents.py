# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # (question 1)

        foodDist, ghostDist = [], []
        foodList = newFood.asList()
        ghostPos = successorGameState.getGhostPositions()

        # Current score:
        curScore = scoreEvaluationFunction(currentGameState)

        # Find the distance of all the foods to the pacman 
        foodDist = [util.manhattanDistance(newPos, food) for food in foodList]
        closestFood = min(foodDist) if len(foodDist) > 0 else 1

        # Find the distance of all the ghost to the pacman
        ghostDist = [util.manhattanDistance(newPos, ghost) for ghost in ghostPos]
        for dist in ghostDist:
            if dist < 2:
                return float("-inf")

        # Number of food left
        numFoodLeft = len(foodList)
        if not len(foodDist):
            return float("inf")

        score = 1*curScore + 10000000./numFoodLeft + 1./sum(ghostDist) + 100./closestFood

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # (question 2)
        return self.minimax(gameState)

    def minimax(self, gameState):
        if self.depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        best = Directions.STOP
        val = float("-inf")
        actions = gameState.getLegalActions(0)  # actions for pacman
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            v = self.minValue(nextState, self.depth, 1)
            if v > val:
                val = v
                best = action
        return best

    def maxValue(self, gameState, depth):
        # depth == 0 means we has reached the bottom of the tree and now we are "going back"
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        # initialize v as negative infinity
        v = float("-inf")
        actions = gameState.getLegalActions(0)  # get actions for pacman
        # get successor states
        successorStates = []
        for action in actions:
            successorStates.append(gameState.generateSuccessor(0, action))
        for successorState in successorStates:
            v = max(v, self.minValue(successorState, depth, 1))
        return v

    def minValue(self, gameState, depth, agentID):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = float("inf")
        actions = gameState.getLegalActions(agentID)
        successorStates = []
        for action in actions:
            successorStates.append(gameState.generateSuccessor(agentID, action))
        for state in successorStates:
            # If this is pacman, then we go to next depth level: min(v, self.maxValue(state, depth - 1))
            # If this is NOT pacman, meaning we have multiple ghosts,
            # then we need to calculate v for each of those ghosts: min(v, self.minValue(state, depth, agentID + 1))
            v = min(v, self.maxValue(state, depth - 1)) if agentID == gameState.getNumAgents() - 1 \
                else min(v, self.minValue(state, depth, agentID + 1))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # (question 3)
        return self.minimax(gameState)

    def minimax(self, gameState):
        if self.depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        val = float("-inf")
        alpha_minimax = -float("inf")
        beta_minimax = float("inf")
        best = Directions.STOP
        actions = gameState.getLegalActions(0)  # actions for pacman
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            v = self.minval(nextState, self.depth, 1, alpha_minimax, beta_minimax)
            alpha_minimax = max(v, alpha_minimax)
            if v > val:
                val = v
                best = action
        return best

    def maxval(self, gameState, depth, alpha, beta):
        if depth==0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        v = -float("inf")
        successorStates = []
        actions = gameState.getLegalActions(0)  # for pacman
        for action in actions:
            successorStates.append(gameState.generateSuccessor(0, action))
        for state in successorStates:
            v = max(v, self.minval(state, depth, 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def minval(self, gameState, depth, agentID, alpha, beta):
        if depth==0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        v = float("inf")
        successorStates = []
        actions = gameState.getLegalActions(agentID)  # for pacman
        for action in actions:
            successorStates.append(gameState.generateSuccessor(agentID, action))
        for state in successorStates:
            v = min(v, self.maxval(state, depth-1, alpha, beta)) if agentID == gameState.getNumAgents() - 1 \
                else min(v, self.minval(state, depth, agentID+1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState)

    def minimax(self, gameState):
        if self.depth==0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        val = float("-inf")
        best = Directions.STOP
        actions = gameState.getLegalActions(0)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            v = self.expValue(nextState, self.depth, 1)
            if v > val:
                val = v
                best = action
        return best

    def maxValue(self, state, depth):
        if self.depth==0 or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        actions = state.getLegalActions(0)
        v = float("inf") if len(actions) else self.evaluationFunction(state)
        successorStates = []
        for action in actions:
            successorStates.append(state.generateSuccessor(0, action))
        for state in successorStates:
            v = max(v, self.expValue(state, depth, 1))
        return v

    def expValue(self, state, depth, agentID):
        # if self.depth==0 or state.isLose() or state.isWin():
        #     return self.evaluationFunction(state)
        if depth == self.depth:
            return self.evaluationFunction(state)
        v = 0
        actions = state.getLegalActions(agentID)
        successorStates = []
        for action in actions:
            successorStates.append(state.generateSuccessor(agentID, action))
        for state in successorStates:
            v = self.maxValue(state, depth + 1) if agentID == state.getNumAgents() - 1 \
                            else self.expValue(state, depth, agentID+1)
            v += v / len(actions)
        return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # (question 5)

    newPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood().asList()
    newGhostPos = currentGameState.getGhostPositions()
    newGhostStates = currentGameState.getGhostStates()
    newCapsule = currentGameState.getCapsules()
    score = currentGameState.getScore()
    newScaredTimes = []
    for state in newGhostStates:
        newScaredTimes.append(state.scaredTimer)

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")

    # Find the distance of all the ghost to the pacman
    ghostDist = [util.manhattanDistance(newPos, ghost) for ghost in newGhostPos]
    for dist in ghostDist:
        if dist < 2:
            return float("-inf")

    # Find the distance of all the foods to the pacman
    foodDist = [util.manhattanDistance(newPos, food) for food in curFood]
    closestFood = min(foodDist) if len(foodDist) > 0 else 1

    # Number of food left
    numFoodLeft = len(curFood)
    if not len(foodDist):
        return float("inf")

    # Eat capsule
    eatCapsule = 100
    capDist = []
    if newCapsule:
        for capsule in newCapsule:
            capDist.append(util.manhattanDistance(newPos, capsule))
        eatCapsule = 10.0 / min(capDist)

    result = 10*score + 10*sum(newScaredTimes) + 10./numFoodLeft + 1./closestFood + 0.1*eatCapsule
    return result

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
