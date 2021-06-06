import numpy as np
import abc
import util
from game import Agent, Action


def eval_board_differences(board):
    '''
    evaluate the board - using the minus of the sum of differences between tiles value
    :param board: the board of the game: as a numpy array
    '''
    differences_sum = 0
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2]:
            differences_sum += abs(board[i][j] - board[i][j + 1])
    for i in [0, 1, 2]:
        for j in [0, 1, 2, 3]:
            differences_sum += abs(board[i][j] - board[i + 1][j])
    return -differences_sum


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def get_action(self, game_state):
        """
        get_action chooses among the best options according to the evaluation function.
        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best
        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.
        """
        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board

        return eval_board_differences(board)


def score_evaluation_function(current_game_state):
    """
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


def return_max_dict(moves_dict):
    '''
    return the item of the maximal value in a dictionary
    :param moves_dict: a dictionary of {move: move_value}
    :return: the move that has the maximum move value
    '''
    max_move = None
    max_value = -np.inf
    for move in moves_dict:
        if max_value <= moves_dict[move]:
            max_move = move
            max_value = moves_dict[move]
    return max_move


def return_max(successors_values):
    '''
    :param successors_values: list of tuple: (successor, value)
    :return: tuple with maximal value
    '''
    if len(successors_values) == 0:
        return -np.inf
    return max(successors_values)


def return_min(successors_values):
    '''
    :param successors_values: list of tuple: (successor, value)
    :return: tuple with maximal value
    '''
    if len(successors_values) == 0:
        return -np.inf
    return min(successors_values)


class MinmaxAgent(MultiAgentSearchAgent):

    def rec_minmax(self, depth, game_state):
        '''
        a recursive algorithm to calculate minimax algorithm
        :param depth: tracking our depth in the tree
        :param game_state: A stateof the game
        :param player: 0 - player, 1 - computer
        :return: value of the state according to minimax
        '''
        if depth == 0:
            return self.evaluation_function(game_state)
        # minimize
        if round(depth) != depth:
            return return_min([self.rec_minmax(depth - 0.5, game_state.generate_successor(1, move)) \
                               for move in game_state.get_legal_actions(1)])
        # maximize
        else:
            return return_max([self.rec_minmax(depth - 0.5, game_state.generate_successor(0, move)) \
                               for move in game_state.get_legal_actions(0)])

    def get_action(self, game_state):
        """
        Here are some method calls that might be useful when implementing minimax.
        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1
        Action.STOP:
            The stop direction, which is always legal
        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        moves = dict()
        for move in game_state.get_legal_actions(0):
            move_value = self.rec_minmax(self.depth - 0.5, game_state.generate_successor(0, move))
            if move_value >= -np.inf:
                moves[move] = move_value
        if len(moves) == 0:
            return Action.STOP
        move_tp_return = return_max_dict(moves)
        return move_tp_return


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """

    def rec_alpha_beta(self, depth, game_state, alpha, beta):
        '''
        a recursive function to calculate alpha beta pruning
        :param depth: tracking our depth in the tree
        :param game_state: The state of the game
        :param player: 0 - player, 1 - computer
        :return: value of state according to alpha beta
        '''
        if depth == 0:
            state_value = self.evaluation_function(game_state)
            return state_value
        # minimize
        if round(depth) != depth:  # depth = something.5
            min_value = np.inf
            for move in game_state.get_legal_actions(1):
                move_value = self.rec_alpha_beta(depth - 0.5, game_state.generate_successor(1, move), alpha, beta)
                if move_value < min_value:
                    min_value = move_value
                if min_value <= alpha:
                    break  # not to comntinue iterating through childs (other moves)
                beta = min_value  # update beta: we found new minimal value in childs
            return min_value

        # maximize
        else:
            max_value = -np.inf
            for move in game_state.get_legal_actions(0):
                move_value = self.rec_alpha_beta(depth - 0.5,
                                                 game_state.generate_successor(0, move), alpha, beta)
                if move_value > max_value:
                    max_value = move_value
                if max_value >= beta:
                    break  # nor to comntinue iterating through childs (other moves)
                alpha = max_value  # update alpha: we found new maximum value in childs
            return max_value

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        moves = dict()
        for move in game_state.get_legal_actions(0):
            move_value = self.rec_alpha_beta(self.depth - 0.5, game_state.generate_successor(0, move), alpha=(-np.inf), beta=(np.inf))
            if move_value >= -np.inf:
                moves[move] = move_value
        if len(moves) == 0:  # didn't found a legal move
            return Action.STOP
        move_tp_return = return_max_dict(moves)
        return move_tp_return


def return_avg(successors_values):
    '''
    :param successors_values: list of tuple: (successor, value)
    :return: average value
    '''
    if len(successors_values) == 0:
        return -np.inf
    return np.average(successors_values)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def rec_expectation(self, depth, game_state):
        '''
        a recursive function to calculatr the ecxpected moves values
        :param depth: tracking our depth in the tree
        :param game_state: The state of the game
        '''
        if depth == 0:
            return self.evaluation_function(game_state)
        # computer_move -- return average
        if round(depth) != depth:  # depth = something.5
            return return_avg([self.rec_expectation(depth - 0.5, game_state.generate_successor(1, move)) \
                               for move in game_state.get_legal_actions(1)])
        # agent_move
        else:  # return max
            return return_max([self.rec_expectation(depth - 0.5, game_state.generate_successor(0, move)) \
                               for move in game_state.get_legal_actions(0)])

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        moves = dict()
        for move in game_state.get_legal_actions(0):
            move_value = self.rec_expectation(self.depth - 0.5, game_state.generate_successor(0, move))
            if move_value >= -np.inf:
                moves[move] = move_value
        if len(moves) == 0:
            return Action.STOP
        move_tp_return = return_max_dict(moves)
        return move_tp_return


def weighted_board(board, empty_tiles):
    '''
    calculate wight of the board according to a specific matrixthat gives
    more values to one corner over the other TODO
    '''
    LEFT_UP_CORNER = 5000 * np.matrix(
        [[0.135, 0.121, 0.102, 0.0999], [0.0997, 0.088, 0.076, 0.0724],
         [0.0606, 0.0562, 0.0371, 0.0161], [0.0125, 0.0099, 0.0057, 0.0033]])
    return np.matrix.sum(empty_tiles * LEFT_UP_CORNER * board)


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION:
    We calculate the minus of the sum of differences in the boarda wight o
    Add to that wight we give to the board with a specific matrix that gives more value to some tiles
    and add to that aconstant C multiply by the number of freetiles in the board
    """
    num_of_empty_tiles = len(current_game_state.get_empty_tiles())
    board_differences = eval_board_differences(current_game_state._board)
    board_weight = weighted_board(current_game_state._board,
                                  num_of_empty_tiles + 1)
    return board_weight + board_differences * 500


# Abbreviation
better = better_evaluation_function
