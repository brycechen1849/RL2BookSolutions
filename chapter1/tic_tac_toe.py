#######################################################################
# Copyright (C)                                                       #
# 2018 - 2020 Bryce Chen (brycechen1849@gmail.com)                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        # The board is represented by an N*N array, with
        # 1 representing the first moving player's positions, and
        # -1 representing the other player's positions.
        # Empty positions are denoted as 0.
        self.data = np.zeros(shape=(BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        self.hash_val = None
        self.winner = None
        self.end = None

    # Each state has a unique ID represented by it's hashed value.
    # The Mechanism:
    # Represent the state as a trinomial number, with data[0,0] being the largest digit
    # and data[N,N] being the right most digit.
    # （-1, 0, 1） +1  ->  (0, 1, 2)
    @property
    def hash(self) -> int:
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
            return self.hash_val

    # check whether a player has won the game, or it's a tie
    @property
    def is_end(self) -> bool:
        if self.end is not None:
            return self.end

        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data, axis=0))
        # check columns
        for j in range(BOARD_COLS):
            results.append(np.sum(self.data, axis=1))
        # check diagonals
        trace = 0
        reversed_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reversed_trace = self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reversed_trace)

        # analysis results, check if there is a winner
        for result in results:
            if np.sum(result) == 3:
                self.winner = 1
                self.end = True
                return self.end
            elif np.sum(result) == -3:
                self.winner = -1
                self.end = True
                return self.end

        # check if it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # The up described cases are not met, thus the game is not finished yet.
        self.end = False
        return self.end

    # execute the move
    @classmethod
    def next_state(cls, current_state, i, j, symbol):
        next_state = State()
        next_state.data = np.copy(current_state.data)
        next_state.data[i, j] = symbol
        # Have to refresh the state
        return next_state

    @classmethod
    def print_state(cls, state) -> None:
        for i in range(BOARD_ROWS):
            print('----------')
            out = '|'
            for j in range(BOARD_COLS):
                if state.data[i, j] == 1:
                    token = '*'
                elif state.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('---------')

    # Recursive implementation of state searching
    @classmethod
    def get_all_states_impl(cls, current_state, current_symbol, all_states):
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if current_state.data[i, j] == 0:
                    new_state = cls.next_state(current_state, i, j, current_symbol)
                    new_hash = new_state.hash
                    # DFS Search possible states (with memory)
                    if new_hash not in all_states:
                        all_states[new_hash] = (new_state, new_state.is_end)
                        if not new_state.is_end:
                            cls.get_all_states_impl(new_state, -current_symbol, all_states)

    # gather all possible state configurations
    @classmethod
    def get_all_states(cls):
        current_symbol = 1
        current_state = State()
        all_states = dict()
        all_states[current_state.hash] = (current_state, current_state.is_end)
        cls.get_all_states_impl(current_state, current_symbol, all_states)
        return all_states


# all possible board configurations
all_states = State.get_all_states()

# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    # update value estimation
    def backup(self):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (
                self.estimations[states[i + 1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

class Judge:
    # @player1: the player who will move first, chessman denoted as 1.
    # @player2: the other player with chessman denoted as -1.

    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state: object = False) -> int:
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)

        if print_state:
            State.print_state(current_state)

        while True:
            player = next(alternator)
            i, j, symbol = player.act()
            next_state_hash = State.next_state(current_state, i, j, symbol).hash
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)

            if print_state:
                State.print_state(current_state)
            if is_end:
                return current_state.winner
