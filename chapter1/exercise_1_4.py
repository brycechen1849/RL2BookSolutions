import tic_tac_toe
from tic_tac_toe import Player, Judge, State
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

matplotlib.use('Agg')


# Change some settings of the original version
class CuriousPlayer(Player):

    def backup(self):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state = states[i]

            # td_error = self.greedy[i] * (self.estimations[states[i + 1]] - self.estimations[state])
            # The original version use greedy[i] indicating if it's a greedy action.
            # we can just leave it out and directly update the estimations.
            td_error = (self.estimations[states[i + 1]] - self.estimations[state])
            self.estimations[state] += self.step_size * td_error


def train(epochs=10000, print_every_n=500, ep1=0.1, ep2=0.1):
    exp_number = int(ep1 * 100 + ep2 * 10)
    player1 = CuriousPlayer(epsilon=ep1, exp_number=exp_number)
    player2 = CuriousPlayer(epsilon=ep2, exp_number=exp_number)
    judge = Judge(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    player1_win_rate_history = np.ndarray(shape=(epochs,))
    player2_win_rate_history = np.ndarray(shape=(epochs,))
    for i in range(1, epochs + 1):
        winner = judge.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('[Exp#%2d] Epoch %d, 1 win rate: %.02f, 2 win rate: %.02f' % (
                exp_number, i, player1_win / i, player2_win / i))
        # Here I use i/epochs to make a step_size decay from 0.1 slowly to 0.
        player1.step_size = 0.1 * (i / epochs)
        player2.step_size = 0.1 * (i / epochs)
        player1.backup()
        player2.backup()
        judge.reset()
        player1_win_rate_history[i - 1] = player1_win / i
        player2_win_rate_history[i - 1] = player2_win / i

    player1.save_policy()
    player2.save_policy()
    return player1_win_rate_history, player2_win_rate_history


def compete(turns, ep1, ep2):
    exp_number = int(ep1 * 100 + ep2 * 10)
    player1 = CuriousPlayer(epsilon=ep1, exp_number=exp_number)
    player2 = CuriousPlayer(epsilon=ep2, exp_number=exp_number)
    player1.load_policy()
    player2.load_policy()
    judge = Judge(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judge.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judge.reset()
    print('[Exp#%02d] %d turns, player 1 win %.05f, player 2 win %.05f' % (
        exp_number, turns, player1_win / turns, player2_win / turns))
    return player1_win / turns, player2_win / turns


def ex14():
    training = True
    if training:
        epochs = 10000
        print_every_n = 1000
        p1_win_rate, p2_win_rate = train(epochs=epochs, print_every_n=print_every_n, ep1=0.1, ep2=0.1)
        p1_winrate_history = p1_win_rate
        p2_winrate_history = p2_win_rate

        np.save("../data/exercies_1_4_p1_winrate_train.npy", p1_winrate_history)
        np.save("../data/exercies_1_4_p2_winrate_train.npy", p2_winrate_history)

    testing = True
    if testing:
        # testing phase
        turns = 2000
        p1_win_rate, p2_win_rate = compete(turns=turns, ep1=0.1, ep2=0.1)
        np.save("../data/exercies_1_4_p1_winrate_test.npy", p1_win_rate)
        np.save("../data/exercies_1_4_p2_winrate_test.npy", p2_win_rate)

    p1_win_rate = np.load("../data/exercies_1_4_p1_winrate_test.npy")
    p2_win_rate = np.load("../data/exercies_1_4_p2_winrate_test.npy")

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.4))

    # ===============
    # First subplot
    # ===============
    # set up the axes for the second plot

    print("Done")


if __name__ == '__main__':
    ex14()
