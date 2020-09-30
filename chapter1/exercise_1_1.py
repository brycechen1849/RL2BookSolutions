import tic_tac_toe
from tic_tac_toe import Player, Judge
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

matplotlib.use('Agg')


def train(epochs=10000, print_every_n=500, ep1=0.1, ep2=0.1):
    exp_number = int(ep1 * 100 + ep2 * 10)
    player1 = Player(epsilon=ep1, exp_number=exp_number)
    player2 = Player(epsilon=ep2, exp_number=exp_number)
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
        player1.backup()
        player2.backup()
        judge.reset()
        player1_win_rate_history[i-1] = player1_win / i
        player2_win_rate_history[i-1] = player2_win / i

    player1.save_policy()
    player2.save_policy()
    return player1_win_rate_history, player2_win_rate_history


def compete(turns, ep1, ep2):
    exp_number = int(ep1 * 100 + ep2 * 10)
    player1 = Player(epsilon=ep1, exp_number=exp_number)
    player2 = Player(epsilon=ep2, exp_number=exp_number)
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


def ex11():
    # Make the epsilon grid
    ep1_list = np.arange(10) / 10
    ep2_list = np.arange(10) / 10
    x_grid, y_grid = np.meshgrid(ep1_list, ep2_list)
    p1_win_rate_grid = np.zeros_like(x_grid)
    p2_win_rate_grid = np.zeros_like(x_grid)
    epochs = 100000
    print_every_n = 1000

    p1_winrate_history = np.ndarray(shape=(len(ep1_list), len(ep2_list), epochs))
    p2_winrate_history = np.ndarray(shape=(len(ep1_list), len(ep2_list), epochs))

    training = False
    if training:
        for x in np.arange(10):
            for y in np.arange(10):
                p1_win_rate, p2_win_rate = train(epochs=epochs, print_every_n=print_every_n, ep1=ep1_list[x],
                                                 ep2=ep2_list[y])
                p1_winrate_history[x, y] = p1_win_rate
                p1_winrate_history[x, y] = p2_win_rate

        np.save("../data/exercies_1_1_p1_winrate_train.npy", p1_winrate_history)
        np.save("../data/exercies_1_1_p2_winrate_train.npy", p2_winrate_history)

    testing = False
    if testing:
        # testing phase
        turns = 2000
        for x in np.arange(10):
            for y in np.arange(10):
                p1_win_rate, p2_win_rate = compete(turns=turns, ep1=ep1_list[x], ep2=ep2_list[y])
                p1_win_rate_grid[x, y] = p1_win_rate
                p2_win_rate_grid[x, y] = p2_win_rate
        np.save("../data/exercies_1_1_p1_winrate_test.npy", p1_win_rate_grid)
        np.save("../data/exercies_1_1_p2_winrate_test.npy", p2_win_rate_grid)

    p1_win_rate_grid = np.load("../data/exercies_1_1_p1_winrate_test.npy")
    p2_win_rate_grid = np.load("../data/exercies_1_1_p2_winrate_test.npy")

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.4))

    # ===============
    # First subplot
    # ===============
    # set up the axes for the second plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    X, Y, Z = x_grid, y_grid, p1_win_rate_grid
    surf = ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax1.set_zlim(0.0, 1.0)
    fig.colorbar(surf, shrink=0.4, aspect=15)
    ax1.set_xlabel("epsilon of player 1")
    ax1.set_ylabel("epsilon of player 2")
    ax1.set_title("Wining rate of Player 1")
    ax1.view_init(20, 120)

    # ===============
    # First subplot
    # ===============
    # set up the axes for the second plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    X, Y, Z = x_grid, y_grid, p2_win_rate_grid
    surf = ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax2.set_zlim(0.0, 1.0)
    fig.colorbar(surf, shrink=0.4, aspect=15)
    ax1.set_ylabel("epsilon of player 1")
    ax1.set_ylabel("epsilon of player 2")
    ax2.set_title("Wining rate of Player 2")
    ax2.view_init(20, -70)

    plt.savefig("../images/exercise_1_1.png")

    # ===============
    # Second fig's 1st subplot
    # ===============
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(22, 8))

    ax3 = fig.add_subplot(1, 3, 1)
    im = ax3.imshow(p1_win_rate_grid)

    # We want to show all ticks...
    ax3.set_xticks(np.arange(10))
    ax3.set_xticklabels(np.arange(10) / 10)

    ax3.set_yticks(np.arange(10))
    ax3.set_yticklabels(np.arange(10) / 10)

    # ... and label them with the respective list entries
    ax3.set_xlabel("epsilon of player 1")
    ax3.set_ylabel("epsilon of player 2")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ep1_list)):
        for j in range(len(ep2_list)):
            text = ax3.text(j, i, "%.3f" % p1_win_rate_grid[i, j],
                            ha="center", va="center", color="black")

    ax3.xaxis.tick_top()
    ax3.set_title("Wining rate of Player 1")
    # ===============
    # Second fig's 1nd subplot
    # ===============
    ax4 = fig.add_subplot(1, 3, 2)
    im = ax4.imshow(p2_win_rate_grid)

    # We want to show all ticks...
    ax4.set_xticks(np.arange(10))
    ax4.set_xticklabels(np.arange(10) / 10)

    ax4.set_yticks(np.arange(10))
    ax4.set_yticklabels(np.arange(10) / 10)

    # ... and label them with the respective list entries
    ax4.set_xlabel("epsilon of player 1")
    ax4.set_ylabel("epsilon of player 2")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ep1_list)):
        for j in range(len(ep2_list)):
            text = ax4.text(j, i, p2_win_rate_grid[i, j],
                            ha="center", va="center", color="black")

    ax4.xaxis.tick_top()
    ax4.set_title("Wining rate of Player 2")
    # ===============
    # Second fig's 3rd subplot
    # ===============
    ax5 = fig.add_subplot(1, 3, 3)
    tie_gird = np.ones_like(p1_win_rate_grid) - p1_win_rate_grid - p2_win_rate_grid
    im = ax5.imshow(tie_gird)

    # We want to show all ticks...
    ax5.set_xticks(np.arange(10))
    ax5.set_xticklabels(np.arange(10) / 10)

    ax5.set_yticks(np.arange(10))
    ax5.set_yticklabels(np.arange(10) / 10)

    # ... and label them with the respective list entries
    ax5.set_xlabel("epsilon of player 1")
    ax5.set_ylabel("epsilon of player 2")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ep1_list)):
        for j in range(len(ep2_list)):
            text = ax5.text(j, i, "%.3f" % tie_gird[i, j],
                            ha="center", va="center", color="black")

    ax5.xaxis.tick_top()
    ax5.set_title("Percentage of games ending as a tie")

    plt.savefig("../images/exercise_1_1_Grid.png")

    print("Done")


if __name__ == '__main__':
    ex11()
