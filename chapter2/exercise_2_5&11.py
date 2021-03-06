#######################################################################
# Copyright (C)                                                       #
# Description : Simulation code for exercise 2.5                      #
# 2018 - 2020 Bryce Chen (brycechen1849@gmail.com)                    #
# Date created: 2020/09/26                                            #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
# #####################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

matplotlib.use('Agg')


class NonstationaryBandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    # real reward for each action
    def reset(self):
        # np.random.rand (normal distribution on a ndarray)
        # Parameters ----------
        # d0, d1, ..., dn : int, optional
        #   The dimensions of the returned array, must be non-negative.
        #   If no argument is given a single Python float is returned

        # self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_true = np.zeros(shape=(self.k,)) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial
        # number of chosen times for each action
        self.action_count = np.zeros(self.k)
        self.time = 0

    # Get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        elif self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                             self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)

            # Why not just directly use argmax here?
            # If there are more than one options that are simultaneously max,
            # we still want to randomly choose among these options.
            # Same reason for the following 2 if branch.
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        elif self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        else:
            q_best = np.max(self.q_estimation)
            return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):

        #  Nonstationary Bandit
        self.q_true += np.random.normal(loc=0, scale=0.01, size=(self.k,))
        self.best_action = np.argmax(self.q_true)

        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1

        # This is the incremental computation for average reward talked in the book.
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            # update estimation using gradient ascend
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            # The name is not estimation actually, it should be preference as described in the book
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward


# The experiment group that corresponds to a figure.
def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    # It's recommend that we save raw results of each parameter setting of the experiment:
    # + in case of being unexpectedly interrupted during the experiments, it would be possible to resume the work

    return mean_best_action_counts, mean_rewards


def exercise_2_6(runs=2000, time=10000):
    epsilon = 0.1
    step_size = 0.1
    bandits = [NonstationaryBandit(epsilon=epsilon, sample_averages=True),
               NonstationaryBandit(epsilon=epsilon, step_size=step_size)]

    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.title("Rewards of Sample Avg. VS Const alpha on non-stationary Bandits")
    plt.plot(rewards[0], label='epsilon = %.02f' % epsilon)
    plt.plot(rewards[1], label='alpha = %.02f' % step_size)

    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Selection of Sample Avg. VS Const alpha on non-stationary Bandits")
    plt.plot(best_action_counts[0], label='epsilon = %.02f' % epsilon)
    plt.plot(best_action_counts[1], label='alpha = %.02f' % step_size)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/exercise_2_5.png')
    plt.close()


def exercise_2_11(runs=100, time=20000):
    new = False
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: NonstationaryBandit(epsilon=epsilon, sample_averages=True),
                  lambda alpha: NonstationaryBandit(epsilon=0, gradient=True, step_size=alpha,
                                                    gradient_baseline=True),
                  lambda coef: NonstationaryBandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: NonstationaryBandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    if new:
        _, average_rewards = simulate(runs, time, bandits)
        np.save("../data/exercise_2_11_R.npy", average_rewards)
    else:
        average_rewards = np.load("../data/exercise_2_11_R.npy")

    # use latest 1000 items moving average as measurement of performance.

    # def moving_average(x, w):
    #     for p in range(x.shape[0]):
    #         for i in range(w, 0, -1):
    #             x[p, w + i - 1] = np.mean(x[p, i:i + w])
    #     return  x[:,w:]
    # rewards = moving_average(average_rewards, 1000)

    rewards = np.mean(average_rewards[:, time // 2:], axis=1)
    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i + l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('../images/exercise_2_11_Non-stationary.png')
    plt.close()


if __name__ == '__main__':
    # exercise_2_6()
    exercise_2_11()
