#######################################################################
# Copyright (C)                                                       #
# Description : Simulation code for exercise 2.5                      #
# 2018 - 2020 Bryce Chen (brycechen1849@gmail.com)                    #
# Date created: 2020/09/27                                            #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
# #####################################################################

# This is another method implemented with the concept of single iteration multi runs,
# which utilized a lot of vector computation tools in numpy,
# thus greatly reduced the time elapsed by the computation.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from tqdm import trange

matplotlib.use('Agg')


def exercise_2_2():
    # The iteration counts for a single experiment
    iterations = 10000
    # Total experiment counts (used the word 'task' in the book).
    runs = 2000
    # The test bed size (action space).
    k_arms = 10
    # The exploration probability.
    epsilon = 0.1
    # The alpha used in constant step-size setting.
    step_size = 0.1

    new_experiment = True

    if new_experiment:
        np.random.seed(0)
        start_time = time.time()
        best_action_counts_sa, rewards_sa = single_iteration_multi_run(iterations, runs, k_arms, epsilon=epsilon)
        end_time = time.time()
        print("SIMR sample-average time elapsed: ", end_time - start_time)

        np.save("../data/exercise_2_5_SIMR_epsilon_0-1_B.npy", best_action_counts_sa)
        np.save("../data/exercise_2_5_SIMR_epsilon_0-1_R.npy", rewards_sa)

        np.random.seed(0)
        start_time = time.time()
        best_action_counts_const, rewards_const = single_iteration_multi_run(iterations, runs, k_arms,
                                                                             step_size=step_size,
                                                                             epsilon=epsilon)
        end_time = time.time()
        print("SIMR constant step size time elapsed: ", end_time - start_time)

        np.save("../data/exercise_2_5_SIMR_const_a_0-1_B.npy", best_action_counts_const)
        np.save("../data/exercise_2_5_SIMR_const_a_0-1_R.npy", rewards_const)

    else:
        best_action_counts_const = np.load("../data/exercise_2_5_SIMR_const_a_0-1_B.npy")
        rewards_const = np.load("../data/exercise_2_5_SIMR_const_a_0-1_R.npy")
        best_action_counts_sa = np.load("../data/exercise_2_5_SIMR_epsilon_0-1_B.npy")
        rewards_sa = np.load("../data/exercise_2_5_SIMR_epsilon_0-1_R.npy")

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title("Rewards of Sample Avg. VS Const alpha on non-stationary Bandits")
    plt.plot(rewards_sa, label='sample average with epsilon = %.02f' % epsilon)
    plt.plot(rewards_const, label='constant alpha = %.02f epsilon = %.02f' % (step_size, epsilon))

    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Selection of Sample Avg. VS Const alpha on non-stationary Bandits")
    plt.plot(best_action_counts_sa, label='sample average with epsilon = %.02f' % epsilon)
    plt.plot(best_action_counts_const, label='constant alpha = %.02f ,epsilon = %.02f' % (step_size, epsilon))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('../images/exercise_2_5_SIMR.png')
    plt.close()


# This is a improved version of the original code that go through a single run at a time
def single_iteration_multi_run(iterations=100, runs=20, k_arms=10, epsilon=None, step_size=None):
    # Init the setting. Q is the estimate of q*.
    Q = np.zeros(shape=(runs, k_arms))
    # The underlying (ground-truth) q*(a) vector.
    q = np.zeros(shape=(runs, k_arms))

    R = np.zeros(shape=(iterations, runs))
    # B stands for the Best actions
    B = np.zeros(shape=(iterations, runs))

    N = np.zeros(shape=(runs, k_arms))

    # run indices [0..499]
    run_range = np.arange(runs).astype('int32')

    # The single step multi task nested looping.
    for iteration in trange(iterations):
        # random walk for q*(a)
        q += np.random.normal(loc=0, scale=0.01, size=(runs, k_arms))

        # epsilon-greedy choosing actions of the step
        sample = np.random.uniform(low=0.0, high=1.0, size=(runs,))

        # Why not just directly use argmax here?
        # If there are more than one options that are simultaneously max,
        # we still want to randomly choose among these options.

        # greedy_actions = np.argmax(Q, axis=1)

        max_estimate_value = np.max(Q[run_range, :], axis=1, keepdims=True)
        greedy_actions = np.ndarray(shape=(runs,))
        for run in range(runs):
            best_actions = np.where(Q[run, :] == max_estimate_value[run])[0]
            greedy_actions[run] = np.random.choice(best_actions)

        # Construct exploratory actions
        exploratory_actions = np.ndarray(shape=(runs,))
        for run in range(runs):
            # By definition we will not choose the greedy action when exploring.
            greedy_action_to_delete = int(greedy_actions[run])
            action_space = np.arange(0, k_arms, 1)
            action_space = np.delete(action_space, greedy_action_to_delete)
            exploratory_actions[run] = np.random.choice(action_space)

        # Choose actions
        actions = np.where(sample > epsilon, greedy_actions, exploratory_actions)
        # record the reward of current step (avg of multi runs)
        actions = actions.astype(dtype='int32')

        # Generate reward from sampling around q*(a)
        # Advanced indexing in numpy: R[iteration, run_range], q[run_range, actions]
        # and on each position, we chose the action for the task by vector a.
        # Usage: q[dim0_vector, dim1_vector], the dim0 vector is just a range from 0~499,
        R[iteration, run_range] = np.random.normal(loc=0, scale=1, size=(runs,)) + q[run_range, actions]

        # update the Q and N matrix
        # Advanced indexing in numpy: N[run_range, actions] (where actions is a 500-element vector for runs==500)
        N[run_range, actions] = N[run_range, actions] + 1

        # For the constant step size setting
        if step_size:
            alpha = step_size
        # For the varying step size setting
        else:
            ones = np.ones(shape=(runs,)).astype('float32')
            alpha = ones / (N[run_range, actions] + 1e-6)
        # alpha is a 500-element vector for runs==500
        Q_target = R[iteration, run_range]
        Q_old = Q[run_range, actions]
        Q[run_range, actions] += alpha * (Q_target - Q_old)

        # real best action can only be acquired from q*(a), which is invisible to the player.
        real_best_actions = np.argmax(q, axis=1)
        B[iteration, run_range] = actions == real_best_actions

    R_avg = np.mean(R, axis=1)
    B_avg = np.mean(B, axis=1)
    return B_avg, R_avg


if __name__ == '__main__':
    exercise_2_2()
