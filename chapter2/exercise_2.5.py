#######################################################################
# Copyright (C)                                                       #
# 2018 - 2020 Bryce Chen (brycechen1849@gmail.com)                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib.pyplot as plt


def RandomWalk(x):
    # x is a vector, each element takes random walk independently, this function return a new vector where each element takes a step by the rule of  random walk
    x = x.astype('float32')
    rand = np.random.normal(loc=0, scale=0.01, size=(arm_number,))
    x += rand
    return x


def eps_greedy(epsilon, Q):
    # This function return an action chosen by epsilon greedy algorithm given the current action value estimate is Q
    sample = np.random.uniform(0, 1)
    if sample <= 1 - epsilon:
        i = np.argmax(Q)
        # dim = np.size(Q)
        return i
    else:
        i = np.argmax(Q)
        dim = np.size(Q)
        action_space = range(0, dim, 1)
        np.delete(action_space, i)
        return np.random.choice(action_space)


def single_step_single_task(max_iter, task_number, epsilon, arm_number, step_size):
    rows, cols = task_number, arm_number
    # my_matrix = np.array([([0.0] * cols) for i in range(rows)])
    constQ = np.array([([0.0] * cols) for i in range(rows)])
    variaQ = np.array([([0.0] * cols) for i in range(rows)])
    q = np.array([([0.0] * cols) for i in range(rows)])
    constN = np.array([([0.0] * cols) for i in range(rows)])
    variaN = np.array([([0.0] * cols) for i in range(rows)])
    constR = np.zeros(max_iter)
    variaR = np.zeros(max_iter)
    for i in range(max_iter):
        for j in range(task_number):
            # random walk of each arm
            task_q = q[j, :]
            task_q = RandomWalk(task_q)
            q[j, :] = task_q
            # constant stepsize

            task_constQ = constQ[j, :]
            task_constN = constN[j, :]
            action_const = eps_greedy(epsilon, task_constQ)

            RewardConst = task_q[action_const]
            constR[i] = constR[i] + RewardConst
            task_constN[action_const] = task_constN[action_const] + 1
            alpha = step_size
            difference = RewardConst - task_constQ[action_const]
            task_constQ[action_const] = task_constQ[action_const] + alpha * difference
            constQ[j, :] = task_constQ
            constN[j, :] = task_constN
            # Changing stepsize
            task_variaQ = variaQ[j, :]
            task_variaN = variaN[j, :]
            action_varia = eps_greedy(epsilon, task_variaQ)
            reward_varia = task_q[action_varia]
            task_variaN[action_varia] = task_variaN[action_varia] + 1
            if i == 0:
                beta = 1
            else:
                beta = 1 / task_variaN[action_varia]
            task_variaQ[action_varia] = task_variaQ[action_varia] + beta * (reward_varia - task_variaQ[action_varia])

            variaN[j, :] = task_variaN
            variaQ[j, :] = task_variaQ
            variaR[i] = variaR[i] + reward_varia
        variaR[i] = variaR[i] / task_number
        constR[i] = constR[i] / task_number
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.plot(variaR, color='r')
    plt.plot(constR, color='b')
    plt.xticks(np.arange(0, max_iter + 1, max_iter / 10))
    # plt.xticks(np.arange(len(constR)), np.arange(100, len(constR)+1) )
    # plt.grid()
    plt.show()
    plt.close()
    print(q)
    print(constQ)
    print(variaQ)


def single_step_multi_task(max_iter, task_number, epsilon, arm_number, step_size):
    tasks, arms = task_number, arm_number

    # Init the constant step-size setting. Q is the estimate of q*.
    Q_const = np.zeros(shape=(tasks, arms))
    N_const = np.zeros(shape=(tasks, arms))
    R_const = np.zeros(max_iter)

    # Init the sample-average method setting (1/N as step_size). Q is the estimate of q*.
    Q_s_avg = np.zeros(shape=(tasks, arms))
    N_s_avg = np.zeros(shape=(tasks, arms))
    R_s_avg = np.zeros(max_iter)

    # The underlying (ground-truth) q*(a) vector.
    q = np.zeros(shape=(tasks, arms))

    # The single step single task way of nested looping.
    for i_step in range(max_iter):

        # random walk for q*(a)
        # q += np.random.randint(low=-1, high=2, size=(tasks, arms))
        q += np.random.normal(loc=0, scale=0.01, size=(tasks, arms))

        # For the constant step size setting
        # epsilon-greedy choosing actions of the step
        a = np.ndarray(shape=(tasks,))
        sample = np.random.uniform(low=0.0, high=1.0, size=(tasks,))
        for index in range(tasks):
            # The greedy action
            if sample[index] <= 1 - epsilon:
                a[index] = np.argmax(Q_const[index, :])
            # The exploratory action
            else:
                # By definition we will not choose the greedy action when exploring.
                greedy_action = np.argmax(Q_const[index, :])
                action_space = range(0, arms, 1)
                np.delete(action_space, greedy_action)
                a[index] = np.random.choice(action_space)

        # record the reward of current step (avg of multi tasks)
        a = a.astype(dtype='int32')
        index = np.arange(tasks).astype('int32')
        # Here we use advanced indexing in numpy for selecting q*(a)
        # Usage: q[dim0_vector, dim1_vector], the dim0 vector is just a range from 0~499,
        # and on each position, we chose the action for the task by vector a.

        R_const[i_step] = np.sum(q[index, a]) / tasks
        # update the Q and N matrix
        N_add = np.zeros(shape=(tasks, arms))
        N_add[index, a] = 1
        N_const += N_add

        alpha = step_size
        Q_target = q[index, a]
        Q_old = Q_const[index, a]
        Q_const[index, a] += alpha * (Q_target - Q_old)

        # For the varying step size setting

        # epsilon-greedy choosing actions of the step
        a = np.ndarray(shape=(tasks,))
        sample = np.random.uniform(low=0.0, high=1.0, size=(tasks,))
        for index in range(tasks):
            # The greedy action
            if sample[index] <= 1 - epsilon:
                a[index] = np.argmax(Q_s_avg[index, :])
            # The exploratory action
            else:
                # By definition we will not choose the greedy action when exploring.
                greedy_action = np.argmax(Q_s_avg[index, :])
                action_space = range(0, arms, 1)
                np.delete(action_space, greedy_action)
                a[index] = np.random.choice(action_space)

        # record the reward of current step (avg of multi tasks)
        a = a.astype(dtype='int32')
        index = np.arange(tasks).astype('int32')
        R_s_avg[i_step] = np.sum(q[index, a]) / tasks

        # update the Q and N matrix
        N_add = np.zeros(shape=(tasks, arms))
        N_add[index, a] = 1
        N_s_avg += N_add

        ones = np.ones(shape=(tasks,)).astype('float32')
        alpha = ones / N_s_avg[index, a]
        Q_target = q[index, a]
        Q_old = Q_s_avg[index, a]
        Q_s_avg[index, a] += alpha * (Q_target - Q_old)

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.plot(R_s_avg, color='r')
    plt.plot(R_const, color='b')
    plt.xticks(np.arange(0, max_iter + 1, max_iter / 10))
    # plt.xticks(np.arange(len(constR)), np.arange(100, len(constR)+1) )
    # plt.grid()
    plt.show()
    # plt.close()
    print(q)
    print(Q_const)
    print(Q_s_avg)
    print(np.average(Q_const - q))
    print(np.average(Q_s_avg - q))


if __name__ == '__main__':
    # The iteration counts for a single experiment
    max_iter = 100000
    # Total experiment counts (used the word 'task' in the book).
    task_number = 10
    # The exploration probability.
    epsilon = 0.1
    # The test bed size (action space).
    arm_number = 10
    # The alpha used in constant step-size setting.
    step_size = 0.1

    import time

    seed = time.time()

    np.random.seed(0)
    start_1 = time.time()
    # np.random.seed(seed)
    single_step_multi_task(max_iter, task_number, epsilon, arm_number, step_size)
    # single_step_multi_task(max_iter, task_number, 0.1, arm_number, step_size)
    # single_step_multi_task(max_iter, task_number, 0.5, arm_number, step_size)
    end_1 = time.time()
    print("SSMT", end_1 - start_1)

    exit(0)
    np.random.seed(0)
    start = time.time()
    single_step_single_task(max_iter, task_number, epsilon, arm_number, step_size)
    end = time.time()

    print("SSMT", end_1 - start_1)
    print("SSST", end - start)
