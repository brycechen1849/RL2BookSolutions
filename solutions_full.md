# Exercise Solutions

## Introduction 
See the source code on [Github Repo](https://github.com/brycechen1849/RL2BookSolutions), and if you have any questions, feel free to contact me at ***brycechen1849@gmail.com*** .
It serves mainly as a public note for the book and it's still being rapidly updated because I'm, at the same time, trying to get familiar with the RL research area.  

### References
The code implementations references are:
+ Solutions to exercise problems (However, this part are somewhat outdated because the latest version of the book has covered a lot of new exercises).
[Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions](https://github.com/LyWangPX/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions)
+ Code for each figure in the book: [reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)  

For figures, usage and examples can be accessed at *[Matplotlib Gallery](https://matplotlib.org/gallery/index.html)*

## Solutions

### Chapter1

Chapter 1 is an introductory one with tic-tac-toe game as an example of the full story. 
It involves more advanced topics of reinforcement learning problem that I would choose to implement the full setting when I finished later chapters.
(Although there is a implementation of the full simulation in tic-tac-toe.py)
1. ***Exercise 1.1***
1. ***Exercise 1.2***
1. ***Exercise 1.3***
1. ***Exercise 1.4***
1. ***Exercise 1.5***


### Chapter2
1. ***Exercise 2.1*** In epsilon-greedy action selection, for the case of two actions and " = 0.5, what is the probability that the greedy action is selected?  

    ***Ans:***  
    p(A_epsilon) = epsilon = 0.5 . Since probability 1-p is for the other action, not all actions combined.

1. ***Exercise 2.2*** Bandit example Consider a k-armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4.
Consider applying to this problem a bandit algorithm using "-greedy action selection, sample-average action-value estimates,
 and initial estimates of Q1(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A1 = 1, R1 = 1,A2 =2,R2 =1,A3 =2,R3 =-2,A4 =2,R4 =2,A5 =3,R5 =0. On some of these time steps the " case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?

    ***Ans:***  
    The greatest take away from here is that when doing epsilon-greedy way, exploratory actions include all actions but the greedy one.
    When there are more than one greedy actions, we randomly select an action among those equally good options.  
    Here is the simulation:
    ```  
    Step|Action|Reward|Estimations of the actions  
    01     1      -1      [-1, 0, 0, 0]
    02     2       1      [-1, 1, 0, 0]
    03     2      -2      [-1, -.5, 0, 0]
    04     2       2      [-1, .33, 0, 0]
    05     3       0      [-1, .33, 0, 0]
   ```
1. ***Exercise 2.3*** In the comparison shown in Figure 2.2, which method will perform best in the long run in terms of cumulative reward and probability of selecting the best action? How much better will it be? Express your answer quantitatively.

    ***Ans:***  
    The experiment is conducted with 10,000 iterations averaged by 2,000 runs and the epsilon=0.01 player performed best (see code generated fig 2.2).
    Quantitative analysis:  
    Reward Performance: ep=0.01 > ep=0.1 > ep=0
    Selection Performance: ep=0.01 > ep=0.1 > ep=0
    ![exercise 2.2](images/exercise_2_2.png)
    
1. ***Exercise 2.4*** If the step-size parameters, an, are not constant, then the estimate Qn is a weighted average of previously received rewards with a weighting different from that given by (2.6).
 What is the weighting on each prior reward for the general case, analogous to (2.6), in terms of the sequence of step-size parameters?
 
    ***Ans:***  
    Qn = Qn-1 + an * (Rn - Qn-1), Thus the weighting on each prior reward would be (1- an) for the n-th time step.
    
1. ***Exercise 2.5 (programming)***  Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for non-stationary problems. 
Use a modified version of the 10-armed testbed in which all the q*(a) start out equal and then take independent random walks
 (say by adding a normally distributed increment with mean zero and standard deviation 0.01 to all the q*(a) on each step).
Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, a = 0.1. Use epsilon = 0.1 and longer runs, say of 10,000 steps.
    
    ***Ans:***  
    Experiments done by exercise_2_5.py  
    The lines inserted to Bandit.step for the non-stationary bandit implementation:  
    ```python  
    # Nonstationary Bandit    
    self.q_true += np.random.normal(loc=0, scale=0.01, size=(self.k,))
    self.best_action = np.argmax(self.q_true)
    ```   
    and in Bandit.reset:
    
    ``` python
    # As stated in the prob, q starts at 0.
    self.q_true = np.zeros(shape=(self.k,)) + self.true_reward
    ```
   
    The constant step-size method outperformed the sample average method in terms of both average reward and best action hit rate.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    
    ![exercise 2.5](images/exercise_2_5.png)
    
    Unsatisfied with the simulation speed, I wrote a new version exercise_2_5_SIMR.py for this exercise prob. 
    SIMR stands for **Single Iteration Multi Runs** (You know it's from SIMD in chips). 
    Instead of going all the way through a complete run one after another, 
    this version simultaneously operates multi-runs at each iteration, as if those runs are in parallel.  
    This allowed us to utilize the power of the optimized vector computation tools in numpy, and it actually
    gets around **8x faster** than the first implementation.
     
    ![exercise 2.5 SIMR](images/exercise_2_5_SIMR.png)

1. ***Exercise 2.6*** *Mysterious Spikes* The results shown in Figure 2.3 should be quite reliable because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks. Why, then, are there oscillations and spikes in the early part of the curve for the optimistic method? In other words, what might make this method perform particularly better or worse, on average, on particular early steps?
    ![Fig 2.3](images/figure_2_3.png)
   
    ***Ans:***  
    The blue line with Q0 = 5 greatly encouraged exploration for each action in the very first 10 iterations.  
    When it comes to the 11-th iteration, the action with highest q value is also statistically expected to have highest Q value
    (all Qs are around ~ 4.5 after first 10 iterations in code simulation).   
    Since epsilon=0, it will not explore but stick to this action until it's Q value is reduced to be sub-largest. Thus is the spike.  
    As the Q value goes down from around 4.5 (real q values are around -1~1 in code simulation, 
    so of course Q will be reducing because it is targeting R sampled from q).   
    Then came the oscillation among best action and the rest because each of them would be having the largest Q value once in a while.  
    **Run the code in `debug mode`** and watch the changing of Q value will help a great deal to understand the progression.
 
1. ***Exercise 2.7*** *Unbiased Constant-Step-Size Trick* In most of this chapter we have used sample averages to estimate action values because sample averages do not produce the initial bias that constant step sizes do (see the analysis leading to (2.6)). However, sample averages are not a completely satisfactory solution because they may perform poorly on non-stationary problems. Is it possible to avoid the bias of constant step sizes while retaining their advantages on non-stationary problems?
   One way is to use a step size of Bn = a / On  (2.8
   to process the nth reward for a particular action, where a > 0 is a conventional constant step size, and On is a trace of one that starts at 0:  
   On = On-1 + a(1 - On-1), for n >=0, with O0=0  
   Carry out an analysis like that in (2.6) to show that Qn is an *exponential recency-weighted average* without initial bias.
   
    ***Ans:*** 
    To do

1. ***Exercise 2.8*** UCB Spikes In Figure 2.4 the UCB algorithm shows a distinct spike in performance on the 11th step. Why is this? Note that for your answer to be fully satisfactory it must explain both why the reward increases on the 11th step and why it decreases on the subsequent steps. Hint: if c = 1, then the spike is less prominent.
    ![Fig 2.4](images/figure_2_4_First_30.png)

    ***Ans:*** 
    ![Fig 2.4](images/exercise_2_4_First_30.png)
    
    c are respectively .5, 1, 2 and 4. The image on the right focuses on the first 30 iteration to get a clear view of the cliff (or spike).  
    **Conclusion:** for c <= 1, avg reward won't drop down after the 10-th iteration, but for c > 1 it will.  
    **Analysis:**  
    With the UCB term, it is encouraged to explore each action within the very first 10 iterations.  
    When it comes to the 11-th iteration, the UCB terms are all equal (because all N are equal to 1 now).  
    Now the action with highest q value is also statistically expected to have highest Q value
    Since epsilon=0, it will not explore but stick to this action until other actions' UCB term get large enough to surpass the optimal one's.
    For larger c, other actions' UCB term gets increased quickly. Thus the the performance degradation come earlier, which explains the downward part of the spike.  
    For smaller c, Q would have more time to get strong enough before it's UCB term are too small compared to others'.  
    **Run the code in `debug mode`** and watch the changing of Q value and UCB value will help a great deal to understand the progression.

1. ***Exercise 2.9*** Show that in the case of two actions, the soft-max distribution is the same as that given by the logistic, or sigmoid, function often used in statistics and artificial neural networks.

    ***Ans:***
    soft-max, assume preference of the 2 actions are H1, H2, then:  
    ```
    g(H1) = P(A = a1| H1, H2) 
          = e^H1 / (e^H1 + e^H2) 
          = e^(H1-H2) / (1 + e^(H1-H2))
    ```  
    
    where H2 is a constant in respect of H1.  
      
    Sigmoid function:  
    `f(x) = e^x / (1 + e^x)`  
    It's same kind of distribution but not identical (right-ward shifted on x-axis).

1. ***Exercise 2.10*** Suppose you face a 2-armed bandit task whose true action values change randomly from time step to time step.  
Specifically, suppose that, for any time step, the true values of actions 1 and 2 are respectively 0.1 and 0.2 with probability 0.5 (case A), and 0.9 and 0.8 with probability 0.5 (case B).  
If you are not able to tell which case you face at any step, what is the best expectation of success you can achieve and how should you behave to achieve it?  
Now suppose that on each step you are told whether you are facing case A or case B (although you still don’t know the true action values). This is an associative search task. What is the best expectation of success you can achieve in this task, and how should you behave to achieve it?
 
    ***Ans:***
    1. If you are not sure which case you are in, you have only 50% chance get the optimal one, in both cases.
    1. You are 100% guaranteed to be get the optimal action if you know which case you are in.  
    If you are in Case A, choose action 2. Else choose action 1.
    
11. ***Exercise 2.11 (programming)***  Make a figure analogous to Figure 2.6 for the non-stationary case outlined in Exercise 2.5. 
Include the constant-step-size epsilon-greedy algorithm with epsilon=0.1. 
Use runs of 200,000 steps and, as a performance measure for each algorithm and parameter setting, use the average reward over the last 100,000 steps.
    ![figure 2.6](images/figure_2_6.png)

    ***Ans:***
    results are as follows.
    ![exercise 2.11](images/exercise_2_11_Non-stationary.png)  
    There is a interesting points about this figure. The first one is why use the latest 10000 points' average?
    I think it's because this figure has a higher dimension than other figures in this chapter. It's x axis is not time steps but parameter settings.
    the lines do not show how the performance improves as time proceeds but what performance a parameters will eventually produce. It aims to describes a stable final state of the learner.
        