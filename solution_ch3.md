<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

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

### Chapter 3
1. ***Exercise 3.1*** Devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as di↵erent from each other as possible. The framework is abstract and flexible and can be applied in many di↵erent ways. Stretch its limits in some way in at least one of your examples.

    ***Ans:***  MDPs have markov property that a state must include information about all aspects of the past agent–environment interaction that make a difference for the future.  
    Three Examples:  
    + An agent that plays the game of go. the board represents the state. Location of the next move is it's action and reward is usually 0 after each move because the game is not yet finished, only when it's finished, the reward is a positive value if it wins or negative if it loses.  
    + An agent that trades assets in a financial market. Prices and other financial indices are the state. Actions include holding, short and long. the rewards are given after each action by it's profitability.
    + An agent that is used as temperature controller chip in an AC. current voltage, power and temperature consist of the state. The voltage and power to output are the action and reward is positive if it keeps the temperature in a predefined range.  
    The limits of example 2 (Trading agent) is that the state is partially observable to the agent. Thus the agent will receive an observation instead of a state in each interaction. Make if sometimes impossible to distinguish from two different states that has same observation.

2. ***Exercise 3.2*** Is the MDP framework adequate to usefully represent all goal-directed learning tasks? Can you think of any clear exceptions? 

    ***Ans:***  MDP is a framework modeling the decision making process of an learning agent. It has limitations. for example, the above described (exercise 3.1) trading agent, it does not meet the markov property because the 'state' it receives does not fully represents the full information it needs to tell which situation it is in. To resolve this problem, POMDP is proposed that an agent receives an observation that is partially observed from a state in markov decision process. it decides actions based on partially observed state instead a full state and receives again an partially observed state.
    
3. ***Exercise 3.3***  Consider the problem of driving. You could define the actions in terms of the accelerator, steering wheel, and brake, that is, where your body meets the machine. Or you could define them farther out—say, where the rubber meets the road, considering your actions to be tire torques. Or you could define them farther in—say, where your brain meets your body, the actions being muscle twitches to control your limbs. Or you could go to a really high level and say that your actions are your choices of where to drive. What is the right level, the right place to draw the line between agent and environment? On what basis is one location of the line to be preferred over another? Is there any fundamental reason for preferring one location over another, or is it a free choice? 
   
    ***Ans:*** I think there will be both agent that has low level control of motors and that has high level control of where to go. The later agent does the decision making job, such as when there is a stop sign, it decides if the whole vehicle should be stopping in a few seconds. And, the low level controller agent will execute the order from that high level agent in the form that it receives a state that explicitly requires it to do so. Such system and sub-systems are common in latest auto-driving implementation. It brakes down the job and assign them to proper disposal departments.

4. ***Exercise 3.4*** 
    Give a table analogous to that in Example 3.3, but for p(s' , r | s, a). It should have columns for s, a, s0, r, and p(s',r|s,a), and a row for every 4-tuple for which p(s',r|s,a) > 0.
    
    ***Ans:***  
    Since $p(s'\mid s,a) = \sum_{s' \in S}{p(s',r \mid s,a)}$ and fortunately each state has only one possible reward (or it's already an expected value). Thus we have:  

    |s|a|s'|r|p(s', r&#124;s,a)|  
    |----|----|----|----|:----:|  
    |high|wait|high|r_wait|1|  
    |high|search|high|r_search|a|  
    |high|search|low|r_search|1-a|  
    | high   | recharge |   high | 0   |  1 (Not necessary)                 |  
    | low   | wait |   low | r_wait   |  1                 |  
    | low   | search |   low | r_search   |  b                 |  
    | low   | search |   high | -3   |  1-b (deplete & recharge)           |  
    | low   | recharge |   high | 0   |  1                 |  
    
1. ***Exercise 3.5*** The equations in Section 3.1 are for the continuing case and need to be modified (very slightly) to apply to episodic tasks. Show that you know the modifications needed by giving the modified version of (3.3).
    
    ***Ans:*** change the set of s' from S(Non-termination state) to S+(All states including termination state)
    The original equation is:  
    $\sum_{s' \in S}{\sum_{r \in R}{p(s',r \mid s,a)}}=1,\forall s\in S,\forall a\in A(s).$         
    The modified version is:  
    $\sum_{s' \in S^+}{\sum_{r \in R}{p(s',r \mid s,a)}}=1,\forall s\in S,\forall a\in A(s).$    
    where, S stands for non-termination states and $S^+$ stands for all states including termination states.  
    
1. ***Exercise 3.6*** Suppose you treated pole-balancing as an episodic task but also used discounting, with all rewards zero except for -1 upon failure. What then would the return be at each time? How does this return differ from that in the discounted, continuing formulation of this task?  
    
    ***Ans:***   Suppose we have episodic length $T$, and current time step $t$, then current return is:  
    $G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_{T} $  
    And for the discount setting we have:  
    $G_t \doteq \gamma^0 R_{t+1} + \gamma^1 R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_{T} = -\gamma^{T-t-1}$  
    where $R_{T} = -1$ and all others 0  
    This is actually the same as $-\gamma^{K}$ in continuous task.
    
1. ***Exercise 3.7*** Imagine that you are designing a robot to run a maze. You decide to give it a reward of +1 for escaping from the maze and a reward of zero at all other times. The task seems to break down naturally into episodes—the successive runs through the maze—so you decide to treat it as an episodic task, where the goal is to maximize expected total reward (3.7). After running the learning agent for a while, you find that it is showing no improvement in escaping from the maze. What is going wrong? Have you effectively communicated to the agent what you want it to achieve?  
    
    ***Ans:***   
    There are 2 ways to tell the agent what we want it to do:
    + Use discount rate $\gamma$ to indicate that the earlier it gets outside the higher reward it would get.  
    + Use reward -0.01 as punishment for each time step before it gets outside.  
    Both method will in effect change the return estimation it assumes at a time step, making it struggle to get outside as soon as possible.  
    
1. ***Exercise 3.8***  Suppose $\gamma$= 0.5 and the following sequence of rewards is received R1 = -1, R2 =2, R3 =6, R4 =3, and R5 =2, with T =5. What are G0, G1, ...,G5? Hint: Work backwards.

    ***Ans:***  
    
    
 
    
    
    
    
    
    