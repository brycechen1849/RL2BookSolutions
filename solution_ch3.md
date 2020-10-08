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

2. ***Exercise 3.2***  

    ***Ans:***  
    
    