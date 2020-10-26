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

### Chapter 4
    
1. ***Exercise 4.1*** In Example 4.1, if $\pi$ is the equiprobable random policy, what is $q_{\pi}(11,down)$? What is $q_{\pi}(7,down)$?

    ***Ans:***   
    \begin{equation}
        q_{\pi}(s,a) = \sum_{s',r}{p(s',r \mid s,a)(r+ \gamma v_{\pi})}
    \end{equation}  
    $q_{\pi}(7,down) = r + v_{\pi}(11) = -1 -14 = -15 $,  
    $q_{\pi}(11,down) = r + v_{\pi}(termination) = -1 + 0 = -1 $,  
      
        
1. ***Exercise 4.2*** 

    ***Ans:***   If transitions of state 13 is unchanged, then state 15 is not reachable unless it's born there.  
    So, state 15 won't be successor state of any states, and it does not affect the value function.  
    v(15) = 1/4 x ( v(12) + v(13) + v(14) + v(15) )  
    v(15)= -18.7  
    
    If state 15 is down from state 13, and you can also go other states from here, it becomes intertwined thus will change values of all states.  
    
    Programming simulation   
    
1. ***Exercise 4.3*** 

    ***Ans:***   
    \begin{equation}
        q_{\pi}(s,a) = E_{\pi} \left[ R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a \right]
    \end{equation}  
    
    \begin{equation}
        q_{\pi}(s,a) \doteq \sum_{s',r}{ \left[ p(s',r \mid s,a) \left[ r+ \gamma \sum_a{\pi(a \mid s') q_{\pi}(s',a)} \right] \right] }
    \end{equation}   
    
    \begin{equation}
        q_{k+1}(s,a) \doteq \sum_{s',r}{ \left[ p(s',r \mid s,a) \left[ r+ \gamma \sum_a{\pi(a \mid s') q_{k}(s',a)} \right] \right] }
    \end{equation}  
        
1. ***Exercise 4.4*** 

    ***Ans:***   if old action not in the set of actions that equally maximized the equation, then update the policy.
    
1. ***Exercise 4.5*** 

    ***Ans:*** Everything remains except 2 points:  
    In step 2 we have the self-consistency equation of $q_{\pi}$ under policy $\pi$:  
    \begin{equation}  
        q_{\pi}(s,a) \gets \sum_{s',r}{\left[p(s',r \mid s,a) \times  \left(r + \gamma \sum_{a' \sim \pi}{\pi(a' \mid s') q_{\pi}(s',a')}\right)\right]}  
    \end{equation} 
    In step 3 again we have $\pi(s)$ and update it to:
    \begin{equation}
         \pi(s) \gets argmax_{a} [q_{\pi}(s,a)]
    \end{equation}  
    
1. ***Exercise 4.6*** 

    ***Ans:***  
    In step 2 this term  
    \begin{equation}
        V_{\pi}(s) = \sum_{s',r}{p(s',r|s,\pi(s))(r + \gamma V_{\pi}(s'))}
    \end{equation}  
    is replaced with  
    \begin{equation}
        V_{\pi}(s) = \sum_{a \sim \pi'(s)}{\left[\pi'(s) \sum_{s',r}{p(s',r|s,\pi(s))(r + \gamma V_{\pi}(s'))}\right]}
    \end{equation}  
    where $\pi'(s)$ is $\epsilon-soft$  
    
1. ***Exercise 4.7*** 

    ***Ans:***    Programming simulation 

1. ***Exercise 4.8*** 

    ***Ans:***     

1. ***Exercise 4.9*** 

    ***Ans:***    Programming simulation 
    
1. ***Exercise 4.10*** 

    ***Ans:***  
    \begin{equation}
        Q_{k+1}(s,a) = \sum_{s',r}{ p(s',r|s,a) (r + \gamma Max_{a'}Q_{k}(s',a')) }
    \end{equation}  

1. ***Exercise 4.11*** 

    ***Ans:***    Programming simulation 
    
