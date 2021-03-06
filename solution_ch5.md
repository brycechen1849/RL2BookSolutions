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

### Chapter 5
    
1. ***Exercise 5.1*** 

    ***Ans:***   
    Because if it's already 20 or 21, the player would sticks and has a great chance of winning.  
    While in other states, it will keep hitting and probably makes it burst.  
    The opponent having Ace would be a advantage for him.  
    Having an usable Ace has already been a better state for the player, which suggests that it has a sum between 12 and 21.  
      
        
1. ***Exercise 5.2*** 

    ***Ans:***   No. Both first-visit MC and every-visit MC converge to $v_{\pi}(s)$ as the number of visits (or first visits) to $s$ goes to infinity.   
    
    
1. ***Exercise 5.3*** 

    ***Ans:***  