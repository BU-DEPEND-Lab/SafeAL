Safety-Aware Apprenticeship Learning
======= 

This is the artifact for paper 'Safety-Aware Apprenticeship Learning'
which has been accepted by CAV2018. All three experiments in the paper are implemented
by using this artifact.

Basics
=======

There is one basic concept in apprenticeship learning: expected feature count. 
By demonstrating how a task is finished, a human expert generates a set of paths, 
which are all sequences of state-action-state tuples. 
By using apprenticeship learning algorithm, expert policy can be
approximated by a policy of which the expected feature counts match with the demonstrated paths.
By using the learnt policy, the learning agent can finish the task in the similar way as expert.

In Safety-Aware apprenticeship learning, a safety specifiation is given, and a counterexample will be
generated if a learnt policy doesn't satisfy a given safety specification. As counterexample is 
also a set of paths, the expected feature counts of the counterexample paths can be used. 
A policy tends to be safer if its expected feature count maximumlly deviate from the counterexample
feature count. By solving a multi-objective optimization function, a policy that resembles expert
policy while satisfing the safety specification can be learnt.


Installation
============
Programs are written in Python. NumPy and SciPy must be on your system to use mdptoolbox.
If you are installing onto Ubuntu or Debian and using Python 2 then this will pull in all the
dependencies:

  ``sudo apt-get install python-numpy python-scipy python-cvxopt``

The ``mdptoolbox`` library is used in the program. Just type

  ``pip install pymdptoolbox``
  
More information about ``mdptoolbox`` can be found in https://github.com/sawcordwell/pymdptoolbox.

As the first experiments requires ``pylab`` library, ``matplotlib`` must be installed.

Two other experiments are implemented in OpenAI gym environment. It's necessary to install OpenAI gym.

  ``pip install gym``
    
More information about ``gym`` can be found in https://github.com/openai/gym.

Quick Use
=========

The program start with a text-based interface. Please run ``main.py`` to start the experiment.

``python main.py``

There will be a list of experiment environment:

```
1. gridworld, 
2. cartpole,
3. mountaincar
```

Please input an index to select the experiment environment.

Gridworld
---------
Firstly input an integer by following the prompt to decide the dimension for the nxn gridworld. If press Enter without inputing any integer, the default value 8 will be used.

After deciding the dimension, a figure of gridworld with predetermined reward mapping will be generated. As described in the paper, there are 2 goal states with highest rewards and 2 darkest states with lowest rewards. Including the 2 darkest states, the areas surrounded by red lines are all unsafe states. When demonstrating the paths, please try to avoid crossing the red line. Press 0~4 in the terminal to select action in each time step: ``0. stay, 1. go right, 2. go down, 3. go up, 4. go left`` to go to the goal states. Because of stochasticity, there is probability that the agent doesn't go in the same direction as the action, except for ``0. stay``, which means staying in the current state and ending the episode. The maximum step length in one single episode is upperbounded by n^2. User can demonstrate as many episodes as needed. Please follow the prompt to choose whether to start the next episode or end the demonstration phase. While human demonstrating, the generated paths are stored in a plain text file. 

Once human expert declines to continue providing demonstration, please follow the prompt to start apprenticeship learning. After reading from the plain text, apprenticeship learning will start the iterations. In the end, a policy will be output and a new gridworld figure will be generated. The gray scale of each cell implies the reward mapping of which the output policy is optimal. One can compare it with the original reward mapping. A report will be ouput on the command line. Note that model checking will be implemented on the policy to show the probability of reaching either unsafe state within n^2 steps. The PCTL formulation is: 

``P=? [ true U<=n^2 'unsafe']``. 

The next step is to implement Safety-Aware Apprenticeship Learning algorithm. According to the model checking result, user can decide the safety threshold p* for the safety specification:

``P<=p* [ true U<=n^2 'unsafe']`` 

Please input a value for p* and start the algorithm. After certain amount of iterations, a new policy will be output. This policy is guaranteed to satisfy the specification above. As for the reward mapping of which the learnt policy is optimal, please follow the prompt to select and compare the reward mappings learnt either via Apprenticeship Learning or via Safety-Aware Apprenticeship Learning. 

Cartpole
---------
Cartpole is the second environment. The detail about this environment can be found in https://github.com/openai/gym/wiki/CartPole-v0.

The unsafe situation is defined as ``(position<-0.3 && angle<-20)||(position>0.3 && angle>20)``. The abstraction of the environment and the expert demonstration has already been generated. As described in the paper, all episodes achieve the maximum step length 200 while none of the episodes reaches the unsafe situation. 

Firstly, please follow the prompt to run apprenticeship learning. Like the gridworld, in the end, model checking will be implemented on the policy to solve the following PCTL constraint.

``P=? [true U<=200 ((position<-0.3 && angle<-20)||(position>0.3 && angle>20))]``

Then please select from ``1. Play the policy visually, 2. Run policy to get statistical data, 3. Quit`` by inputing an integer. If one chooses to play the policy visually, please press Enter in the terminal for 5 times until a window popping up. Please observe how the policy controls the cart to keep the pole stable. If one chooses to get statistical data, the policy will be run for 2000 episodes. The statistical results include the average step length and the rate of performing an unsafe episode. Note that once an unsafe state is reached, the episode is regarded as unsafe. If ``Quit`` is chosen, then the program will continue to run Safety-Aware Apprenticeship Learning.

Please refer to the paper as well as the model checking results of Apprenticeship Learning policy, and then input the safety threshold for the safety specification. After a certain amount of iterations, a policy will be generated. User can select the same options as above to test the newly learnt policy. The policy learnt via Apprenticeship Learning can also be chosen so that user can compare the two policies.

Mountaincar
------------
Mountaincar is the second environment. The detail about this environment can be found in https://github.com/openai/gym/wiki/MountainCar-v0. Note that in the original environment, the maximum step length is 200. However, in our setting, all actions must last for 3 time steps. The MDP is also built upon sampling in such way that one same action must last for 3 time steps before the next observation is accepted. Therefore, unlike the original environment setting, we use 66 as the maximum time step length, since there are at most 66 times to observe and choose actions. 

The unsafe situation is defined as ``(position<-0.9 && velocity<-0.03)||(position>0.3 && velocity>0.03)``. The expert demonstration has already been generated. As described in the paper, the average step length is 51 while none of the episodes reaches the unsafe situation. 

Firstly, please follow the prompt to run apprenticeship learning. Like the cartpole, in the end, model checking will be implemented on the policy to solve the following PCTL constraint.

``P=? [true U<=66 ((position<-0.9 && velocity<-0.03)||(position>0.3 && velocity>0.03))]``

Then please select from ``1. Play the policy visually, 2. Run policy to get statistical data, 3. Quit`` by inputing an integer. If one chooses to play the policy visually, please press Enter in the terminal for 5 times until a window popping up. Please observe how the policy controls the car to reach the mountaintop on the right. If one chooses to get statistical data, the policy will be run for 2000 episodes. The statistical results include the average step length and the rate of performing an unsafe episode. Note that once an unsafe state is reached, the episode is regarded as unsafe. If ``Quit`` is chosen, then the program will continue to run Safety-Aware Apprenticeship Learning.

Please refer to the paper as well as the model checking results of Apprenticeship Learning policy, and then input the safety threshold for the safety specification. After a certain amount of iterations, a policy will be generated. User can select the same options as above to test the newly learnt policy. The policy learnt via Apprenticeship Learning can also be chosen so that user can compare the two policies.


Source Code Tree
===============
There are three main components in the artifact. 

1. The first one controls the learning algorithms.

    [[mdp class]] provides resolution of discrete-time Markov Decision Process, including basic value iteration algorithm.
   
    [[grids class]] provides the discretion functions, such as translating observation vector to coordinate, and translating a state index to a coordinate.
  
    [[apirl class]] implements Apprenticeship Learning algorithm.

    [[cegal class]] implements Safety-Aware Apprenticeship Learning.

    [[gridworld class]] sets up environment for gridworld experiment.

    [[cartpole class]] sets up environment for cartpole experiment.

    [[mountaincar clas]] sets up environment for mountaincar experiment.

2. The second component is model checking functions. In this artifact, PRISM model checking tool is used. The source code is in ./prism-4.4.beta-src. More detail can be found in http://www.prismmodelchecker.org/.
 
3. The third component is counterexample generator. The COMICS tool is used. The relevant code is in ./comics-1.0. More detail can be found in http://www-i2.informatik.rwth-aachen.de/i2/comics/.

When running any experiment, such as gridworld, an grids class is firstly instantiated so that the observation vector can be indexed. Then mdp object is built up according to the states, action and transitions of the experiment object. When human expert demonstrates how to finish a task, the observations will be recorded. The apirl class takes an MDP and expert demonstation record as inputs and implements Apprenticeship Learning algorithm to learn a policy. The cegal class is a subclass of apirl class and share the same functions as apirl class. But when it is instantiated, it can implement Safety-Aware Apprenticeship Learning algorithm instead.
 

Check the log files
===================
The log files are all in ./data folder. 

1. The files named 'demo_XXXX' are the record of expert demonstrations. Note that the demo for gridworld can be generated by user, while cartpole and mountaincar demos are already provided and shouldn't be modified, since the requirements on human demonstration in those two experiments are much more rigorious.

2. The files named 'mdp_XXXX' are the temp file of the experiment object. For gridworld, since the dimension can be accustomed by user, the MDP may be modified by the program. For other two experiments, since the MDP is built up by sampling from OpenAI gym and shouldn't be modified. 

3. The files named 'policy_XXX_YY' are the policy files. Every time a policy is learnt, it's stored in a policy file. Note that 'XXX' is the experiment name while 'YY' is the safety threshould decided by user. When user wants to implement a policy, the program will load the policy by reading from the policy file.

4. The file named 'log' records the model checking results and statistical data whenever a policy is learnt or implemented in the environment. User can look up in this file to compare the performance of the learnt policies.

Error report
============
1. It's very common to see error report in the prompt when running the policy, such as 'No counterexample found ...'. If the process is not killed by the system, please don't stop it by hand.

2. The result after one run may be slightly deviated from the data in the paper. This is largely due to the randomization in the algorithm. One can run the program more times to see how the results may change. 