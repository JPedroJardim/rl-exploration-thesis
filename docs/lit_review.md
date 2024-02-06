# Research on the exploration of reinforcement learning

## Literature Review

Include at least 10 high quality and relevant research papers to the final project proposal in order to demonstrate that you have an adequate initial understanding of the appropriate literature relating to your chosen research area. 
It is important to note that these 10 papers should not just be of marginal relevance, but central to the topic.

### Papers

#### Exploration in deep reinforcement learning: a survey
Reward novel states
- agent is rewarded for discovering novel or suprising state.
- computed using prediction error, count, or memory.
- in prediction error methods, the reward is given based on the accuracy of the agent's interval env model.
- in count-based methods, the reward is given based on how often a state is visited.
- in memory-based methods, the reward is computed based on how different a state is compared to other states in a buffer.

Reward diverse behaviours
- agent is rewarded for many diverse behaviours as possible (behaviour = sequence of actions or a policy).
- can be divided into evolutionary strategies and policy learning.
- in evolution strats, diversity among the population of agents is encouraged.
- in policy learning, tthe diversity of policy parameters is encouraged.

goal-based methods
- the agent is given the goal of either exploring from or exploring while trying to reach the goal.
- exploring from, the agent chooses the goal to get to and then explore from it (very efficient exploration as the agent visits predominantly unknown areas).
- in exploratory goal, the agent explores while travelling towards a goal (the key idea is to provide goals which are suitable for exploration).

probabilistic methods
- agent holds an uncertainty model about the env and uses it to make its next move.
- in optimistic methods, the agents follows under uncertainty principle (samples the most optimistic understanding of the reward).
- in uncertainty methods, the agent samples from internal uncertainty tto move towards least known areas.

imitation-based methods
- rely on demonstrations to help exploration by combining demonstrations with experience replay and combining them with an exploration strategy.
- combining demonstrations: samples from demonstrations and collected by the agent are combined into one buffer for the agent to learn from.
- combining with an exploration strategy: used as a starting point for other exploration techniques such as reward novel state. 

safe exploration
- ensures the safe behaviour of the agents during explorattion.
- human designer develops boundaries for the agent.
- agent can be discourged from visiting dangerous states with a negative reward.

random methods
- improve standard random exploration. Includes modifying the states for exploration, modifying exploration parameters, putting the noise on network parameters.
- in modifying states for exploration, certain stattes and actions are removed from the random choice if they have been sufficiently explored.
- in modifying exploration parameters, the parameters affecting when to randomly explore are automatically chosen based on the agent's learning progress.
- in network parameters noise approach, random noise is applied to the parameters to induce exploration before the weight convergence.

Easiest to implement: reward novel states, reward diverse behaviours, and random-based approaches.
Computational efficiency: random-based, reward novel states and reward divers behaviours generally require the least resources.
Best performing: goal-based and reward novel states methods. Goal-based has achieved high scores in difficult exploratory problems such as Montezuma's revenge, but tend to be the most complex in terms of implementation.