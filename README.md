# rl-exploration-thesis
Reinforcement Learning Thesis on exploration through reward functions

The agent should be able to:
- Identify its inner state (its character in the game).
- Differenciate its inner state from the outer state (game env without the agent).
- Identify the elements seen (beasts and whatnot).
- Have a causality engine:
    - This action in this state results in x.
    - That action results in something else.
- Humans play Montezuma's Revenge by focusing on their character, the enemies, and the goal.
    - The agent should identify which is which through experience.
    - There shouldn't be any human designed elements here, i.e. the algorithm should manage to resolve how to pass the level without previous knowledge of the environment.
- The base algorithm of choice is a simple DQN.

#### Idenfication of the elements in the environment state
How to identify the element in the state?


#### Causality engine
Rewards should attributed based on interaction (states plus action) with elements in close proximity. Proximity should also be taken into consideration (an enemy 100 pixels away should not have the same threat level as one 5 pixels away).

Would this work in chess? The opponent could be 10 moves away from checkmate while not being a threat in the next 5 moves.