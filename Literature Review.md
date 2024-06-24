#### Literature Review

Literature Review

Ideas:
    From the summary of the review done below, one can see that most, if not all, of the algorithms have a high and a lower-level layer. The lower level layer is supposed to be able to solve smaller portions of the larger problem, while the high level layer makes sense of the results of the lower-level layer to solve the larger problem. Some algorithms use expert knowledge (cough cheating cough), others are thrown blind into an unknown universe and are expected to make sense of it. From an engineering perspective, this is perfectly as some frameworks/algorithms work better in some given task than their predecessors and it can be considered progress, but none really advances on the bigger question of "how can this algorithm/framework generate intelligent and flexible behaviour?".

    Let us look at our environment from a human perspective. We are thrown into this world crying and effectively blind, but with a body containing an immense amount of extrinsic and intrinsic sensors. From the progressive interpretation of these sensors arises our understanding and manipulation of the physical world we live in. Humans learn that an object bigger than some hole will not go through it (in most cases), that an untouched apple will fall down from a tree, that hitting somone with a blunt object may damage that person, i.e. humans learn the causality of their environment. And so, and this is unsubstantiated, abstraction may come from an understanding of the physical aspects of the environment.

    To this effect, HRL may necessitate a physics-based environmental decoder at its lowest level before tackling subtasks and, consequentely, the larger problem.
    
    There are intuitive-physics deep learning approaches. The bottom layer of the model could contain one of those approaches, followed by options framework. It is still undefined how to use the options framework since the whole point is for the agent to solve the env without expert knowledge, but alas that may not be possible at this time.


HRL paper (HRL: a survey and open research challenges)

    Hierarchical reinforcement learning is an extention on the capabilities of reinforcement learning using a divide-and-conquer strategy. Decomposing a problem into smaller problems (or tasks) makes it more manageable for humans. This approach is the basis of HRL, and has proven to make significant contributions towards solving open RL problems such as reward-function specification, exploration, sample efficiency, transfer learning, lifelong learning and interpretability.

    Mechanisms:
    - Temporal abstractions
    - State abstractions
    
    Advantages:
    - Credit Assignment
    - Exploration
    - Continual Learning

    Challenges:
    - Automatic discovery of abstractions
    - Development of abstractions
    - Decomposing example trajectories
    - Composition

    Problem-specific Models:
    - Feudal Q-learning
    - Hierarchies of Abstract Machines
    - MAXQ
        - Variable Influence Structure Analysis (VISA)
        - Hierarchy Induction via Models and Trajectories (HI-MAT)
    
    Options:
    - Framework
        - Initiation Set
        - Termination Condition
        - Intra-Option Policy
        - Policy-over-Options
    - Option Subgoal Discovery
        - Landmark States
            - Hierarchical Distance to Goal (HDG) algorithm
        - Reinforcement Learning Signal
        - Bottleneck States
        - Access States
        - Graph Partitioning
            - Q-Cut algorithm
            - L-Cut algorithm
        - State Clustering
            - PCCA+
        - Skill Chaining
    - Motion Templates
    - Macro-Actions
        - STRategic Attentive Write (STRAW)
        - STRAWe
        - Fine Grained Action Repetition (FiGAR)
    - Using Options in High-Dimensional State-Spaces
        - Hierarchical-DQN (h-DQN)
        - Hierarchical Deep Reinforcement Learning Network (H-DRLN)
        - Deep Skill Networks (DSN)
    - Option Discovery as Optimization Problem
        - Hierarchical Relative Entropy Policy Search (HiREPs)
        - Option-Critic algorithm (OC)
        - Proximal Policy Option-Critic (PPOC)
        - Actor-Critic TTermination Critic (ACTC)
    - Options as a tool for Meta-Learning
        - Meta Learning Shared Hierarchies (MLSH)
        - HiPPO
    
    Goal-Conditional:
    - General Value Functions
        - Horde algorithm
        - Universal Value Function Approximation (UVFA) architecture
        - Hybrid Reward Architecture (HRA)
        - UNsupervised REinforcement and Auxiliary Learning (UNREAL) architecture
    - Information Hiding
    - Unsupervised Learning in HRL
        - Entropy Maximization
        - Unsupervised self-play for learning goal-embeddings
        - Stochastic Neural Networks (SNN) with information-theoretic regularizer
        - Variational Intrinsic Control (VIC)
        - Diversity is All You Need (DIAYN)
        - Variational Autoencoding Learning of Options by Reinforcement (VALOR) architecture
        - Latent Space Policies (LSP)
    - End-to-End algorithms
        - FeUdal Networks (FuN)
        - HIRO
    
    Benchmarks:
    - Low-Dimensional State-Space Environments
    - High-Dimensional State-Space Environments
        - Discrete Action Spaces
        - Continuous Control

    Comparative Analysis
    - Frameworks Summary
    - Algorithms
        - Training Method
        - Required Priors
        - Interpretable Elements
        - Transferrable Policy

    Open Research Challenges
    - Top-Down Hierarchical Learning
    - Subgoal Representation
    - Lifelong Learning
    - HRL for Exploration
    - Increasing Depth
    - Interpretable RL
    - Benchmark
    - Alternative Frameworks



Use arkiv related papers option on review document and go from there.