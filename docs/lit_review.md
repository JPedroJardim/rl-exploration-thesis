## Literature Review

#### Paper on MDP

#### Paper on HRL

#### Paper on Options HRL

#### Paper on goal-conditional HRL



### Section on End-to-End HRL
#### Feudal Reinforcement Learning (Feudal Networks) (1992)
Dayan, P.; Hinton, G.E. Feudal reinforcement learning. In Proceedings of the Advances in Neural Information Processing Systems, Denver, CO, USA, 30 November–3 December 1992.

CHATGPT summary:

"Feudal Reinforcement Learning" presents a novel approach to hierarchical reinforcement learning (HRL) inspired by the organizational structure of feudal societies. In this framework, the learning agent is divided into two components: the manager and the worker. The manager sets goals for the worker, which then executes actions to achieve these goals. This hierarchical organization allows for the decomposition of complex tasks into simpler subgoals, facilitating more efficient learning. The manager-worker relationship is established through a combination of reward shaping and intrinsic motivation mechanisms. Experimental results demonstrate that this feudal RL approach outperforms traditional RL methods in various tasks, showcasing improved sample efficiency and performance. Overall, the article proposes a hierarchical RL framework that draws inspiration from historical organizational structures, offering potential benefits for learning complex tasks in artificial intelligence systems.




#### Feudal networks for hierarchical reinforcement learning (FuN) (2017)
Vezhnevets, A.S.; Osindero, S.; Schaul, T.; Heess, N.; Jaderberg, M.; Silver, D.; Kavukcuoglu, K. Feudal networks for hierarchical reinforcement learning. In Proceedings of the International Conference on Machine Learning, Sydney, Australia,
6–11 August 2017

CHATGPT summary:

"Feudal Networks for Hierarchical Reinforcement Learning" introduces a novel framework for hierarchical reinforcement learning (HRL) called Feudal Networks (FuNs). FuNs are designed to address the challenge of learning hierarchical control policies in complex environments by decomposing tasks into subgoals and actions. This decomposition is inspired by the hierarchical organization seen in feudal societies, where a central authority (manager) delegates tasks to lower-level agents (workers). In the FuN framework, the manager selects subgoals for the worker agents, while the workers execute actions to achieve those subgoals. The interactions between the manager and workers are learned through a combination of supervised learning and reinforcement learning. Experimental results demonstrate that FuNs outperform traditional RL methods in various tasks, showing improved sample efficiency and performance. Overall, the article presents FuNs as a promising approach to hierarchical RL, drawing inspiration from hierarchical structures observed in real-world systems.




#### Data-efficient hierarchical reinforcement learning (HIRO) (2018)
Nachum, O.; Gu, S.S.; Lee, H.; Levine, S. Data-efficient hierarchical reinforcement learning. In Proceedings of the Advances in Neural Information Processing Systems, Montreal, QC, Canada, 3–8 December 2018

CHATGPT summary:

"Data-efficient Hierarchical Reinforcement Learning" introduces a method aimed at improving the sample efficiency of hierarchical reinforcement learning (HRL) algorithms. The approach focuses on leveraging previously learned hierarchical structures to accelerate learning in new tasks, thereby reducing the amount of data required for effective training. By transferring knowledge from previously learned tasks to new ones, the method enhances the agent's ability to generalize across different environments and tasks. Experimental results demonstrate that the proposed approach significantly improves sample efficiency compared to traditional HRL methods, enabling agents to learn complex tasks with fewer interactions with the environment. Overall, the article contributes to advancing the field of reinforcement learning by providing a data-efficient approach to hierarchical learning, which has practical implications for real-world applications where data collection may be expensive or time-consuming.




#### Adjacency constraint for efficient hierarchical reinforcement learning (HRAC) (APR 2023)
T. Zhang, S. Guo, T. Tan, X. Hu, and F. Chen, “Adjacency constraint
for efficient hierarchical reinforcement learning,” IEEE Trans. Pattern
Anal. Mach. Intell., vol. 45, no. 4, pp. 4152–4166, Apr. 2023.

CHATGPT Summary:

"Adjacency Constraint for Efficient Hierarchical Reinforcement Learning" proposes a novel method to enhance the efficiency of hierarchical reinforcement learning (HRL) by introducing an adjacency constraint. This constraint aims to guide the learning process by encouraging the hierarchy to reflect the underlying structure of the task, ensuring that higher-level actions are composed of relevant lower-level actions. By enforcing adjacency, the method promotes the creation of hierarchical structures that are more interpretable and efficient for solving complex tasks. Experimental results demonstrate that incorporating the adjacency constraint leads to improved performance and sample efficiency compared to traditional HRL methods. Overall, the article presents a promising approach to enhancing HRL algorithms by incorporating task-specific constraints to guide the learning process effectively.




#### Learning multi-level hierarchies with hindsight (HAC) (SEP 2019)
A. Levy, G. D. Konidaris, R. Platt, Jr., and K. Saenko, “Learning multi-level hierarchies with hindsight,” in Proc. 7th Int. Conf. Learn. Represent., (ICLR), New Orleans, LA, USA, May 2019, pp. 1–15.

CHATGPT summary:

"Learning multi-level hierarchies with hindsight" introduces a novel approach to hierarchical reinforcement learning (HRL) by incorporating hindsight into the learning process. The method aims to improve the efficiency and effectiveness of learning hierarchies by leveraging hindsight information to guide the creation of subgoals and the hierarchical structure. By utilizing hindsight, which involves revisiting past experiences and extracting valuable insights, the agent can better understand the task structure and learn to achieve goals more efficiently. Experimental results demonstrate that the proposed approach outperforms traditional HRL methods in various environments, showing improved performance and sample efficiency. Overall, the article contributes to advancing the field of hierarchical reinforcement learning by integrating hindsight into the learning process, enabling agents to learn multi-level hierarchies more effectively.




#### Paper on subgoal discovery heuristic (FPS) (APR 2003)
. Moenning and N. A. Dodgson, “Fast marching farthest point
sampling,” Univ. Cambridge, Computer Lab., Cambridge, U.K.,
Tech. Rep. UCAM-CL-TR-562, Apr. 2003. [Online]. Available:
https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-562.pd

CHATGPT summary:
"Fast marching farthest point sampling" is an article that presents a novel method for efficiently sampling points on a surface. The technique is based on the fast marching algorithm, a widely-used method in computational geometry. By adapting this algorithm, the authors propose a way to sample points that are maximally spaced apart on the surface, which is useful in various applications such as computer graphics, mesh generation, and shape analysis. The method offers significant improvements in efficiency compared to existing techniques, making it a valuable tool for researchers and practitioners working with surface sampling tasks.




#### Sparse Graphical Memory for Robust Planning (heuristic) (2020)

CHATGPT summary:

"Sparse Graphical Memory for Robust Planning" introduces a novel memory structure designed to enhance the robustness of planning algorithms in uncertain environments. The proposed Sparse Graphical Memory (SGM) employs a graph-based representation that captures relevant information while minimizing redundancy, allowing for efficient storage and retrieval of critical data. By encoding relationships between states and actions in a sparse graphical format, the memory enables planners to make informed decisions even when faced with incomplete or noisy observations. Experimental results demonstrate that SGM improves the performance and reliability of planning algorithms across a range of tasks, including navigation and manipulation in dynamic environments. Overall, the article presents SGM as a promising approach for addressing the challenges of robust planning in real-world scenarios.




#### End-to-End Hierarchical Reinforcement Learning with Integrated Subgoal Discovery (DEC 2022)

- Proposal of an end-to-end HRL method called LIDOSS.
- LIDOSS is composed of an HRL agent with a heuristic for subgoal discovery.
    - The agent is similar to the HAC framework with the distinction that the subgoal discovery is done at the highest level of the hierarchy, using a nonparametric discrete-action policy as the highest level policy, and using the discrete set of discovered subgoals as the outputs of that policy.
    - The heuristic used in LIDOSS searches for the subgoal states that appear more frequently than others on the sequence of states observed during a given episode. Such subgoals are denoted salient subgoals, which are then used as the outputs of the highest-level policy of the agent.
- The heuristic is designed to achieve bottleneck subgoal discovery and subgoal reachability without using the rewards or the subgoal testing penalty.
- In the experiment phase, LIDOSS was compared to HAC in a set of continuous control tasks in the MuJoCo task domain. It was shown that LIDOSS performs better than HAC in all tasks containing obstacles or bottlenecks, to which the authors accredited mainly to the agent's distinction (and not really the heuristic).
- LIDOSS has the advantages that it reduces the overhead of learning a continuous-action highest-level policy over a continuous subgoal space, and that it explicitly includes the subgoal states lying at the bottleneck regions, among other discovered subgoals, which is beneficial in the task domains containing the state-space bottlenecks.
- LIDOSS has the following limitations: 
    - it does not have advantage over other subgoal discovery heuristics in task domains that do not contain bottlenecks.
    - it requires quantization of the subgoal space, which can be difficult in high dimensional subgoal spaces (curse of dimensionality).
    - it is not suitable for task domains with a very large state space (possibly infinite) because the continuous subgoal discovery makes the set of outputs of the highest-level policy unstable, thus affecting learning.

CHATGPT summary:

"End-to-End Hierarchical Reinforcement Learning with Integrated Subgoal Discovery" presents a novel approach to hierarchical reinforcement learning (HRL) that combines subgoal discovery with end-to-end learning. The authors propose a method where an agent learns to discover subgoals autonomously while simultaneously learning to achieve those subgoals in a hierarchical manner. This integrated approach improves the efficiency and effectiveness of learning complex tasks by providing the agent with intermediate goals to focus on during training. Experimental results demonstrate the effectiveness of the proposed method in various environments, showing improved performance and sample efficiency compared to traditional RL methods. Overall, the article contributes to advancing the field of HRL by integrating subgoal discovery seamlessly into the learning process, enabling agents to solve complex tasks more efficiently.



#### Goal-Conditioned Hierarchical Reinforcement Learning with High-Level Model Approximation (HLMA) (FEB 2024)

CHATGPT summary:

"Goal-Conditioned Hierarchical Reinforcement Learning with High-Level Model Approximation" introduces a novel approach to hierarchical reinforcement learning (HRL) that incorporates high-level model approximation. The method focuses on enabling agents to efficiently learn complex tasks by utilizing both low-level control policies and high-level goal-conditioned models. By leveraging a hierarchical structure, the agent is capable of learning to achieve abstract goals while also refining its control policies at a lower level. This approach enhances the agent's ability to generalize across tasks and environments by incorporating a high-level understanding of the task space. Experimental results demonstrate the effectiveness of the proposed method in various scenarios, highlighting its ability to achieve superior performance and sample efficiency compared to conventional RL methods. Overall, the article contributes to advancing the field of HRL by integrating high-level model approximation into the learning process, facilitating the efficient acquisition of complex skills in reinforcement learning tasks.