import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torchinfo import summary

from torchviz.dot import make_dot

import gymnasium as gym
from gym.wrappers import RecordVideo


import numpy as np
import math
import random
import json

import logging
from datetime import datetime
from itertools import count
import os
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from continuous_gridworld import ContinuousRoomsEnvironment


class fixedSizeList():
    def __init__(self, max_size: int, device=None):
        self.max_size = max_size
        self.list = []
        self.device = device


    # will receive tensors
    def push(self, item: torch.Tensor):
        if len(self.list) == self.max_size:
            self.list.pop(0)

        self.list.append(item)

    def reset(self):
        self.list = []

    def is_at_max_capacity(self):
        return len(self.list) == self.max_size

    def getListAsTensor(self):
        return torch.stack(self.list, dim=0)
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        #return torch.tensor(self.list[index], device=device)
        return self.list[index].to(self.device)


class dLSTM(nn.Module):
    def __init__(self, r: int, input_size: int, hidden_size: int, device=None):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.hidden_size = self.lstm.hidden_size
        self.r = r

        self.device=device
        
        # note that we cannot keep the state in only one tensor as updating one place of the tensor counts
        # as an inplace operation and breaks the gradient history
        self.hn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=device) for _ in range(self.r)]
        self.cn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=device) for _ in range(self.r)]

        self.tick = 0


    def reset(self):
        #self.hn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=self.device) for _ in range(self.r)]
        #self.cn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=self.device) for _ in range(self.r)]

        self.hn = [x.clone().detach() for x in self.hn]
        self.cn = [x.clone().detach() for x in self.cn]

        self.tick = 0


    def forward(self, x):
        logging.debug(f'dLSTM - Shape of x before lstm forward: {x.shape}')

        self.hn[self.tick], self.cn[self.tick] = self.lstm.forward(x, (self.hn[self.tick].detach(), self.cn[self.tick].detach()))

        logging.debug(f'dLSTM - Shape of hn arr: {[tensor.shape for tensor in self.hn]}')
        logging.debug(f'dLSTM - Shape of cn arr: {[tensor.shape for tensor in self.cn]}')

        self.tick = (self.tick + 1) % self.r
        
        return sum(self.hn)/self.r


class Worker(nn.Module):
    def __init__(self, 
                 d: int, 
                 n_actions: int, 
                 k: int, 
                 c: int,
                 device=None):
        super().__init__()

        self.d, self.k, self.n_actions, self.c = d, k, n_actions, c

        self.hidden_size = n_actions * k

        self.device = device

        self.f_wrnn = nn.LSTMCell(input_size=d, hidden_size=self.hidden_size)
        
        self.hn = torch.zeros(self.hidden_size, requires_grad=False, device=device)
        self.cn = torch.zeros(self.hidden_size, requires_grad=False, device=device)

        self.phi = nn.Linear(d, k, bias=False)

        self.critic = nn.Linear(self.hidden_size, 1)


    def reset(self):
        self.hn = self.hn.clone().detach()
        self.cn = self.cn.clone().detach()


    def forward(self, z: torch.Tensor, goals: torch.Tensor):
        logging.debug(f'Worker - Z: {z}')
        goal_summation = torch.sum(goals, dim=0)
        logging.debug(f'Worker - Goal sum: {goal_summation.shape}')
        w = self.phi(goal_summation)
        #w = w.unsqueeze(1)
        logging.debug(f'Worker - W: {w}')

        u_flat, c_x = self.f_wrnn(z, (self.hn.detach(), self.cn.detach())) # output is R^|a|*k
        #logging.debug(f'Worker - RNN states: {self.f_wrnn_states}')
        logging.debug(f'Worker - U Flat: {u_flat}')
        logging.debug(f'Worker - C_x: {c_x}')

        self.hn = u_flat.clone().detach()
        self.cn = c_x.clone().detach()

        u = u_flat.view((self.n_actions, self.k))

        logging.debug(f'Worker - U: {u}')

        logging.debug(f'Worker - U@w: {u@w}')

        return Categorical(logits=u@w), self.critic(u_flat)


class Manager(nn.Module):
    def __init__(self, d: int, k: int, c: int, r: int, goal_eps: float, device=None):
        super().__init__()

        self.d, self.k, self.c, self.r = d, k, c, r
        
        self.f_mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        ).to(device=device)

        self.f_mrnn = dLSTM(r=r, input_size=d, hidden_size=d, device=device)

        self.value_function = nn.Linear(d, 1).to(device=device)

        self.goal_epsilon = goal_eps


    def reset(self):
        self.f_mrnn.reset()


    def forward(self, z):
        s = self.f_mspace(z)
        logging.debug(f'Manager - State space map: {s.shape}')

        g_hat = self.f_mrnn(s)

        logging.debug(f'Manager - Post LSTM: {g_hat.shape}')
        g = F.normalize(g_hat, dim=0)
        logging.debug(f'Manager - Post goal normalization: {g.shape}')

        # incentivate exploration
        sample = random.random()
        if sample < self.goal_epsilon:
            g = torch.rand_like(g)

        value = self.value_function(g_hat)

        #cosine_similarity = self._get_cosine_similarity(state_space_arr, g, s)
        #logging.debug(f'Manager - Cosine similarity: {cosine_similarity}')

        return g.detach(), s.detach(), value#, cosine_similarity.detach()


class FuN(nn.Module):
    def __init__(self, 
                 d: int, 
                 n_actions: int, 
                 k: int, 
                 c: int,
                 r: int,
                 device=None,
                 manager_goal_eps=0.1):
        
        super().__init__()
        self.d, self.n_actions, self.k, self.c, self.r = d, n_actions, k, c, r
        
        self.device = device

        self.manager_goal_eps = manager_goal_eps

        self.percept = nn.Sequential(
                                nn.Linear(2, self.d), # 2 because on a grid the agent location (x, y) = state
                                nn.ReLU(),
                            ).requires_grad_(False).to(device=device)

        self.manager = Manager(d, k, c, r, goal_eps=manager_goal_eps, device=device)
        
        self.worker = Worker(d, n_actions, k, c, device=device)

        self.state_space_arr = fixedSizeList(r+1, device=device) # stores the last c state plus the most recent one
        self.goal_arr = fixedSizeList(r, device=device)


    def _worker_intrinsic_reward(self):
        current_state = self.state_space_arr[-1]

        cosine_sim_sum = torch.zeros(size=(self.d,), requires_grad=False, device=self.device)

        for i in range(0, len(self.state_space_arr)-1):
            cosine_sim_sum += F.cosine_similarity((current_state - self.state_space_arr[i]).unsqueeze(0),
                                                  self.goal_arr[i].unsqueeze(0))

        logging.debug(f'Worker Loss Function - sum of cosine: {torch.sum(cosine_sim_sum)/self.c}')

        return torch.sum(cosine_sim_sum)/self.c


    def reset_internal_state(self):
        self.manager.reset()
        self.worker.reset()

        self.state_space_arr.reset()
        self.goal_arr.reset()



    def forward(self, x):
        # Percept section
        logging.debug(f'FuN - Initial Size: {x.shape}')
        with torch.no_grad():
            percept_output = self.percept(x)
        logging.debug(f'FuN - Size after percept: {percept_output.shape}')

        #with torch.no_grad():
        goal, state, m_value, m_cosine_sim = self.manager(percept_output.data, self.state_space_arr)
        logging.debug(f'FuN - Size after manager: {goal.shape}')

        #with torch.no_grad():
        self.goal_arr.push(goal)
        self.state_space_arr.push(state)
        logging.debug(f'FuN - Size of goal arr: {self.goal_arr.getListAsTensor().shape}')

        #with torch.no_grad():
        worker_actor, w_value = self.worker(percept_output.data, self.goal_arr.getListAsTensor())

        with torch.no_grad():
            w_intrinsic_reward = self._worker_intrinsic_reward()

        return worker_actor, w_intrinsic_reward, w_value, m_value, m_cosine_sim


def calculate_manager_cosine_similarity(s_t: torch.Tensor, s_tc: torch.Tensor, g_t: torch.Tensor):
    return F.cosine_similarity(s_tc - s_t, g_t, dim=0)


def calculate_worker_intrinsic_reward(s_t: torch.Tensor, states: torch.Tensor, goals: torch.Tensor):
    #c = (len(states) if len(states) != 0 else 1)
    
    cosines = []

    for i in reversed(range(len(states))):
        s_ti = states[i]
        g_ti = goals[i]

        cosines.append(F.cosine_similarity(s_t - s_ti, g_ti, dim=0))

    return torch.mean(torch.tensor(cosines))


def train_fun_model(
        device_spec:str,
        epochs: int,
        steps_per_episode: int,
        steps_per_epoch: int,
        environment_to_train = None,
        dilation_radius=10,
        prediction_horizon=10
        ):

    device = torch.device(device_spec)

    WORKER_ACTOR_COEFFICIENT = 1
    WORKER_CRITIC_COEFFICIENT = 0.5

    MANAGER_ACTOR_COEFFICIENT = 1
    MANAGER_CRITIC_COEFFICIENT = 0.5

    MANAGER_EXPLORATION = 0.05

    WORKER_ALPHA = 0.1
    WORKER_GAMMA = 0.99

    MANAGER_GAMMA = 0.99

    LR = 1e-4
    D = 128 #64 #128 #256
    K = 8 #4 #8 #16
    C = prediction_horizon
    R = dilation_radius


    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'fun')
    logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'fun')
    heatmap_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'heatmaps', 'fun')
    envs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'envs')

    env = ContinuousRoomsEnvironment(room_template_file_path=os.path.join(envs_path, environment_to_train), movement_penalty=0)
    env.name = environment_to_train.split('.txt')[0]

    env.metadata['render_fps'] = 30

    # create log folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(logs_path, env.name))
    except FileExistsError:
        # directory already exists
        pass

    # create results folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(results_path, env.name))
    except FileExistsError:
        # directory already exists
        pass

    # create heatmaps folder if it doesn't exist
    try:
        os.makedirs(os.path.join(heatmap_results_path))
    except FileExistsError:
        # directory already exists
        pass

    # create heatmaps folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(heatmap_results_path, env.name))
    except FileExistsError:
        # directory already exists
        pass

    log_level = logging.INFO
    logging.basicConfig(filename=os.path.join(logs_path, env.name, 'fun_{:%Y-%m-%d}.log'.format(datetime.now())),
                    filemode='a', 
                    level=log_level,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    force=True)

    logging.info(f'Using device {device}.')
    logging.info(f'Prepared environment {env.name}.')
    logging.info(f'R = {R}, C = {C}')


    percept = nn.Sequential(
                nn.Linear(2, D), # 2 because on a grid the agent location (x, y) = state
                nn.ReLU(),
            ).requires_grad_(False).to(device=device)

    manager = Manager(D, K, C, R, goal_eps=MANAGER_EXPLORATION, device=device)
    worker = Worker(D, env.action_space.n, K, C, device=device)

    #state_space_arr = fixedSizeList(C+1, device=device) # stores the last c state plus the most recent one
    #goal_arr = fixedSizeList(C+1, device=device) # stores the last c goals plus the most recent one

    #summary(model, input_size=(2,))

    # Optimizer
    m_optimizer = optim.Adam(list(manager.parameters()),
                           lr=LR)
    w_optimizer = optim.Adam(list(worker.parameters()),
                           lr=LR)

    manager.train()
    worker.train()

    eps_steps = 0

    # epochs correspond to a collection of environment steps, defined by steps_per_epoch
    for epoch in range(epochs):
        # an episode is an environment playthrough, that ends when the env sends the "terminated" flag
        # or reaches a max of steps_per_episode
        epoch_steps = 0
        logging.info(f"Starting Epoch {epoch}")

        epoch_rewards = {}

        env_size = env.size
        epoch_heatmap = np.zeros(shape=env_size)

        for episode in count():
            logging.info(f"\tEpisode {episode}")
            # reset of env
            state, _ = env.reset()

            converted_state = env._get_cell(state)
            heatmap_state = [int(converted_state[0]), int(converted_state[1])]

            state = torch.from_numpy(state).to(torch.float32).to(device)

            epoch_rewards[episode] = {}
            epoch_rewards[episode]['sum_reward'] = 0
            epoch_rewards[episode]['duration'] = 0

            terminated = False

            episode_steps = 0
            episode_rewards = []
            
            worker_values = []
            worker_log_probs = []
            
            manager_values = []
            
            masks = []

            m_states = []
            m_goals = []

            while episode_steps < steps_per_episode and not terminated:
                #fig = make_dot(model(state, eps_threshold), show_attrs=True, show_saved=True)
                #fig.render('fun_dot', format='png')
                with torch.no_grad():
                    intermediate_state = percept(state)

                goal_vector, mstate, m_value = manager(intermediate_state.data)

                m_states.append(mstate)
                m_goals.append(goal_vector)

                worker_actor, w_value = worker(intermediate_state.data, torch.stack(m_goals[-C:], dim=0))
                
                action = worker_actor.sample()

                epoch_heatmap[heatmap_state[0]][heatmap_state[1]] += 1
                
                new_state, reward, terminated, _, _ = env.step(action.item())

                converted_new_state = env._get_cell(new_state)
                heatmap_state = [int(converted_new_state[0]), int(converted_new_state[1])]

                new_state = torch.from_numpy(new_state).to(torch.float32).to(device)

                episode_rewards.append(reward)

                worker_values.append(w_value)
                worker_log_probs.append(worker_actor.log_prob(action))
                #worker_intrinsic_rewards.append(w_intrinsic_reward)

                manager_values.append(m_value)
                

                masks.append(1.0 - terminated)

                episode_steps += 1
                eps_steps += 1

                state = new_state

            epoch_steps += episode_steps

            manager_cosine_sims = []
            worker_intrinsic_rewards = []

            for i in range(len(m_goals)):
                if i+C >= len(m_states)-1:
                    s_tc = m_states[-1]
                else:
                    s_tc = m_states[(i+C)%(len(m_states)-1)]

                s_t = m_states[i]
                g_t = m_goals[i]

                if i == 0:
                    last_c_states = m_states[i:]
                    last_c_goals = m_goals[i:]
                elif i < C:
                    last_c_states = m_states[:i]
                    last_c_goals = m_goals[:i]
                else:
                    last_c_states = m_states[i-C:i]
                    last_c_goals = m_goals[i-C:i]

                manager_cosine_sims.append(calculate_manager_cosine_similarity(s_t, s_tc, g_t))
                worker_intrinsic_rewards.append(calculate_worker_intrinsic_reward(s_t, last_c_states, last_c_goals))
            

            with torch.no_grad():
                intermediate_new_state = percept(new_state)
                _, _, m_value = manager(intermediate_new_state)
                _, w_value = worker(intermediate_new_state, torch.stack(m_goals[-C:]))

                worker_q_vals = []
                manager_q_vals = []

                worker_r = w_value
                manager_r = m_value

                for t in reversed(range(len(episode_rewards))):
                    worker_r = episode_rewards[t] + WORKER_ALPHA * worker_intrinsic_rewards[t] + (WORKER_GAMMA * worker_r) * masks[t]

                    manager_r = episode_rewards[t] + MANAGER_GAMMA * manager_r * masks[t]

                    worker_q_vals.insert(0, worker_r)
                    manager_q_vals.insert(0, manager_r)

                worker_q_vals = torch.cat(worker_q_vals)
                manager_q_vals = torch.cat(manager_q_vals)

            worker_values = torch.cat(worker_values)
            manager_values = torch.cat(manager_values)

            worker_log_probs = torch.stack(worker_log_probs)
            manager_cosine_sims = torch.stack(manager_cosine_sims)

            #print("Rewards", episode_rewards)
            #print("M states", m_states)
            #print("M goals", m_goals)
            #print("Manager Cosine Sims", manager_cosine_sims)
            #print("Worker intrinsic rewards", worker_intrinsic_rewards)

            # worker section
            worker_advantages = worker_q_vals - worker_values
            worker_loss = -(worker_log_probs * worker_advantages.detach()).mean()
            worker_critic_loss = worker_advantages.pow(2).mean()
            worker_policy_delta = WORKER_ACTOR_COEFFICIENT * worker_loss + WORKER_CRITIC_COEFFICIENT * worker_critic_loss

            # manager section
            manager_advantages = manager_q_vals - manager_values
            manager_loss = -(manager_cosine_sims * manager_advantages.detach()).mean()
            manager_critic_loss = manager_advantages.pow(2).mean()
            manager_goal_delta = MANAGER_ACTOR_COEFFICIENT * manager_loss + MANAGER_CRITIC_COEFFICIENT * manager_critic_loss

            # reset gradients
            m_optimizer.zero_grad()
            w_optimizer.zero_grad()

            worker_policy_delta.backward()
            manager_goal_delta.backward()

            m_optimizer.step()
            w_optimizer.step()

            # reset internal state of the model
            manager.reset()
            worker.reset()
     
            # optimization here
            logging.info(f"\t\tEpisode steps {episode_steps}")
            logging.info(f"\t\tTerminated flag {terminated}")
            logging.info(f"\t\tEpoch steps {epoch_steps}")
            logging.info(f"\t\tTotal reward {sum(episode_rewards)}")
            #logging.info(f"\t\tCurrent exploration threshold {format(eps_threshold, '.5f')}")

            epoch_rewards[episode]['sum_reward'] = sum(episode_rewards)
            epoch_rewards[episode]['duration'] = episode_steps

            # if max num of steps per epoch is reached, move on to next epoch
            if epoch_steps >= steps_per_epoch:
                logging.info(f"------ Max steps per epoch have been reached {epoch_steps}")
                break

            #break

        with open(os.path.join(results_path, env.name, f'epoch{epoch}_{C}_{R}.json'), 'w') as f:
            json.dump(epoch_rewards, f)

        np.savetxt(
            os.path.join(heatmap_results_path, env.name, f'epoch{epoch}_{C}_{R}.csv'), 
            epoch_heatmap, 
            delimiter=',',
            fmt='%u')
        
        #break

    env.close()


if __name__ == "__main__":
    """
    Args:
        --device (-d): str, device to be used in training (e.g., "mps" or "cuda:0")
        --epochs (-e): int, epochs to train.
        --steps_per_episode (-spe): int, maximum number of steps per episode.
        --steps_per_epoch (-spep): int, maximum number of steps per epoch.
        --env_record_step (-evs): int, record environment every x episodes.
        --unit_test (-ut): int, boolean signalling unit testing on gridworld.
        --environment (-env): str, which environment to train: mspacman, spaceinvaders, or montezuma
        --environment_type (-et): str, type of env: ram, grayscale
        --dilation_radius (-dr): int, dilation on Manager's dLSTM
        --prediction_horizon (-ph): int, prediction horizon of Manager
    """


    parser = argparse.ArgumentParser(prog="FuN Model",
                                     description="Training script for the FuN model.")
    parser.add_argument('-d', '--device', required=True)
    parser.add_argument('-e', '--epochs', required=True)
    parser.add_argument('-spe', '--steps_per_episode', required=True)
    parser.add_argument('-spep', '--steps_per_epoch', required=True)
    parser.add_argument('-evs', '--env_record_step', required=True)
    parser.add_argument('-env', '--environment', required=False)
    parser.add_argument('-et', '--env_type', required=False)
    parser.add_argument('-dr', '--dilation_radius', required=False)
    parser.add_argument('-ph', '--prediction_horizon', required=False)

    args = parser.parse_args()
    device_spec = args.device
    epochs = int(args.epochs)
    steps_per_episode = int(args.steps_per_episode)
    steps_per_epoch = int(args.steps_per_epoch)
    env_record_step = int(args.env_record_step)
    environment_to_train = args.environment
    env_type = args.env_type
    dilation_radius = int(args.dilation_radius) # R
    prediction_horizon = int(args.prediction_horizon) # C

    if device_spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("The mps device is not available.")
    elif "cuda" in device_spec:
        if not torch.cuda.is_available():
            raise RuntimeError("The cuda device is not available.")
    
    train_fun_model(
        device_spec=device_spec,
        epochs=epochs, 
                    steps_per_episode=steps_per_episode, 
                    steps_per_epoch=steps_per_epoch,
                    env_record_freq=env_record_step,
                    environment_to_train=environment_to_train,
                    env_type=env_type,
                    dilation_radius=dilation_radius,
                    prediction_horizon=prediction_horizon)
    