import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class fixedSizeList():
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.list = []


    # will receive tensors
    def push(self, item: torch.Tensor):
        if len(self.list) == self.max_size:
            self.list.pop(0)

        self.list.append(item)


    def is_at_max_capacity(self):
        return len(self.list) == self.max_size

    def getListAsTensor(self):
        return torch.stack(self.list, dim=0)
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        #return torch.tensor(self.list[index], device=device)
        return self.list[index].to(device)



    def forward(self, 
                value: float, 
                reward: float, 
                policy_value: float,
                last_c_states: torch.Tensor, 
                last_c_goals: torch.Tensor):
        intrinsic_reward = self._calculate_intrinsic_reward(last_c_states, last_c_goals)

        # Calculate advantage estimate
        advantage = reward + self.alpha * intrinsic_reward - value

        logging.debug(f'Worker Loss Function - intrinsic reward: {intrinsic_reward}')
        logging.debug(f'Worker Loss Function - advantage: {advantage}')
        logging.debug(f'Worker Loss Function - policy value: {policy_value}')

        return advantage * policy_value


class dLSTM(nn.Module):
    def __init__(self, r: int, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.hidden_size = self.lstm.hidden_size
        self.r = r
        
        # note that we cannot keep the state in only one tensor as updating one place of the tensor counts
        # as an inplace operation and breaks the gradient history
        self.hn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=device) for _ in range(self.r)]
        self.cn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=device) for _ in range(self.r)]

        self.tick = 0


    def reset(self):
        self.hn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=device) for _ in range(self.r)]
        self.cn = [torch.zeros(self.lstm.hidden_size, requires_grad=False, device=device) for _ in range(self.r)]

        self.tick = 0


    def forward(self, x):
        logging.debug(f'dLSTM - Shape of x before lstm forward: {x.shape}')

        self.hn[self.tick], self.cn[self.tick] = self.lstm.forward(x, (self.hn[self.tick], self.cn[self.tick]))

        logging.debug(f'dLSTM - Shape of hn arr: {[tensor.shape for tensor in self.hn]}')
        logging.debug(f'dLSTM - Shape of cn arr: {[tensor.shape for tensor in self.cn]}')

        self.tick = (self.tick + 1) % self.r
        
        return sum(self.hn)/self.r


class Percept(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(640, 256)

    def forward(self, x):
        #logging.debug("Forward on Percept module.")
        logging.debug(f'Percept - Size of x before: {x.shape}')
        x = self.pool(F.relu(self.conv1(x)))
        logging.debug(f'Percept - Size of x after first layer: {x.shape}')
        x = self.pool(F.relu(self.conv2(x)))
        logging.debug(f'Percept - Size of x after second layer: {x.shape}')
        x = torch.flatten(x)
        logging.debug(f'Percept - Size of x after flatten: {x.shape}')
        x = F.relu(self.fc1(x))
        logging.debug(f'Percept - Size of x after fully connected layer: {x.shape}')
        return x


class Worker(nn.Module):
    def __init__(self, 
                 d: int, 
                 n_actions: int, 
                 k: int, 
                 c: int):
        super().__init__()

        self.d, self.k, self.n_actions, self.c = d, k, n_actions, c

        self.f_wrnn = nn.LSTMCell(input_size=d, hidden_size=n_actions*k)
        
        self.f_wrnn_states = (
            torch.zeros(self.f_wrnn.hidden_size, requires_grad=False, device=device),
            torch.zeros(self.f_wrnn.hidden_size, requires_grad=False, device=device)
        )

        self.phi = nn.Linear(d, k, bias=False)

        self.value_function = nn.Linear(self.n_actions * k, 1)


    def reset(self):
        self.f_wrnn_states = (
            torch.zeros(self.f_wrnn.hidden_size, requires_grad=False, device=device),
            torch.zeros(self.f_wrnn.hidden_size, requires_grad=False, device=device)
        )


    def forward(self, z: torch.Tensor, goals: torch.Tensor):
        goal_summation = torch.sum(goals, dim=0)
        logging.debug(f'Worker - Goal sum: {goal_summation.shape}')
        w = self.phi(goal_summation)
        #w = w.unsqueeze(1)
        logging.debug(f'Worker - W: {w.shape}')

        u_flat, c_x = self.f_wrnn(z, self.f_wrnn_states) # output is R^|a|*k
        logging.debug(f'Worker - U Flat: {u_flat.shape}')
        logging.debug(f'Worker - C_x: {c_x.shape}')

        #self.f_wrnn_states = (u_flat.detach(), c_x.detach())

        u = u_flat.view((self.n_actions, self.k))

        logging.debug(f'Worker - U: {u.shape}')

        logging.debug(f'Worker - U@w: {(u@w).shape}')

        action_probs = torch.softmax(u@w, dim=0)#.squeeze(1)

        logging.debug(f'Worker - Action Probabilities: {action_probs}')

        value = self.value_function(u_flat)

        logging.debug(f'Worker - value: {value}')

        self.picked_action_prob, picked_action = torch.max(action_probs, dim=0) 

        logging.debug(f'Worker - picked action index and probability: {picked_action} {self.picked_action_prob}')

        return picked_action, self.picked_action_prob, value


class Manager(nn.Module):
    def __init__(self, d: int, k: int, c: int, r: int):
        super().__init__()

        self.d, self.k, self.c, self.r = d, k, c, r
        
        self.f_mspace = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU()
        )
        
        self.f_mrnn = dLSTM(r=r, input_size=d, hidden_size=d)

        self.value_function = nn.Linear(d, 1)


    def _get_cosine_similarity(self, state_space_arr, goal_vec, state):
        if state_space_arr.is_at_max_capacity():
            return F.cosine_similarity((state_space_arr[self.f_mrnn.tick] - state).unsqueeze(0), goal_vec.unsqueeze(0))
        else:
            #logging.debug(f'Manager - Cosine Sim - state: {state.unsqueeze(0)}')
            #logging.debug(f'Manager - Cosine Sim - goal: {goal_vec.unsqueeze(0)}')
            #logging.debug(f'Manager - Cosine Sim - cosine: {F.cosine_similarity(state.unsqueeze(0), goal_vec.unsqueeze(0))}')
            return F.cosine_similarity(state.unsqueeze(0), goal_vec.unsqueeze(0))


    def reset(self):
        self.f_mrnn.reset()

    def forward(self, z, state_space_arr):
        s = self.f_mspace(z)
        logging.debug(f'Manager - State space map: {s.shape}')

        g_hat = self.f_mrnn(s)

        logging.debug(f'Manager - Post LSTM: {g_hat.shape}')
        g = F.normalize(g_hat, dim=0)
        logging.debug(f'Manager - Post goal normalization: {g.shape}')

        value = self.value_function(g_hat)

        cosine_similarity = self._get_cosine_similarity(state_space_arr, g, s)
        logging.debug(f'Manager - Cosine similarity: {cosine_similarity.shape}')

        return g, s, value, cosine_similarity


class FuN(nn.Module):
    def __init__(self, 
                 d: int, 
                 n_actions: int, 
                 k: int, 
                 c: int,
                 r: int,
                 unit_training=False):
        
        super().__init__()
        self.d, self.n_actions, self.k, self.c, self.r = d, n_actions, k, c, r

        self.unit_training = unit_training

        if self.unit_training:
            self.percept = nn.Sequential(
                                nn.Linear(4, self.d),
                                nn.ReLU()
                            )
        else:
            self.percept = Percept()

        self.manager = Manager(d, k, c, r)
        
        self.worker = Worker(d, n_actions, k, c)

        self.state_space_arr = fixedSizeList(c+1) # stores the last c state plus the most recent one
        self.goal_arr = fixedSizeList(c)


    def _worker_intrinsic_reward(self):
        current_state = self.state_space_arr[-1]

        cosine_sim_sum = torch.zeros(size=(256,), device=device)

        for i in range(0, len(self.state_space_arr)-1):
            cosine_sim_sum += F.cosine_similarity((current_state - self.state_space_arr[i]).unsqueeze(0),
                                                  self.goal_arr[i].unsqueeze(0))

        logging.debug(f'Worker Loss Function - sum of cosine: {torch.sum(cosine_sim_sum)}')

        return torch.sum(cosine_sim_sum/self.c)


    def reset_internal_state(self):
        self.manager.reset()
        self.worker.reset()


    def load_state_if_exists(self, path, env_name):
        current_n = 0
        current_state = ''

        for name in os.listdir(path):
            if name != 'placeholder.txt' and env_name in name:
                n = int(name.split(".model")[0].split('_')[0])
                if n > current_n:
                    current_n = n
                    current_state = name

        if current_n > 0:
            self.load_state_dict(torch.load(os.path.join(path, current_state)))
            current_n += 1
            logging.info(f"Loading model {current_state}")

        return current_n


    def forward(self, x):
        # Percept section
        logging.debug(f'FuN - Initial Size: {x.shape}')
        with torch.no_grad():
            percept_output = self.percept(x)
        logging.debug(f'FuN - Size after percept: {percept_output.shape}')

        #with torch.no_grad():
        goal, state, m_value, m_cosine_sim = self.manager(percept_output, self.state_space_arr)
        logging.debug(f'FuN - Size after manager: {goal.shape}')

        #with torch.no_grad():
        self.goal_arr.push(goal.detach())
        logging.debug(f'FuN - Size of goal arr: {self.goal_arr.getListAsTensor().shape}')

        #with torch.no_grad():
        action, policy_value, w_value = self.worker(percept_output, self.goal_arr.getListAsTensor())
        logging.debug(f'FuN - after worker: {action}')

        #with torch.no_grad():
        self.state_space_arr.push(state.detach())

        w_intrinsic_reward = self._worker_intrinsic_reward()

        return action.data, policy_value, w_intrinsic_reward, w_value, m_value, m_cosine_sim


def train_fun_model(epochs: int,
                    steps_per_episode: int,
                    steps_per_epoch: int,
                    env_record_freq: int,
                    environment_to_train: str,
                    unit_test_on_gridworld=False
                    ):

    WORKER_ALPHA = 0.99
    WORKER_GAMMA = 0.99
    MANAGER_GAMMA = 0.99
    LR = 0.99
    D = 256
    K = 16
    C = 10
    R = 10

    
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'recordings', 'fun')
    agent_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents_states', 'fun')
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'fun')
    logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'fun')

    def record_ep(ep: int) -> bool:
        return not(ep % env_record_freq)

    if unit_test_on_gridworld:
        from gridworld import GridWorldEnv

        tmp_env = GridWorldEnv()
        tmp_env.name = 'gridworld'
    else:
        if environment_to_train == 'spaceinvaders':
            tmp_env = gym.make("ALE/SpaceInvaders-v5",
                    obs_type="grayscale", 
                    render_mode='rgb_array')
            tmp_env.name = 'spaceinvaders'
        elif environment_to_train == 'mspacman':
            tmp_env = gym.make("ALE/MsPacman-v5",
                    obs_type="grayscale", 
                    render_mode='rgb_array')
            tmp_env.name = 'mspacman'
        elif environment_to_train == 'montezuma':
            tmp_env = gym.make("ALE/MontezumaRevenge-v5",
                    obs_type="grayscale", 
                    render_mode='rgb_array')
            tmp_env.name = 'montezuma'
        else:
            raise ValueError("No suitable environment was given.")

        tmp_env.metadata['render_fps'] = 30

    # create log folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(logs_path, tmp_env.name))
    except FileExistsError:
        # directory already exists
        pass

    # create results folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(results_path, tmp_env.name))
    except FileExistsError:
        # directory already exists
        pass

    log_level = logging.INFO
    logging.basicConfig(filename=os.path.join(logs_path, tmp_env.name, 'fun_{:%Y-%m-%d}.log'.format(datetime.now())),
                    filemode='a', 
                    level=log_level,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    force=True)

    logging.info(f'Starting.')
    logging.info(f'Using device {device}.')
    logging.info('Prepared environment.')

    model = FuN(d=D,
            n_actions=tmp_env.action_space.n,
            k=K,
            c=C,
            r=R,
            unit_training=unit_test_on_gridworld).to(device)

    # returns the epoch+1 on which the model was last saved. Default return is 0
    saved_epoch = model.load_state_if_exists(agent_state_path, tmp_env.name)
    logging.info(f'Epoch loaded: {saved_epoch}')

    # Optimizer
    optimizer = optim.Adam(list(model.manager.parameters()) + list(model.worker.parameters()), 
                           lr=LR, maximize=True)
    #w_optimizer = optim.Adam(model.worker.parameters(), lr=LR, amsgrad=True, maximize=True)

    model.manager.train()
    model.worker.train()

    
    # Exploration epsylon
    EPS_START = 0.9
    EPS_END = 0.1
    # decay should be according to training epoch
    # so the longer the model has been trained (higher epoch n)
    # the lower should be the epsylon decay, which signals a faster decay of EPS
    TOTAL_TRAIN_STEPS = epochs * steps_per_epoch
    EPS_DECAY = (EPS_START - EPS_END)/TOTAL_TRAIN_STEPS

    logging.info(f'EPS DECAY: {EPS_DECAY}')


    eps_steps = 0
    if saved_epoch:
        eps_steps = (saved_epoch-1) * steps_per_epoch # this won't work unless steps per epoch is static throughout the epochs.

    logging.info(f'STEPS DONE SO FAR: {eps_steps}')

    # epochs correspond to a collection of environment steps, defined by steps_per_epoch
    for epoch in range(saved_epoch, saved_epoch + epochs):
        # an episode is an environment playthrough, that ends when the env sends the "terminated" flag
        # or reaches a max of steps_per_episode
        epoch_steps = 0
        logging.info(f"Starting Epoch {epoch}")

        epoch_rewards = {}

        if not unit_test_on_gridworld:
            env = RecordVideo(env=tmp_env, 
                            video_folder=video_path, 
                            episode_trigger=record_ep,
                            name_prefix=f"{tmp_env.name}_fun_epoch{epoch}")
        else:
            # wrapper
            env = tmp_env

        for episode in count():
            logging.info(f"\tEpisode {episode}")
            episode_steps = 0
            episode_rewards = []

            # reset of env
            state, _ = env.reset()
            if unit_test_on_gridworld:
                state = torch.from_numpy(state).to(torch.float32).to(device)
            else:
                state = torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)

            epoch_rewards[episode] = {}
            epoch_rewards[episode]['sum_reward'] = 0
            epoch_rewards[episode]['duration'] = 0
            terminated = False

            worker_policy_delta = 0
            manager_goal_delta = 0

            while episode_steps < steps_per_episode and not terminated:
                model_action, w_policy_value, w_intrinsic_reward, w_value, m_value, m_cosine_similarity = model(state)

                # incentivate exploration
                sample = random.random()
                eps_threshold = EPS_START - min((EPS_START - EPS_END), (EPS_DECAY * eps_steps))

                if sample < eps_threshold:
                    action = model_action
                else:
                    action = torch.tensor(env.action_space.sample())
                
                state, reward, terminated, _, _ = env.step(action.item())
                if unit_test_on_gridworld:
                    state = torch.from_numpy(state).to(torch.float32).to(device)
                else:
                    state = torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)

                # worker section
                w_advantage_function = (WORKER_GAMMA * reward) + (WORKER_ALPHA * w_intrinsic_reward) - w_value
                worker_policy_delta += w_advantage_function * torch.log(w_policy_value)

                # manager section
                m_advantage_function = (MANAGER_GAMMA * reward) - m_value
                manager_goal_delta += m_advantage_function * m_cosine_similarity

                episode_rewards.append(reward)
                episode_steps += 1
                eps_steps += 1

            epoch_steps += episode_steps

            worker_policy_delta.backward()
            manager_goal_delta.backward()

            optimizer.step()

            # reset gradients
            optimizer.zero_grad()

            # reset internal state of the model
            model.reset_internal_state()
     
            # optimization here
            logging.info(f"\t\tEpisode steps {episode_steps}")
            logging.info(f"\t\tTerminated flag {terminated}")
            logging.info(f"\t\tEpoch steps {epoch_steps}")
            logging.info(f"\t\tTotal reward {sum(episode_rewards)}")
            logging.info(f"\t\tCurrent exploration threshold {eps_threshold}")

            epoch_rewards[episode]['sum_reward'] = sum(episode_rewards)
            epoch_rewards[episode]['duration'] = episode_steps

            # if max num of steps per epoch is reached, move on to next epoch
            if epoch_steps >= steps_per_epoch:
                logging.info(f"------ Max steps per epoch have been reached {epoch_steps}")
                break

        if not unit_test_on_gridworld:
            torch.save(model.state_dict(), os.path.join(agent_state_path, f'{epoch}_{env.name}.model'))
            logging.info('\tSaved model state.')

        with open(os.path.join(results_path, env.name, f'epoch{epoch}.json'), 'w') as f:
            json.dump(epoch_rewards, f)

    env.close()


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    """
    Args:
        --device (-d): str, device to be used in training (e.g., "mps" or "cuda:0")
        --epochs (-e): int, epochs to train.
        --steps_per_episode (-spe): int, maximum number of steps per episode.
        --steps_per_epoch (-spep): int, maximum number of steps per epoch.
        --env_record_step (-evs): int, record environment every x episodes.
        --unit_test (-ut): int, boolean signalling unit testing on gridworld.
        --environment (-env): str, which environment to train: mspacman, spaceinvaders, or montezuma
    """


    parser = argparse.ArgumentParser(prog="FuN Model",
                                     description="Training script for the FuN model.")
    parser.add_argument('-d', '--device')
    parser.add_argument('-e', '--epochs')
    parser.add_argument('-spe', '--steps_per_episode')
    parser.add_argument('-spep', '--steps_per_epoch')
    parser.add_argument('-evs', '--env_record_step')
    parser.add_argument('-ut', '--unit_test')
    parser.add_argument('-env', '--environment')

    args = parser.parse_args()
    device_spec = args.device
    epochs = int(args.epochs)
    steps_per_episode = int(args.steps_per_episode)
    steps_per_epoch = int(args.steps_per_epoch)
    env_record_step = int(args.env_record_step)
    unit_test = bool(int(args.unit_test))
    environment_to_train = args.environment
    

    if device_spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("The mps device is not available.")
    elif "cuda" in device_spec:
        if not torch.cuda.is_available():
            raise RuntimeError("The cuda device is not available.")
    
    device = torch.device(device_spec)

    train_fun_model(epochs=epochs, 
                    steps_per_episode=steps_per_episode, 
                    steps_per_epoch=steps_per_epoch,
                    env_record_freq=env_record_step,
                    unit_test_on_gridworld=unit_test,
                    environment_to_train=environment_to_train)
    