import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gymnasium as gym
from gym.wrappers import record_video

import numpy as np
import math
import random
import json

#from torchinfo import summary
#from torchviz import make_dot
#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter()

import logging
from datetime import datetime
from itertools import count
import os
import argparse


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
                 r: int):
        
        super().__init__()
        self.d, self.n_actions, self.k, self.c, self.r = d, n_actions, k, c, r

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
                    model_state_step: int,
                    env_record_freq: int
                    ):

    WORKER_ALPHA = 0.99
    WORKER_GAMMA = 0.99
    MANAGER_GAMMA = 0.99
    LR = 0.99
    D = 256
    K = 16
    C = 10
    R = 10

    # Exploration epsylon
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 100

    
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'recordings', 'fun')
    agent_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents_states', 'fun')
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'fun')
    logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'fun')

    log_level = logging.INFO
    logging.basicConfig(filename=os.path.join(logs_path, 'fun_{:%Y-%m-%d}.log'.format(datetime.now())),
                    filemode='a', 
                    level=log_level,
                    format="%(asctime)s %(levelname)s - %(message)s")

    def record_ep(ep: int) -> bool: return not(ep % env_record_freq)

    logging.info(f'Starting.')
    logging.info(f'Using device {device}.')

    #torch.autograd.set_detect_anomaly(True)

    env = gym.make("ALE/SpaceInvaders-v5", 
                   obs_type="grayscale", 
                   render_mode='rgb_array')
    
    env = record_video.RecordVideo(env, video_path, episode_trigger=record_ep,name_prefix="spaceinvaders_fun")
    logging.info('Prepared environment.')

    model = FuN(d=D,
            n_actions=env.action_space.n,
            k=K,
            c=C,
            r=R).to(device)

    # Optimizer
    optimizer = optim.Adam(list(model.manager.parameters()) + list(model.worker.parameters()), 
                           lr=LR, maximize=True)
    #w_optimizer = optim.Adam(model.worker.parameters(), lr=LR, amsgrad=True, maximize=True)

    model.manager.train()
    model.worker.train()

    eps_steps = 0
    #total_training_steps = epochs * steps_per_epoch

    # epochs correspond to a collection of environment steps, defined by steps_per_epoch
    for epoch in range(epochs):
        # an episode is an environment playthrough, that ends when the env sends the "terminated" flag
        # or reaches a max of steps_per_episode
        epoch_steps = 0
        logging.info(f"Starting Epoch {epoch}")

        epoch_rewards = {}

        for episode in count():
            logging.info(f"\tEpisode {episode}")
            episode_steps = 0
            episode_rewards = []

            epoch_rewards[episode] = {}
            epoch_rewards[episode]['sum_reward'] = 0
            epoch_rewards[episode]['duration'] = 0

            state, _ = env.reset()
            state = torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)
            terminated = False

            worker_policy_delta = 0
            manager_goal_delta = 0

            while episode_steps < steps_per_episode and not terminated:
                model_action, w_policy_value, w_intrinsic_reward, w_value, m_value, m_cosine_similarity = model(state)

                # incentivate exploration
                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * eps_steps / EPS_DECAY)

                if sample > eps_threshold:
                    action = model_action
                else:
                    action = torch.tensor(env.action_space.sample())
                
                state, reward, terminated, _, _ = env.step(action.item())
                state = torch.from_numpy(state).to(torch.float32).unsqueeze(0).to(device)

                # worker section
                w_advantage_function = (WORKER_GAMMA * reward) + (WORKER_ALPHA * w_intrinsic_reward) - w_value
                worker_policy_delta += w_advantage_function * torch.log(w_policy_value)

                # manager section
                m_advantage_function = (MANAGER_GAMMA * reward) - m_value
                manager_goal_delta += m_advantage_function * m_cosine_similarity

                episode_rewards.append(reward)
                episode_steps += 1

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

            epoch_rewards[episode]['sum_reward'] = sum(episode_rewards)
            epoch_rewards[episode]['duration'] = episode_steps

            # save everything necessary
            with open(os.path.join(results_path, f"episode{episode}_rewards.txt"), "w") as f:
                f.writelines([str(item)+'\n' for item in episode_rewards])


            if not(episode % model_state_step):
                torch.save(model.state_dict(), os.path.join(agent_state_path, f'fun_{episode}.model'))
                logging.info('\tSaved model state.')


            # if max num of steps per epoch is reached, move on to next epoch
            if epoch_steps >= steps_per_epoch:
                logging.info(f"------ Max steps per epoch have been reached {epoch_steps}")
                break


        with open(os.path.join(results_path, f'epoch{epoch}.json'), 'w') as f:
            json.dump(epoch_rewards, f)



def test_forward_backward_worker(n_steps: int):
    D = 256
    K = 16
    C = 10
    R = 10
    LR = 0.99

    env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")
    state, _ = env.reset()
    state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)

    model = FuN(d=D,
            n_actions=env.action_space.n,
            k=K,
            c=C,
            r=R)

    optimizer = optim.Adam(model.worker.parameters(), 
                           lr=LR)

    # forward
    with torch.no_grad():
        x = model.percept(state)

    dot = make_dot(model.worker(x, torch.zeros((1, 256))), params=dict(model.worker.named_parameters()))
    dot.render('worker')

    #dot = make_dot(model.worker.f_wrnn(x, model.worker.f_wrnn_states), params=dict(model.worker.f_wrnn.named_parameters()))
    #dot.render('lstm')

    # test steps and optimization
    for i in range(n_steps):
        action = env.action_space.sample()
        state, _, _, _, _ = env.step(action.item())
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)

        x = model.percept(state)
        w_action, policy_value, w_value = model.worker(x, torch.rand((1, D)))

        loss = 0.01 * w_action * policy_value * w_value

        optimizer.zero_grad()

        loss.backward()

        print("WORKER GRADIENTS", i)
        for name, param in model.worker.named_parameters():
            print(name, param.size(), param.grad, param.requires_grad)
        print()

        optimizer.step()


def test_forward_backward_manager(n_steps: int):
    D = 256
    K = 16
    C = 10
    R = 10
    LR = 0.99

    torch.autograd.set_detect_anomaly(True)

    env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")
    state, _ = env.reset()
    state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)

    model = FuN(d=D,
            n_actions=env.action_space.n,
            k=K,
            c=C,
            r=R)

    optimizer = optim.Adam(model.manager.parameters(),
                           lr=LR)


    model.train()

    # forward
    with torch.no_grad():
        x = model.percept(state)

    #result = model.worker()

    #dot = make_dot(model.manager(x, fixedSizeList(C+1)), params=dict(model.manager.named_parameters()))
    #dot.format = 'png'
    #dot.render('manager')

    #dot = make_dot(model.manager.f_mrnn(x), params=dict(model.manager.f_mrnn.named_parameters()))
    #dot.format = 'png'
    #dot.render('dlstm')

    optimizer.zero_grad()

    #weights_before = 

    loss = 0

    # test steps and optimization
    for i in range(n_steps):
        action = env.action_space.sample()
        state, _, _, _, _ = env.step(action.item())
        state = torch.from_numpy(state).to(torch.float32).unsqueeze(0)

        x = model.percept(state)
        _, _, value, cosine_similarity = model.manager(x, fixedSizeList(C+1))

        loss += 0.01 * value * cosine_similarity

        
    loss.backward()

    print("MANAGER GRADIENTS", i)
    for name, param in model.manager.named_parameters():
        if name == 'value_function.weight':
            print(name, param, param.requires_grad)
    print()

    optimizer.step()

    print("MANAGER GRADIENTS", i)
    for name, param in model.manager.named_parameters():
        if name == 'value_function.weight':
            print(name, param,  param.requires_grad)
    print()



if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    """
    Args:
        --device (-d): str, device to be used in training (e.g., "mps" or "cuda:0")
        --epochs (-e): int, epochs to train.
        --steps_per_episode (-spe): int, maximum number of steps per episode.
        --steps_per_epoch (-spep): int, maximum number of steps per epoch.
        --model_state_step (-mss): int, record model state every x episodes.
        --env_record_step (-evs): int, record environment every x episodes.
    """


    parser = argparse.ArgumentParser(prog="FuN Model",
                                     description="Training script for the FuN model.")
    parser.add_argument('-d', '--device')
    parser.add_argument('-e', '--epochs')
    parser.add_argument('-spe', '--steps_per_episode')
    parser.add_argument('-spep', '--steps_per_epoch')
    parser.add_argument('-mss', '--model_state_step')
    parser.add_argument('-evs', '--env_record_step')

    args = parser.parse_args()
    device_spec = args.device
    epochs = int(args.epochs)
    steps_per_episode = int(args.steps_per_episode)
    steps_per_epoch = int(args.steps_per_epoch)
    model_state_step = int(args.model_state_step)
    env_record_step = int(args.env_record_step)
    

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
                    model_state_step=model_state_step,
                    env_record_freq=env_record_step)
    