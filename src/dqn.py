import gymnasium as gym
from gym.wrappers import RecordVideo

import math
import random
#import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import os
import json
import logging
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from gridworld import GridWorldEnv


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
		# initialize sets of FC
        self.fc1 = nn.Linear(in_features=2, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=0)

        return x


def train_dqn_model(
        device_spec:str,
        epochs: int,
        steps_per_episode: int,
        steps_per_epoch: int,
        env_record_freq: int,
        environment_to_train = None,
        record=False,
        run_id = 0):
    
    device = torch.device(device_spec)

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.005
    LR = 1e-4
    MEM_SIZE = 10000

    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'recordings', 'dqn')
    #agent_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents_states', 'dqn')
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'dqn')
    logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'dqn')
    heatmap_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'heatmaps', 'dqn')

    def record_ep(ep: int) -> bool:
        return ep == env_record_freq

    run_on_gridworld = '.txt' in environment_to_train

    if run_on_gridworld:
        tmp_env = GridWorldEnv(filename=environment_to_train, render_mode='rgb_array')
        tmp_env.name = tmp_env.filename.split('.txt')[0]
    else:
        if environment_to_train == 'spaceinvaders':
            tmp_env = gym.make("ALE/SpaceInvaders-v5",
                    obs_type=env_type, 
                    render_mode='rgb_array')
            tmp_env.name = 'spaceinvaders'
        elif environment_to_train == 'mspacman':
            tmp_env = gym.make("ALE/MsPacman-v5",
                    obs_type=env_type, 
                    render_mode='rgb_array')
            tmp_env.name = 'mspacman'
        elif environment_to_train == 'montezuma':
            tmp_env = gym.make("ALE/MontezumaRevenge-v5",
                    obs_type=env_type, 
                    render_mode='rgb_array')
            tmp_env.name = 'montezuma'
        else:
            raise ValueError("No suitable environment was given.")

        tmp_env.metadata['render_fps'] = 30

    # create video folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(video_path, tmp_env.name))
    except FileExistsError:
        # directory already exists
        pass

    # create agent state folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(logs_path, tmp_env.name))
    except FileExistsError:
        # directory already exists
        pass

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

    # create heatmaps folder if it doesn't exist
    try:
        os.makedirs(os.path.join(heatmap_results_path))
    except FileExistsError:
        # directory already exists
        pass

    # create heatmaps folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(heatmap_results_path, tmp_env.name))
    except FileExistsError:
        # directory already exists
        pass

    log_level = logging.INFO
    logging.basicConfig(filename=os.path.join(logs_path, tmp_env.name, 'dqn_{:%Y-%m-%d}.log'.format(datetime.now())),
                    filemode='a', 
                    level=log_level,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    force=True)

    logging.info(f'Starting run {run_id}.')
    logging.info(f'Using device {device}.')
    logging.info(f'Prepared environment {tmp_env.name}.')

    policy_net = DQN(tmp_env.action_space.n).to(device)
    target_net = DQN(tmp_env.action_space.n).to(device)
    logging.info('Built Policy and Target networks')

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEM_SIZE)

    policy_net.train()

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
    logging.info(f'STEPS DONE SO FAR: {eps_steps}')

    # epochs correspond to a collection of environment steps, defined by steps_per_epoch
    for epoch in range(epochs):
        # an episode is an environment playthrough, that ends when the env sends the "terminated" flag
        # or reaches a max of steps_per_episode
        epoch_steps = 0
        logging.info(f"Starting Epoch {epoch}")

        epoch_rewards = {}

        epoch_heatmap = {}

        if record:
            env = RecordVideo(env=tmp_env, 
                            video_folder=video_path, 
                            episode_trigger=record_ep,
                            name_prefix=f"{tmp_env.name}_dqn_{run_id}_epoch{epoch}")
        else:
            env = tmp_env

        for episode in count():
            logging.info(f"\tEpisode {episode}")
            episode_steps = 0
            episode_rewards = []

            # reset of env
            state, _ = env.reset()

            heatmap_state = state.copy()

            state = torch.from_numpy(state).unsqueeze(0).to(torch.float32).to(device)

            epoch_rewards[episode] = {}
            epoch_rewards[episode]['sum_reward'] = 0
            epoch_rewards[episode]['duration'] = 0

            epoch_heatmap[episode] = {}

            terminated = False


            while episode_steps < steps_per_episode and not terminated:
                if str(heatmap_state) not in epoch_heatmap[episode]:
                    epoch_heatmap[episode][str(heatmap_state)] = 1
                else:
                    epoch_heatmap[episode][str(heatmap_state)] = epoch_heatmap[episode][str(heatmap_state)] + 1

                sample = random.random()
                eps_threshold = EPS_START - min((EPS_START - EPS_END), (EPS_DECAY * eps_steps))

                if sample > eps_threshold:
                    with torch.no_grad():
                        # t.max(1) will return the largest column value of each row.
                        # second column on max result is index of where max element was
                        # found, so we pick action with the larger expected reward.
                        values, indexes = torch.max(policy_net(state), dim=1)
                        # quite complicated. Basically find max value from each row, and then the max of it. 
                        # Then retrieve its index, which is the action
                        #print("Values", values)
                        #print("Indexes", indexes)

                        action = torch.tensor(
                            [[indexes[((values == torch.max(values)).nonzero(as_tuple=True)[0]).item()].item()]], 
                            device=device, 
                            dtype=torch.long
                        )
                else:
                    action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


                next_state, reward, terminated, _, _ = env.step(action.item())
                heatmap_state = next_state.copy()
                next_state = torch.from_numpy(next_state).unsqueeze(0).to(torch.float32).to(device)

                reward_for_log = reward

                reward = torch.tensor([reward], device=device)

                if terminated:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory) < BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                    # detailed explanation). This converts batch-array of Transitions
                    # to Transition of batch-arrays.
                    batch = Transition(*zip(*transitions))

                    # Compute a mask of non-final states and concatenate the batch elements
                    # (a final state would've been the one after which simulation ended)
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.bool)
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                                if s is not None])

                    # requires shape of itself because some states are None
                    non_final_next_states = non_final_next_states.unsqueeze(0).reshape((len(non_final_next_states), 2))

                    state_batch = torch.cat(batch.state, 0).unsqueeze(0).reshape((BATCH_SIZE, 2))
                    action_batch = torch.cat(batch.action, 0)
                    reward_batch = torch.cat(batch.reward, 0)
                    
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = policy_net(state_batch).gather(1, action_batch)

                    # Compute V(s_{t+1}) for all next states.
                    # Expected values of actions for non_final_next_states are computed based
                    # on the "older" target_net; selecting their best reward with max(1)[0].
                    # This is merged based on the mask, such that we'll have either the expected
                    # state value or 0 in case the state was final.
                    next_state_values = torch.zeros(BATCH_SIZE, device=device)
                    with torch.no_grad():
                        #print(non_final_next_states.size())
                        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                    # Compute the expected Q values
                    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                    # Compute Huber loss
                    criterion = nn.SmoothL1Loss()
                    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    # In-place gradient clipping
                    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                    optimizer.step()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    
                target_net.load_state_dict(target_net_state_dict)

                episode_rewards.append(reward_for_log)
                episode_steps += 1
                eps_steps += 1

            epoch_steps += episode_steps
     
            # optimization here
            logging.info(f"\t\tEpisode steps {episode_steps}")
            logging.info(f"\t\tTerminated flag {terminated}")
            logging.info(f"\t\tEpoch steps {epoch_steps}")
            logging.info(f"\t\tTotal reward {sum(episode_rewards)}")
            logging.info(f"\t\tCurrent exploration threshold {format(eps_threshold, '.5f')}")

            epoch_rewards[episode]['sum_reward'] = sum(episode_rewards)
            epoch_rewards[episode]['duration'] = episode_steps

            # if max num of steps per epoch is reached, move on to next epoch
            if epoch_steps >= steps_per_epoch:
                logging.info(f"------ Max steps per epoch have been reached {epoch_steps}")
                break


        with open(os.path.join(results_path, env.name, f'{run_id}_epoch{epoch}.json'), 'w') as f:
            json.dump(epoch_rewards, f)

        with open(os.path.join(heatmap_results_path, env.name, f'{run_id}_epoch{epoch}.json'), 'w') as f:
            json.dump(epoch_heatmap, f)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="FuN Model",
                                     description="Training script for the FuN model.")
    parser.add_argument('-d', '--device', required=True)
    parser.add_argument('-e', '--epochs', required=True)
    parser.add_argument('-spe', '--steps_per_episode', required=True)
    parser.add_argument('-spep', '--steps_per_epoch', required=True)
    parser.add_argument('-evs', '--env_record_step', required=True)
    parser.add_argument('-env', '--environment', required=False)

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
    
    train_dqn_model(
        device_spec=device_spec,
        epochs=epochs, 
                    steps_per_episode=steps_per_episode, 
                    steps_per_epoch=steps_per_epoch,
                    env_record_freq=env_record_step,
                    environment_to_train=environment_to_train,
                    env_type=env_type,
                    dilation_radius=dilation_radius,
                    prediction_horizon=prediction_horizon)