import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gym

import os
from datetime import datetime
import logging
import numpy as np
from itertools import count
import random
import json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from continuous_gridworld import ContinuousRoomsEnvironment


class LSTMAgent(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, env_n_actions: int, device: str):
        super(LSTMAgent, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        #self.input_layer = nn.Linear(input_size, hidden_size)

        self.input_layer = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.hn = torch.zeros(hidden_size, requires_grad=False, device=device)
        self.cn = torch.zeros(hidden_size, requires_grad=False, device=device)

        self.actor = nn.Linear(hidden_size, env_n_actions)
        self.critic = nn.Linear(hidden_size, 1)


    def forward(self, x):
        #x = F.relu(self.input_layer(x))

        self.hn, self.cn = self.input_layer(x, (self.hn, self.cn))

        return Categorical(logits=self.actor(self.hn)), self.critic(self.hn)
    

def train_lstm_model(
        device_spec:str,
        epochs: int,
        steps_per_episode: int,
        steps_per_epoch: int,
        environment_to_train = None,
        env = None
        ):
    
    device = torch.device(device_spec)

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    GAMMA = 0.99
    LR = 1e-4
    D = 128
    ACTOR_LOSS_COEFFICIENT = 1.0
    CRITIC_LOSS_COEFFICIENT = 0.5

    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'recordings', 'lstm')
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'lstm')
    logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'lstm')
    heatmap_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'heatmaps', 'lstm')
    envs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'envs')

    if env is None:
        env = ContinuousRoomsEnvironment(room_template_file_path=os.path.join(envs_path, environment_to_train), movement_penalty=0)
        env.name = environment_to_train.split('.txt')[0]
        env.metadata['render_fps'] = 30


    # create video folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(video_path, env.name))
    except FileExistsError:
        # directory already exists
        pass

    # create agent state folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(logs_path, env.name))
    except FileExistsError:
        # directory already exists
        pass

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
    logging.basicConfig(filename=os.path.join(logs_path, env.name, 'dqn_{:%Y-%m-%d}.log'.format(datetime.now())),
                    filemode='a', 
                    level=log_level,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    force=True)


    logging.info(f'Using device {device}.')
    logging.info(f'Prepared environment {env.name}.')

    lstm_agent = LSTMAgent(input_size=env.observation_space.shape[0], 
                           hidden_size=D, 
                           env_n_actions=env.action_space.n, 
                           device=device)

    logging.info('Built LSTM Agent networks')


    optimizer = optim.AdamW(lstm_agent.parameters(), lr=LR)

    lstm_agent.train()

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
            episode_steps = 0
            
            # reset of env
            state, _ = env.reset()
            converted_state = env._get_cell(state)
            heatmap_state = [int(converted_state[0]), int(converted_state[1])]

            state = torch.from_numpy(state).to(torch.float32).to(device)

            #logging.info(f"Reset state: {state}")

            epoch_rewards[episode] = {}
            epoch_rewards[episode]['sum_reward'] = 0
            epoch_rewards[episode]['duration'] = 0

            terminated = False

            episode_rewards = []
            values = []
            log_probs = []
            masks = []

            while episode_steps < steps_per_episode and not terminated:
                epoch_heatmap[heatmap_state[0]][heatmap_state[1]] += 1
                
                actor, value = lstm_agent(state)
                action = actor.sample()

                new_state, reward, terminated, _, _ = env.step(action.item())

                converted_new_state = env._get_cell(new_state)
                heatmap_state = [int(converted_new_state[0]), int(converted_new_state[1])]

                new_state = torch.from_numpy(new_state).to(torch.float32).to(device)

                episode_rewards.append(reward)
                values.append(value)
                log_probs.append(actor.log_prob(action))
                masks.append(1.0 - terminated)

                episode_steps += 1
                eps_steps += 1

                state = new_state


            with torch.no_grad():
                _, last_value = lstm_agent.forward(new_state)

                q_vals = []

                r = last_value

                for t in reversed(range(len(episode_rewards))):
                    r = episode_rewards[t] + GAMMA * r * masks[t]
                    q_vals.insert(0, r)

                q_vals = torch.cat(q_vals)
            

            values = torch.cat(values)
            log_probs = torch.stack(log_probs)

            advantages = q_vals - values
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()


            ac_loss = ACTOR_LOSS_COEFFICIENT * actor_loss + CRITIC_LOSS_COEFFICIENT * critic_loss

            # reset gradients
            optimizer.zero_grad()
            ac_loss.backward()

            optimizer.step()

            #lstm_agent.reset()
            
            epoch_steps += episode_steps


            # optimization here
            logging.info(f"\t\tEpisode steps {episode_steps}")
            logging.info(f"\t\tTerminated flag {terminated}")
            logging.info(f"\t\tEpoch steps {epoch_steps}")
            logging.info(f"\t\tTotal reward {sum(episode_rewards)}")

            epoch_rewards[episode]['sum_reward'] = sum(episode_rewards)
            epoch_rewards[episode]['duration'] = episode_steps

            # if max num of steps per epoch is reached, move on to next epoch
            if epoch_steps >= steps_per_epoch:
                logging.info(f"------ Max steps per epoch have been reached {epoch_steps}")
                break


        with open(os.path.join(results_path, env.name, f'epoch{epoch}.json'), 'w') as f:
            json.dump(epoch_rewards, f)

        np.savetxt(
            os.path.join(heatmap_results_path, env.name, f'epoch{epoch}.csv'), 
            epoch_heatmap, 
            delimiter=',',
            fmt='%u')

    env.close()


if __name__ == "__main__":#

    env = gym.make("CartPole-v1")
    env.name = 'cartpole'

    train_lstm_model(
        device_spec='cpu',
        epochs=10, 
        steps_per_episode=2000, 
        steps_per_epoch=10000,
        env=env
    )