import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import seaborn as sns

def build_heatmpaps(model: str, env: str, env_size: int):
    heatmaps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'heatmaps', model, env)
    charts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'charts', 'heatmaps', model)
    
    print(f'Building heatmap for {env}.')

    # create log folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(charts_path, env))
    except FileExistsError:
        # directory already exists
        pass

    for file in os.listdir(heatmaps_path):
        with open(os.path.join(heatmaps_path, file)) as f:
            data = np.loadtxt(f, delimiter=',')
        
        if model == 'fun':
            epoch_id = int(file.split('_')[0][5:])
            id = '_'.join(file.split('_')[1:]).removesuffix('.csv')

            plt.title(f"Heatmap - {model} - {env} - {epoch_id} - {id}")
            axs = sns.heatmap(pd.DataFrame(data, columns=range(env_size[1])))

            plt.savefig(os.path.join(charts_path, env, f'{env}_{epoch_id}_{id}.png'))
            plt.cla()
            plt.clf()
        elif model == 'dqn':
            epoch_id = int(file.split('_')[1].split('.csv')[0][5:])
            
            plt.title(f"Heatmap - {model} - {env} - {epoch_id}")
            axs = sns.heatmap(pd.DataFrame(data, columns=range(env_size[1])))

            plt.savefig(os.path.join(charts_path, env, f'{env}_{epoch_id}.png'))
            plt.cla()
            plt.clf()
        elif model == 'lstm':
            epoch_id = int(file.split('.csv')[0][5:])
            
            plt.title(f"Heatmap - {model} - {env} - {epoch_id}")
            axs = sns.heatmap(pd.DataFrame(data, columns=range(env_size[1])))

            plt.savefig(os.path.join(charts_path, env, f'{env}_{epoch_id}.png'))
            plt.cla()
            plt.clf()


def build_results(model: str, env: str):
    print(f'Initiating plot creation for environment {env}')

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model, env)
    charts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'charts', model)

    print(f'Results path: {results_path}')
    print(f'Charts path: {charts_path}')

    # create log folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(charts_path, env))
    except FileExistsError:
        # directory already exists
        pass

    # create results folder for env if it doesn't exist
    try:
        os.makedirs(results_path)
    except FileExistsError:
        # directory already exists
        pass

    reward_results = {}
    duration_results = {}

    if model == 'fun':
        for file in os.listdir(results_path):
            if 'placeholder' not in file:
                with open(os.path.join(results_path, file), 'r') as f:
                    data = json.load(f)
                    epoch_id = int(file.split('_')[0][5:])
                    id = '_'.join(file.split('_')[1:]).removesuffix('.json')

                if id not in reward_results.keys():
                    reward_results[id] = {}
                
                if id not in duration_results.keys():
                    duration_results[id] = {}

                reward_results[id][epoch_id] = sum([data[k]['sum_reward'] for k in data.keys()])
                duration_results[id][epoch_id] = np.mean([data[k]['duration'] for k in data.keys()])

        fig, axs = plt.subplots(1, 2)

        fig.suptitle(f'{model} - {env}')

        colors = ['red', 'limegreen', 'blue', 'darkcyan', 'black', 'magenta', 'grey', 'orange', 'violet']

        axs[0].set_title(f'Sum of Rewards per Epoch')
        axs[1].set_title(f'Mean Steps per Episode')

        axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
        axs[1].xaxis.set_major_locator(plt.MultipleLocator(5))


        for idx, key in enumerate(reward_results.keys()):
            rewards = sorted(reward_results[key].items()) # sorted by key, return a list of tuples
            r_x, r_y = zip(*rewards) # unpack a list of pairs into two tuples

            durations = sorted(duration_results[key].items()) # sorted by key, return a list of tuples
            d_x, d_y = zip(*durations) # unpack a list of pairs into two tuples

            axs[0].plot(r_x, r_y, color=colors[idx%len(colors)], label=key)
            axs[1].plot(d_x, d_y, color=colors[idx%len(colors)], label=key)
    else:
        # DQN and LSTM stuff
        for file in os.listdir(results_path):
            with open(os.path.join(results_path, file), 'r') as f:
                data = json.load(f)
                if model == 'dqn':
                    epoch_id = int(file.split('.json')[0][7:])
                elif model == 'lstm':
                    epoch_id = int(file.split('.json')[0][5:])

            reward_results[epoch_id] = sum([data[k]['sum_reward'] for k in data.keys()])
            duration_results[epoch_id] = np.mean([data[k]['duration'] for k in data.keys()])


        fig, axs = plt.subplots(1, 2)

        fig.suptitle(f'{model} - {env}')


        colors = ['blue']

        axs[0].set_title(f'Sum of Rewards per Epoch')
        axs[1].set_title(f'Mean Steps per Episode')

        axs[0].xaxis.set_major_locator(plt.MultipleLocator(5))
        axs[1].xaxis.set_major_locator(plt.MultipleLocator(5))

        rewards = sorted(reward_results.items()) # sorted by key, return a list of tuples
        r_x, r_y = zip(*rewards) # unpack a list of pairs into two tuples

        durations = sorted(duration_results.items()) # sorted by key, return a list of tuples
        d_x, d_y = zip(*durations) # unpack a list of pairs into two tuples

        axs[0].plot(r_x, r_y, color=colors[0])
        axs[1].plot(d_x, d_y, color=colors[0])

    plt.legend()
    fig.set_size_inches(16, 6)
    plt.savefig(os.path.join(charts_path, env, f'{env}.png'))
    print(f'Saved image at {os.path.join(charts_path, env)}')
    plt.cla()
    plt.clf()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', 
                        '--model', 
                        required=True,
                        choices=['fun', 'dqn', 'lstm'])
    parser.add_argument('-envs', 
                        '--environments', 
                        required=False,
                        nargs='+')

    args = parser.parse_args()
    model = args.model

    list_of_possible_grids = {
        'empty_room': (13, 13),
        'mygridworld': (13, 13),
        'bridge_room': (13, 13),
        'four_rooms': (13, 13),
        'nine_rooms': (25, 25),
        'spiral_room': (18, 18),
        'ramesh_maze': (28, 31),
        'parr_maze': (86, 86)
    }

    working_dict = {}

    if args.environments:
        for env in args.environments:
            working_dict[env] = list_of_possible_grids[env]
    else:
        working_dict = list_of_possible_grids


    for env, env_size in working_dict.items():
        build_results(model=model, 
                      env=env)

        build_heatmpaps(model=model, 
                        env=env, 
                        env_size=env_size)