import os
import json
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', 
                        '--environment', 
                        required=False,
                        choices=['bridge_room', 'empty_room', 'four_rooms', 'mygridworld', 'nine_rooms', 'parr_maze', 'parr_mini_maze', 'ramesh_maze', 'six_rooms', 'spiral_room', 'two_rooms'])

    args = parser.parse_args()
    env = args.environment

    print(f'Initiating plot creation for environment {env}')

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'fun', env)
    charts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'charts')

    # create log folder for env if it doesn't exist
    try:
        os.makedirs(os.path.join(charts_path, env))
    except FileExistsError:
        # directory already exists
        pass

    reward_results = {}
    duration_results = {}

    reward_results['1_1'] = {}
    reward_results['5_5'] = {}
    reward_results['10_10'] = {}
    reward_results['20_20'] = {}
    reward_results['50_50'] = {}
    reward_results['1_10'] = {}
    reward_results['10_1'] = {}
    reward_results['10_50'] = {}
    reward_results['50_10'] = {}
    
    duration_results['1_1'] = {}
    duration_results['5_5'] = {}
    duration_results['10_10'] = {}
    duration_results['20_20'] = {}
    duration_results['50_50'] = {}
    duration_results['1_10'] = {}
    duration_results['10_1'] = {}
    duration_results['10_50'] = {}
    duration_results['50_10'] = {}
    

    for file in os.listdir(results_path):
        if 'placeholder' not in file:
            with open(os.path.join(results_path, file), 'r') as f:
                data = json.load(f)
                epoch_id = int(file.split('_')[0][5:])
                id = '_'.join(file.split('_')[1:]).removesuffix('.json')

            reward_results[id][epoch_id] = sum([data[k]['sum_reward'] for k in data.keys()])
            duration_results[id][epoch_id] = np.mean([data[k]['duration'] for k in data.keys()])

    fig, axs = plt.subplots(1, 2)

    fig.suptitle(f'{env}')

    print(len(axs))

    colors = ['red', 'limegreen', 'blue', 'darkcyan', 'black', 'magenta', 'grey', 'orange', 'violet']

    axs[0].set_title(f'Sum of Rewards per Epoch')
    axs[1].set_title(f'Mean Steps per Episode')

    axs[0].xaxis.set_major_locator(plt.MultipleLocator(1))
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(1))


    for idx, key in enumerate(reward_results.keys()):
        rewards = sorted(reward_results[key].items()) # sorted by key, return a list of tuples
        r_x, r_y = zip(*rewards) # unpack a list of pairs into two tuples

        durations = sorted(duration_results[key].items()) # sorted by key, return a list of tuples
        d_x, d_y = zip(*durations) # unpack a list of pairs into two tuples

        axs[0].plot(r_x, r_y, color=colors[idx], label=key)
        axs[1].plot(d_x, d_y, color=colors[idx], label=key)

    plt.legend()

    fig.set_size_inches(16, 6)

    plt.savefig(os.path.join(charts_path, env, f'{env}.png'))

    print(f'Saved image at {os.path.join(charts_path, env)}')

    plt.show()