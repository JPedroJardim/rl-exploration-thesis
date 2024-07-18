import os
import json
import matplotlib.pylab as plt
import numpy as np

results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'fun')

reward_results = {}
duration_results = {}

for file in os.listdir(results_path):
    if 'placeholder' not in file and 'gridworld' in file:
        with open(os.path.join(results_path, file), 'r') as f:
            data = json.load(f)
            id = int(file.split('_')[0][5:])

        reward_results[id] = sum([data[k]['sum_reward'] for k in data.keys()])
        duration_results[id] = np.mean([data[k]['duration'] for k in data.keys()])
    

rewards = sorted(reward_results.items()) # sorted by key, return a list of tuples
r_x, r_y = zip(*rewards) # unpack a list of pairs into two tuples

durations = sorted(duration_results.items()) # sorted by key, return a list of tuples
d_x, d_y = zip(*durations) # unpack a list of pairs into two tuples

fig, axs = plt.subplots(2)

axs[0].plot(r_x, r_y)
axs[1].plot(d_x, d_y)

axs[0].set_title('Sum of Rewards per Epoch')
axs[1].set_title('Mean Steps per Episode')

axs[0].xaxis.set_major_locator(plt.MultipleLocator(1))
axs[1].xaxis.set_major_locator(plt.MultipleLocator(1))

plt.show()