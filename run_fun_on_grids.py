from src.fun_agent import train_fun_model
import torch

import numpy as np

import argparse


if __name__ == "__main__":
    """
    Args:
        --device (-d): str, device to be used in training (e.g., "mps" or "cuda:0")
        --run_id (-rid): int, id of run.
        --environment (-env): str, which environment to train: mspacman, spaceinvaders, or montezuma
    """


    parser = argparse.ArgumentParser(prog="FuN Model",
                                     description="Training script for the FuN model.")
    parser.add_argument('-d', '--device', required=True)
    parser.add_argument('-r', '--runs', required=True)
    parser.add_argument('-rec', '--record', required=True)
    parser.add_argument('-envs', '--environments', nargs='+', required=False)

    args = parser.parse_args()
    device_spec = args.device
    runs = int(args.runs)
    record = bool(int(args.record))

    if device_spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("The mps device is not available.")
    elif "cuda" in device_spec:
        if not torch.cuda.is_available():
            raise RuntimeError("The cuda device is not available.")

    #device = torch.device(device_spec)

    list_of_possible_grids = [
        'empty_room.txt',
        'mygridworld.txt',
        'bridge_room.txt',
        'two_rooms.txt',
        'four_rooms.txt',
        'six_rooms.txt',
        'nine_rooms.txt',
        'spiral_room.txt',
        'ramesh_maze.txt',
        'parr_mini_maze.txt',
        'parr_maze.txt'
    ]

    if args.environments:
        envs = args.environments
    else:
        envs = list_of_possible_grids

    list_of_r_c = [
        (10, 10),
        (5, 5),
        (1, 1),
        (20, 20),
        (50, 50),
        (10, 50),
        (50, 10),
        (1, 10),
        (10, 1)
    ]

    if not np.all([grid in list_of_possible_grids for grid in envs]):
        raise ValueError('One of the environments given is not possible.')

    print("Args:")
    print(f"\tDevice: {device_spec}")
    print(f"\tRuns: {runs}")
    print(f"\tRecord: {record}")
    print(f"\tEnvironments: {envs}")

    for run in range(runs):
        for grid in envs:
            for params in list_of_r_c:
                print(f'Running {grid} with R = {params[0]} and C = {params[1]}')
                train_fun_model(
                    device_spec=device_spec,
                    epochs=20, 
                    steps_per_episode=50000, 
                    steps_per_epoch=200000,
                    env_record_freq=0,
                    environment_to_train=grid,
                    dilation_radius=params[0],
                    prediction_horizon=params[1],
                    record=record,
                    run_id=run
                )