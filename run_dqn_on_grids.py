from src.dqn import train_dqn_model
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
    parser.add_argument('-rid', '--run_id', required=True)
    parser.add_argument('-rec', '--record', required=True)
    parser.add_argument('-envs', '--environments', nargs='+', required=False)
    parser.add_argument('-e', '--epochs', required=False, default=20)
    parser.add_argument('-spe', '--steps_per_episode', required=False, default=20_000)
    parser.add_argument('-spep', '--steps_per_epoch', required=False, default=100_000)

    args = parser.parse_args()
    device_spec = args.device
    run_id = int(args.run_id)
    record = bool(int(args.record))

    epochs=int(args.epochs)
    steps_per_episode=int(args.steps_per_episode)
    steps_per_epoch=int(args.steps_per_epoch)

    
    if device_spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("The mps device is not available.")
    elif "cuda" in device_spec:
        if not torch.cuda.is_available():
            raise RuntimeError("The cuda device is not available.")


    list_of_possible_grids = [
        'empty_room.txt',
        'mygridworld.txt',
        'bridge_room.txt',
        'two_rooms.txt',
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


    if not np.all([grid in list_of_possible_grids for grid in envs]):
        raise ValueError('One of the environments given is not possible.')

    print("Args:")
    print(f"\tDevice: {device_spec}")
    print(f"\tRun ID: {run_id}")
    print(f"\tRecord: {record}")
    print(f"\tEnvironments: {envs}")

    for grid in envs:
        print(f'Running {grid}')
        train_dqn_model(
            device_spec=device_spec,
            epochs=epochs, 
            steps_per_episode=steps_per_episode, 
            steps_per_epoch=steps_per_epoch,
            env_record_freq=0,
            environment_to_train=grid,
            record=record,
            run_id=run_id
        )