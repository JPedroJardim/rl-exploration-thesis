import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from continuous_gridworld import ContinuousRoomsEnvironment

if __name__ == "__main__":
    envs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs')
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagrams')
    
    list_of_possible_grids = [
        'empty_room.txt',
        'mygridworld.txt',
        'bridge_room.txt',
        'four_rooms.txt',
        'nine_rooms.txt',
        'spiral_room.txt',
        'ramesh_maze.txt',
        'parr_maze.txt'
    ]

    for env_name in list_of_possible_grids:
        print('Rendering', env_name)
        env = ContinuousRoomsEnvironment(room_template_file_path=os.path.join(envs_path, env_name), movement_penalty=0, render_frame=True, frame_save_path=file_path, render_mode='rbg_array')
        state, _ = env.reset()

        env.render()
        #env.save_frame(os.path.join(file_path, env_name))

        #try:
        #    while True:
        #        _
        #except KeyboardInterrupt:
        #    continue