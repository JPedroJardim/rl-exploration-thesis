import numpy as np
import pygame
import random
import os.path

import gymnasium as gym
from gymnasium import spaces

CELL_TYPES_DICT = {".": "floor", "#": "wall", "S": "start", "G": "goal", "A": "agent"}

envs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs')

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, filename='mygridworld.txt', render_mode=None, size=6, repeat_action=0.25):
        self.filename = filename
        
        self._agent_location = None
        self._target_location = None

        self._load_layout(os.path.join(envs_path, filename))

        #self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.repeat_action = repeat_action

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # Right
            1: np.array([0, 1]), # Down
            2: np.array([-1, 0]), # Left
            3: np.array([0, -1]), # Up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.previous_action = None


    def _load_layout(self, file_path):
        env_data = []
        self._wall_location = []

        try:
            with open(file_path, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    env_data.append(list(line.strip('\n')))
        except:
            raise FileNotFoundError(f'Could not find {file_path}.')

        self.size = len(env_data[0])    

        for idr, row in enumerate(env_data):
            for idc, column in enumerate(row):
                if column == '#':
                    self._wall_location.append([idr, idc])
                elif column == 'S':
                    self._agent_location = np.array([idr, idc])
                elif column == 'G':
                    self._target_location = np.array([idr, idc])

        if self._agent_location is None or self._target_location is None:
            raise ValueError(f'Agent and/or Target location is not set. Env: {file_path}')

        self._wall_location = np.array(self._wall_location)
        

    def _get_obs(self):
        return np.array([self._agent_location[0], \
                    self._agent_location[1]])
    
    
    def _get_info(self):
        return np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        #self._agent_location = np.array([max(0, int(self.size / 5) - 1), max(0, int(self.size / 5) - 1)])
        #self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._load_layout(os.path.join(envs_path, self.filename))

        # Choose the target's location
        #self._target_location = np.array([max(0, (int(self.size / 5) + 1) + int(self.size/2)), max(0, (int(self.size / 5) + 1) + int(self.size/2))])

        # Choose the wall's location
        #self._wall_location = np.array([[2,2], [3,2], [4,2], [5,2], [9,4], [2,5]])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        #self._target_location = self._agent_location
        #while np.array_equal(self._target_location, self._agent_location):
        #    self._target_location = self.np_random.integers(
        #        0, self.size, size=2, dtype=int
        #    )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        sample = random.random()

        if sample < self.repeat_action and self.previous_action is not None:
            direction = self._action_to_direction[self.previous_action]
        else:
            direction = self._action_to_direction[action]
            self.previous_action = action

        if self._agent_location[0] + direction[0] not in range(self.size) or self._agent_location[1] + direction[1] not in range(self.size):
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )
        else:
        # Hitting a wall turns into no action
            hit_wall = False
            for wall in self._wall_location:
                if np.array_equal(self._agent_location + direction, wall):
                    hit_wall = True
                    break

            if not hit_wall:
                self._agent_location = self._agent_location + direction
            #else:
            #    print('Hit a wall at', self._agent_location)

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the walls
        for wall in self._wall_location:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                pix_square_size * wall,
                (pix_square_size, pix_square_size),
                )
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



if __name__ == "__main__":
    env = GridWorldEnv(render_mode='human', size=6, filename='mygridworld.txt')

    
    test_steps = 200

    state, info = env.reset()
    episode = 0
    print('Episode', episode)

    for step in range(test_steps):
        action = env.action_space.sample()
        print('Chosen action:', action)
        print('Info:', info)
        state, reward, terminated, _, info = env.step(action)

        if terminated:
            print('\t\tTerminated episode.')
            state, info =  env.reset()
            episode += 1
            print('Episode', episode)
    

