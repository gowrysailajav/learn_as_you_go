import gym
from gym import spaces
from s2_1_agent_conf import agent_configs, num_agents
import numpy as np
import random
import pygame
import pickle
import os
import cv2 

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)

with open('./Testing/result_clusters.pkl', 'rb') as f:
    result_clusters = pickle.load(f)

# with open('./Testing/first_aoi.pkl', 'rb') as f:
#     aoi = pickle.load(f)

min_val = 0
max_val = grid_size
num_agents = num_agents
aoi = []
survivors = []

_action_to_direction = {
    0: np.array([1, 0]),   # east
    1: np.array([0, 1]),   # north
    2: np.array([-1, 0]),  # west
    3: np.array([0, -1]),  # south
    4: np.array([1, 1]),   # ne
    5: np.array([-1, 1]),  # nw
    6: np.array([-1, -1]), # sw
    7: np.array([1, -1]),  # se
    8: np.array([0, 0])    # Hover 
    }
actions = _action_to_direction.keys()
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "grb_array"], "render_fps": 4}

    def __init__(self, min_val, max_val, render_mode="human"):
        super(GridWorldEnv, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(9)
        self._action_to_direction = _action_to_direction
        self.actions = self._action_to_direction.keys()
        # Defining the state space and observation space for each agent
        self.agent_state_spaces = [spaces.Discrete(grid_size) for _ in range(num_agents)]
        self.agent_observation_spaces = [spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype = np.float32) for _ in range(num_agents)]
        # Defining initial positions of the agents
        self.agent_positions = [[0, 0] for _ in range(num_agents)]
        # self.agent_positions = [[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] for _ in range(num_agents)]
        # print("self.agent_positions : ", self.agent_positions)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 800
        self.result_clusters = result_clusters
        self.aoi = aoi
        self.survivors = survivors

    # reset method is used to reset the environment to the initial state
    def reset(self):
        for agent_config in agent_configs:
            agent_config.start_state = None
            agent_config.current_state = None
        # self.agent_positions = [[random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)] for _ in range(self.num_agents)]
        return [self._get_observation(agent_id) for agent_id in range(1, self.num_agents + 1)]

    # _get_observation method fetches the observation for a specific agent
    def _get_observation(self, agent_id):
        # print("agent_id : ", agent_id)
        if 1 <= agent_id <= self.num_agents:
            observation = np.zeros((self.grid_size, self.grid_size), dtype = np.float32)
            # print("observation shape : ", observation.shape)
            position = self.agent_positions[agent_id - 1] 
            position = np.clip(position, 0, self.grid_size - 1)
            # print("position : ", position)
            observation[position[0].astype(int), position[1].astype(int)] = 1.0
            return observation
        else:
            raise ValueError("Invalid agent_id")

    # _move_agent method is used to update the agent position considering the action chosen
    def _move_agent(self, agent_position, action, agent_speed):
        direction = list(self._action_to_direction[action])
        new_position = [
            agent_position[0] + agent_speed * direction[0],
            agent_position[1] + agent_speed * direction[1]
        ]
        new_position[0] = np.clip(new_position[0], 0, self.grid_size - 1)
        new_position[1] = np.clip(new_position[1], 0, self.grid_size - 1)
        agent_position[0], agent_position[1] = new_position[0], new_position[1]
        return agent_position
        
    def calculate_reward(self):
        reward = 0.0
        for position in self.agent_positions:
            if position == [self.grid_size - 1, self.grid_size - 1]:
                reward += 1.0
        return reward

    def _render_frame(self, _agent_location, agent_movements):
        traversed_aoi_path = './Testing/traversed_aoi.pkl'
        if os.path.exists(traversed_aoi_path):
            with open(traversed_aoi_path, 'rb') as f:
                self.traversedaoi = pickle.load(f)
        else:
            self.traversedaoi = []
        background_image_path = './Training/c2sparse2.jpg'
        aoi = './Testing/first_aoi.pkl'
        if os.path.exists(aoi):
            with open(aoi, 'rb') as f:
                self.aoi = pickle.load(f)
        background_image_cv = cv2.imread(background_image_path)
        background_image_cv = cv2.resize(background_image_cv, (self.window_size, self.window_size))
        blurred_image_cv = cv2.GaussianBlur(background_image_cv, (21, 21), 0)
        blurred_image_cv = cv2.cvtColor(blurred_image_cv, cv2.COLOR_BGR2RGB)
        background_image = pygame.surfarray.make_surface(np.transpose(blurred_image_cv, (1, 0, 2)))


        # background_image = pygame.image.load(background_image)
        # background_image = pygame.transform.scale(background_image, (self.window_size, self.window_size))
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))
        if self.render_mode == "human":
            canvas.blit(background_image, (0, 0))  # Draw background image
        else:
            canvas.fill((255, 255, 255))  # Fill with white for rgb_array mode
        pix_square_size = self.window_size / self.grid_size
        # Draw AOI rectangles
        for location in self.aoi:
            # print(f'len = {len(self.aoi)}')
            if location in self.traversedaoi:
                color = (0, 255, 0, 128)  # Green color with adjustable transparency for traversed AOI
            else:
                color = (255, 255, 153, 128)  # Yellow color with adjustable transparency for other AOI
        
            pygame.draw.rect(
                canvas,
                color,  
                pygame.Rect(
                    pix_square_size * location[0],
                    pix_square_size * location[1],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        # Draw survivor icons
        user_icon = pygame.image.load("./user_icon.png")
        user_icon = pygame.transform.scale(user_icon, (int(pix_square_size * 0.6), int(pix_square_size * 0.6)))
        for location in self.survivors:
            x = location[0] * pix_square_size + int(pix_square_size * 0.2)
            y = location[1] * pix_square_size + int(pix_square_size * 0.2)
            canvas.blit(user_icon, (x, y))
        # Draw agent movements
        for i, (_agent_loc, movement) in enumerate(zip(_agent_location, agent_movements)):
            agent_center = (int((_agent_loc[0] + 0.5 ) * pix_square_size), int((_agent_loc[1] + 0.5 ) * pix_square_size))
            # Highlight agent positions with different colors
            if movement == "hover":
                pygame.draw.circle(canvas, (255, 0, 0), agent_center, int(pix_square_size / 3))  # Red for hover
            else:
                pygame.draw.circle(canvas, (0, 0, 255), agent_center, int(pix_square_size / 3))  # Blue for movement
        # Draw grid lines
        grid_color = (128, 128, 128)
        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas, grid_color, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=1)
            pygame.draw.line(canvas, grid_color, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=1)
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    
    def change_icon_color(self, icon, new_color):
        # Create a copy of the original icon
        colored_icon = icon.copy()
        # Iterate over each pixel and change its color
        for y in range(colored_icon.get_height()):
            for x in range(colored_icon.get_width()):
                # Get the color of the current pixel
                color = colored_icon.get_at((x, y))
                # If the pixel is not transparent, change its color
                if color.a > 0:
                    colored_icon.set_at((x, y), new_color)
        return colored_icon


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    # step method 
    def step(self, actions):
        _agent_location = []
        agent_movements = []
        # Take action for each agent and update the environment
        for i, agent_config in enumerate(agent_configs):
            # print("self.agent_positions[i] : ", self.agent_positions[i])
            self._move_agent(self.agent_positions[i], actions[i], agent_config.agent_speed)
            # self._move_agent(self.agent_positions[i], actions[i])
            _agent_location.append(self.agent_positions[i])
            if actions[i] == 8:
                agent_movements.append("hover")
            else:
                agent_movements.append("move")
        # calculate reward 
        reward = self.calculate_reward()
        # implement terminal condition here
        done = False
        if self.render_mode == "human":
            self._render_frame(_agent_location, actions)
            self._render_frame(_agent_location, agent_movements)
        return [self._get_observation(agent_id + 1) for agent_id in range(self.num_agents)], reward, done


GridWorld = GridWorldEnv(grid_size, num_agents)
observations = GridWorld.reset()

