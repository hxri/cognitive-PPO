from operator import add

from gymnasium.spaces import Discrete

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Goal
from minigrid.minigrid_env import MiniGridEnv

import torch


class DynamicObstaclesEnv(MiniGridEnv):
    """
    ### Description

    This environment is an empty room with moving obstacles.
    The goal of the agent is to reach the green goal square without colliding
    with any obstacle. A large penalty is subtracted if the agent collides with
    an obstacle and the episode finishes. This environment is useful to test
    Dynamic Obstacle Avoidance for mobile robots with Reinforcement Learning in
    Partial Observability.

    ### Mission Space

    "get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure. A '-1' penalty is
    subtracted if the agent collides with an obstacle.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent collides with an obstacle.
    3. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-Dynamic-Obstacles-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-8x8-v0`
    - `MiniGrid-Dynamic-Obstacles-16x16-v0`

    """

    def __init__(
        self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, n_obstacles=4, max_steps=4, n_goals=1, dynamic_wall=False, dynamic_goal=False, dynamic_obstacles=False, moving_goal=False, **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)
        self.dynamic_wall = dynamic_wall
        self.dynamic_goal = dynamic_goal
        self.dynamic_obstacles = dynamic_obstacles
        self.moving_goal = moving_goal
        self.n_goals = n_goals

    def _gen_grid(self, width, height):

        if not self.dynamic_wall:
            # Create an empty grid
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.wall_rect(0, 0, width, height)

        else:
            # Create the grid
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height - 1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width - 1, 0)

            room_w = width // 2
            room_h = height // 2

            # For each row of rooms
            for j in range(0, 2):

                # For each column
                for i in range(0, 2):
                    xL = i * room_w
                    yT = j * room_h
                    xR = xL + room_w
                    yB = yT + room_h

                    # Bottom wall and door
                    if i + 1 < 2:
                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))
                        self.grid.set(*pos, None)

                    # Bottom wall and door
                    if j + 1 < 2:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)

        if not self.dynamic_goal:
            # Place a goal square in the bottom-right corner
            self.grid.set(width - 2, height - 2, Goal())
        else:
            # Place dynamic goal
            self.goals = []
            for i_goal in range(self.n_goals):
                self.goals.append(Goal())
                self.place_obj(self.goals[i_goal], max_tries=100)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"

        if self.dynamic_obstacles:
            # Update obstacle positions
            for i_obst in range(len(self.obstacles)):
                old_pos = self.obstacles[i_obst].cur_pos
                top = tuple(map(add, old_pos, (-1, -1)))

                try:
                    self.place_obj(
                        self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                    )
                    self.grid.set(old_pos[0], old_pos[1], None)
                except Exception:
                    pass
        
        if self.moving_goal:
            for i_goal in range(len(self.goals)):
                old_pos_goal = self.goals[i_goal].cur_pos
                top1 = tuple(map(add, old_pos_goal, (-1, -1)))
                try:
                    self.place_obj(
                        self.goals[i_goal], top=top1, size=(3, 3), max_tries=100
                    )
                    self.grid.set(old_pos_goal[0], old_pos_goal[1], None)
                except Exception:
                    pass


        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info
        
        for i_goal in range(len(self.goals)):
            if super().agent_sees(self.goals[i_goal].cur_pos[0], self.goals[i_goal].cur_pos[1]):
                gc = 1
            else:
                gc = 0
        # print("Goal congruence: {}" .format(gc))
        
        x = 0
        for i_obst in range(len(self.obstacles)):
            if super().agent_sees(self.obstacles[i_obst].cur_pos[0], self.obstacles[i_obst].cur_pos[1]):
                x += 1
        cp = 1 - (x/(len(self.obstacles) + 0.001))
        # print("Coping Potential : {}" .format(cp))
        
        info['goal_congruence'] = gc
        info['coping_potential'] = cp

        return obs, reward, terminated, truncated, info
