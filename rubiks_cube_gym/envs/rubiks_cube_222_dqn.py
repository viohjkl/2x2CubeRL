import pickle
import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
import gc
import wget


class RubiksCube222DQNEnv(gym.Env):

    def __init__(self, initial_scramble_steps=1, max_scramble_steps=12):
        self.cube = None
        self.cube_reduced = None
        self.cube_state = None
        self.episode_steps = []
        
        self.initial_scramble_steps = initial_scramble_steps
        self.max_scramble_steps = max_scramble_steps
        self.current_scramble_steps = initial_scramble_steps
        self.episode_count = 0
        self.success_count = 0
        self.solve_rate_threshold = 0.9
        
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=1, shape=(144,), dtype=np.uint8)

        state_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rubiks_cube_222_states_FRU.pickle")
        if not os.path.exists(state_file):
            print("State file not found")
            print("Downloading...")
            wget.download("https://storage.googleapis.com/rubiks_cube_gym/rubiks_cube_222_states_FRU.pickle", state_file)
            print("Download complete")

        with open(state_file, "rb") as f:
            self.cube_states = pickle.load(f)

    def update_cube_reduced(self):
        self.cube_reduced = ''.join(TILE_MAP[tile] for tile in self.cube)

    def update_cube_state(self):
        self.cube_state = self.cube_states[self.cube_reduced]

    def generate_scramble(self, scramble_step):
        scramble_len = 0
        prev_move = None
        scramble = []
        moves = ['F', 'R', 'U']
        move_type = ['', '2', "'"]

        while scramble_len < scramble_step:
            move = np.random.choice(moves)
            while move == prev_move:
                move = np.random.choice(moves)
            scramble.append(move + np.random.choice(move_type))
            prev_move = move
            scramble_len += 1

        return ' '.join(scramble)

    def move(self, move_side, move_type=None):
        repetitions = dict({None: 1, "2": 2, "'": 3})[move_type]

        if move_side == "R":
            side_cubies_old = np.array([1, 3, 7, 15, 21, 23, 18, 10])
            face_cubies_old = np.array([[8, 9], [16, 17]])
        elif move_side == "L":
            side_cubies_old = np.array([2, 0, 11, 19, 22, 20, 14, 6])
            face_cubies_old = np.array([[4, 5], [12, 13]])
        elif move_side == "F":
            side_cubies_old = np.array([2, 3, 13, 5, 21, 20, 8, 16])
            face_cubies_old = np.array([[6, 7], [14, 15]])
        elif move_side == "B":
            side_cubies_old = np.array([0, 1, 9, 17, 23, 22, 12, 4])
            face_cubies_old = np.array([[10, 11], [18, 19]])
        elif move_side == "U":
            side_cubies_old = np.array([6, 7, 8, 9, 10, 11, 4, 5])
            face_cubies_old = np.array([[0, 1], [2, 3]])
        elif move_side == "D":
            side_cubies_old = np.array([14, 15, 12, 13, 18, 19, 16, 17])
            face_cubies_old = np.array([[20, 21], [22, 23]])

        side_cubies_new = np.roll(side_cubies_old, -2 * repetitions)
        face_cubies_new = np.rot90(face_cubies_old, 4 - repetitions).flatten()
        face_cubies_old = face_cubies_old.flatten()

        np.put(self.cube, side_cubies_old, self.cube[side_cubies_new])
        np.put(self.cube, face_cubies_old, self.cube[face_cubies_new])

    def rotate_cube(self, rotation):
        rotation_maps = {
            "x'": [18,10,19,11,12,4,2,0,16,8,23,21,13,5,3,1,17,9,22,20,14,6,15,7],
            "x": [7,15,6,14,5,13,21,23,9,17,1,3,4,12,20,22,8,16,0,2,11,19,10,18],
            "y": [2,0,3,1,6,7,8,9,10,11,12,13,14,15,16,17,18,19,4,5,22,20,23,21],
            "y'": [1,3,0,2,18,19,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,23,22],
            "z": [12,4,13,5,20,21,14,6,2,0,18,10,22,23,15,7,3,1,19,11,8,9,16,17],
            "z'": [9,17,8,16,1,3,7,15,21,23,11,19,0,2,6,14,20,22,10,18,5,13,4,12]
        }
        if rotation in rotation_maps:
            new_cube = np.zeros(24, dtype=np.uint8)
            np.put(new_cube, range(24), self.cube[rotation_maps[rotation]])
            self.cube = new_cube
            self.update_cube_reduced()

    def algorithm(self, moves):
        for move in moves.split(" "):
            if len(move) == 2:
                self.move(move[0], move[1])
            else:
                self.move(move[0])

    def step(self, action):
        move = ACTION_MAP[action]
        self.move(move[0], move[1])
        self.episode_steps.append(action)

        self.update_cube_reduced()
        self.update_cube_state()

        reward, done = self.reward()
        
        observation = self.get_observation()
        info = {"cube": self.cube, "cube_reduced": self.cube_reduced,
                "scramble_steps": self.current_scramble_steps,
                "episode": {
                    "r": reward,
                    "l": len(self.episode_steps)
                }
            }

        return observation, reward, done, False, info

    def get_observation(self):
        color_to_num = {'W': 0, 'O': 1, 'G': 2, 'R': 3, 'B': 4, 'Y': 5}
        color_indices = np.array([color_to_num[TILE_MAP[i]] for i in self.cube], dtype=np.uint8)
        one_hot = np.zeros((24, 6), dtype=np.uint8)
        one_hot[np.arange(24), color_indices] = 1
        return one_hot.flatten()


    def reward(self):
        if self.cube_reduced == "WWWWOOGGRRBBOOGGRRBBYYYY":
            self.success_count += 1
            return 100, True
        else:
            return -1, False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = []

        self.episode_count += 1
        
        if self.episode_count % 100 == 0 and self.episode_count > 0:
            solve_rate = self.success_count / 100
            if solve_rate >= self.solve_rate_threshold and self.current_scramble_steps < self.max_scramble_steps:
                self.current_scramble_steps = min(self.current_scramble_steps + 1, self.max_scramble_steps)
            self.success_count = 0

        self.cube = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                            dtype=np.uint8)
        
        scramble = None if options is None else options.get("scramble")
        if scramble:
            self.algorithm(scramble)
        elif scramble == False:
            pass
        else:
            self.algorithm(self.generate_scramble(self.current_scramble_steps))

        self.update_cube_reduced()
        self.update_cube_state()

        info = {"cube": self.cube, "cube_reduced": self.cube_reduced, 
                "scramble_steps": self.current_scramble_steps}

        return self.get_observation(), info

    def close(self):
        del self.cube_states
        gc.collect()


TILE_MAP = {
    0: 'W', 1: 'W', 2: 'W', 3: 'W',
    4: 'O', 5: 'O', 6: 'G', 7: 'G', 8: 'R', 9: 'R', 10: 'B', 11: 'B',
    12: 'O', 13: 'O', 14: 'G', 15: 'G', 16: 'R', 17: 'R', 18: 'B', 19: 'B',
    20: 'Y', 21: 'Y', 22: 'Y', 23: 'Y'
}

COLOR_MAP = {
    'W': (255, 255, 255),
    'O': (255, 165, 0),
    'G': (0, 128, 0),
    'R': (255, 0, 0),
    'B': (0, 0, 255),
    'Y': (255, 255, 0)
}

ACTION_MAP = {
    0: ("F", None), 1: ("R", None), 2: ("U", None),
    3: ("F", "'"), 4: ("R", "'"), 5: ("U", "'")
}