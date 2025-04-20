from rubiks_cube_gym.envs.rubiks_cube_222 import RubiksCube222Env
import numpy as np
from operator import itemgetter


class RubiksCube222EnvOneFace(RubiksCube222Env):
    def __init__(self):
        super(RubiksCube222EnvOneFace, self).__init__()

    def reward(self):
        if all(self.cube_reduced[i] == 'W' for i in [0,1,2,3]):
            return 100, True
        if all(self.cube_reduced[i] == 'O' for i in [4,5,12,13]):
            return 100, True
        if all(self.cube_reduced[i] == 'G' for i in [6,7,14,15]):
            return 100, True
        if all(self.cube_reduced[i] == 'R' for i in [8,9,16,17]):
            return 100, True
        if all(self.cube_reduced[i] == 'B' for i in [10,11,18,19]):
            return 100, True
        if all(self.cube_reduced[i] == 'Y' for i in [20,21,22,23]):
            return 100, True
        else:
            return -10, False
        
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