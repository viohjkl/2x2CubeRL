import numpy as np
import gymnasium as gym
import rubiks_cube_gym
import os

ACTION_MAP = {0: 'F', 1: 'R', 2: 'U', 3: "F'", 4: "R'", 5: "U'"}

class QLearningSolver:
    def __init__(self, q_table_filename):
        if os.path.exists(q_table_filename):
            self.q_table = np.load(q_table_filename)
            self.n_actions = 6
            self.epsilon = 0.01
            print(f"Loaded Q table from {q_table_filename}")
        else:
            raise FileNotFoundError(f"Q table file {q_table_filename} not found.")
   
    def get_action(self, state):
        q_values = self.q_table[state].copy()
        return np.argmax(q_values), q_values.tolist()

    def generate_scramble(self, scramble_steps):
        moves = ['F', 'R', 'U']
        move_type = ['', '2', "'"]
        scramble = []
        prev_move = None

        for _ in range(scramble_steps):
            move = np.random.choice(moves)
            while move == prev_move:
                move = np.random.choice(moves)
            scramble.append(move + np.random.choice(move_type))
            prev_move = move

        return ' '.join(scramble)
    
    def solve(self, env, scramble_steps=11, scramble=None):
        if scramble is None:
            scramble = env.generate_scramble(scramble_steps)
        state, _ = env.reset(options={"scramble": scramble})
        done = False
        steps = 0
        solution = []

        while not done:
            action, q_values = self.get_action(state)
            next_state, reward, done, _, info = env.step(action)
            solution.append(ACTION_MAP[action])
            state = next_state
            steps += 1

            if steps > 50:
                break

        return solution, steps, q_values

if __name__ == "__main__":
    Q_TABLE_FILENAME = '3000w.npy'
    SCRAMBLE_STEPS = 12

    env = gym.make('rubiks-cube-222-v0')
    solver = QLearningSolver(Q_TABLE_FILENAME)
    solved = 0


    '''
    for _ in range(10000):
        scramble = solver.generate_scramble(scramble_steps=SCRAMBLE_STEPS)
        solution, steps = solver.solve(env, scramble_steps=SCRAMBLE_STEPS, scramble=scramble)
        if len(solution) <= 50:
            solved += 1
    
    print(f"Solved {solved} out of 10000 Rubik's Cubes.")

    '''
    print("Generating a scramble...")
    scramble = solver.generate_scramble(scramble_steps=SCRAMBLE_STEPS)
    print(f"Scramble: {scramble}")

    print("Solving...")
    solution, steps = solver.solve(env, scramble_steps=SCRAMBLE_STEPS, scramble=scramble)

    if len(solution) <= 50:
        print(f"Solution found in {steps} steps: {' '.join(solution)}")
    