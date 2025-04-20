import gymnasium as gym
import numpy as np
import rubiks_cube_gym
import time
from ppo import PPOAgent

ACTION_MAP = {0: 'F', 1: 'R', 2: 'U', 3: "F'", 4: "R'", 5: "U'"}

class PPOSolver:
    def __init__(self, model_path):
        self.env = gym.make('rubiks-cube-222-v0')
        self.agent = PPOAgent(
            n_states=self.env.observation_space.n,
            n_actions=self.env.action_space.n,
            model_path=model_path
        )
    
    def get_action(self, state):
        action, _ = self.agent.get_action(state, training=False)
        return action

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
    
    def solve(self, scramble_steps=11, scramble=None):
        if scramble is None:
            scramble = self.generate_scramble(scramble_steps)
        state, _ = self.env.reset(options={"scramble": scramble})
        done = False
        steps = 0
        solution = []

        while not done:
            action = self.get_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            solution.append(ACTION_MAP[action])
            state = next_state
            steps += 1

            if steps > 50:
                break

        return solution, steps

if __name__ == "__main__":
    MODEL_PATH = "1000w.pkl"
    SCRAMBLE_STEPS = 12
    N_TESTS = 10000

    solver = PPOSolver(MODEL_PATH)
    solved = 0
    total_steps = 0

    print(f"Testing model {MODEL_PATH} with {SCRAMBLE_STEPS} scramble steps...")
    print(f"Running {N_TESTS} tests...")

    for i in range(N_TESTS):
        scramble = solver.generate_scramble(scramble_steps=SCRAMBLE_STEPS)
        solution, steps = solver.solve(scramble_steps=SCRAMBLE_STEPS, scramble=scramble)
        if len(solution) <= 50:
            solved += 1
            total_steps += steps
    avg_steps = total_steps / N_TESTS
    solve_rate = (solved / N_TESTS) * 100

    print(f"\nResults:")
    print(f"Solved {solved} out of {N_TESTS} cubes ({solve_rate:.2f}%)")
    print(f"Average steps: {avg_steps:.2f}")

    '''
    scramble = solver.generate_scramble(scramble_steps=SCRAMBLE_STEPS)
    print(f"Scramble: {scramble}")
    solution, steps = solver.solve(scramble_steps=SCRAMBLE_STEPS, scramble=scramble)
    if len(solution) <= 50:
        print(f"Solution found in {steps} steps: {' '.join(solution)}")
    else:
        print("No solution found within 50 steps")
    '''