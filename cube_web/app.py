from flask import Flask, render_template, request, jsonify
import numpy as np
import gymnasium as gym
import rubiks_cube_gym
from q_learning_solver import QLearningSolver
from rubiks_cube_gym.envs.rubiks_cube_222 import RubiksCube222Env
from q_learning_solver_one_face import QLearningSolver as QLearningSolverOneFace
from rubiks_cube_222_one_face import RubiksCube222EnvOneFace

ACTION_MAP = {0: 'F', 1: 'R', 2: 'U', 3: "F'", 4: "R'", 5: "U'"}


app = Flask(__name__)

Q_TABLE_FILENAME = '3000w.npy'
Q_TABLE_ONE_FACE_FILENAME = '2000w_of.npy'
env = RubiksCube222Env()
solver = QLearningSolver(Q_TABLE_FILENAME)
solver_one_face = QLearningSolverOneFace(Q_TABLE_ONE_FACE_FILENAME)
current_scramble = None

@app.route('/')
def index():

    state, info = env.reset(options={"scramble": False})
    cube_reduced = env.render(mode='ansi')
    return render_template('index.html', initial_state=cube_reduced)

@app.route('/scramble', methods=['POST'])
def scramble():
    global current_scramble
    steps = int(request.json.get('steps', 11))
    scramble = solver.generate_scramble(steps)
    cube_reduced = reset_env(scramble)
    current_scramble = scramble
    return jsonify({'scramble': scramble, 'cube_reduced': cube_reduced})

@app.route('/solve', methods=['POST'])
def solve():
    global current_scramble
    current_cube = env.cube.copy()
    current_cube_reduced = env.cube_reduced
    current_cube_state = env.cube_state
    
    temp_env = RubiksCube222Env()
    temp_env.cube = current_cube.copy()
    temp_env.cube_reduced = current_cube_reduced
    temp_env.cube_state = current_cube_state

    solution = []
    steps = 0
    state = temp_env.cube_state
    done = False
    
    solution_process = []
    
    while not done and steps <= 50:
        action, q_values = solver.get_action(state)
        solution_process.append({
            'q_values': q_values,
            'action': int(action),
            'step': steps
        })
        next_state, reward, done, _, _ = temp_env.step(action)
        solution.append(ACTION_MAP[action])
        state = next_state
        steps += 1
    
    if steps <= 50 and done:
        return jsonify({
            'solution': solution, 
            'steps': steps, 
            'cube_reduced': current_cube_reduced,
            'solution_process': solution_process
        })
    return jsonify({
        'solution': [], 
        'steps': steps, 
        'cube_reduced': current_cube_reduced,
        'solution_process': []
    })


def reset_env(scramble=None):
    state, info = env.reset(options={"scramble": scramble})
    cube_reduced = env.render(mode='ansi')
    return cube_reduced

@app.route('/rotate', methods=['POST'])
def rotate():
    move = request.json['move']
    action = [k for k, v in ACTION_MAP.items() if v == move][0]
    env.step(action)
    cube_reduced = env.render(mode='ansi')
    return jsonify({'cube_reduced': cube_reduced})

@app.route('/reset', methods=['POST'])
def reset():
    cube_reduced = reset_env(scramble=False)
    return jsonify({'cube_reduced': cube_reduced})

@app.route('/check_solved', methods=['POST'])
def check_solved():

    is_solved = (env.cube_state == 0)
    return jsonify({'is_solved': is_solved})

@app.route('/next_step', methods=['POST'])
def next_step():
    if env.cube_state == 0:
        return jsonify({'next_move': None})
    
    action, q_values = solver.get_action(env.cube_state)
    next_move = ACTION_MAP[action]
    
    return jsonify({
        'next_move': next_move,
        'q_values': q_values
    })

@app.route('/check_solved_one_face', methods=['POST'])
def check_solved_one_face():
    cube_reduced = env.cube_reduced
    
    if (all(cube_reduced[i] == 'W' for i in [0,1,2,3]) or
        all(cube_reduced[i] == 'O' for i in [4,5,12,13]) or
        all(cube_reduced[i] == 'G' for i in [6,7,14,15]) or
        all(cube_reduced[i] == 'R' for i in [8,9,16,17]) or
        all(cube_reduced[i] == 'B' for i in [10,11,18,19]) or
        all(cube_reduced[i] == 'Y' for i in [20,21,22,23])):
        return jsonify({'is_face_solved': True})
    
    return jsonify({'is_face_solved': False})


@app.route('/solve_one_face', methods=['POST'])
def solve_one_face():
    current_cube = env.cube.copy()
    current_cube_reduced = env.cube_reduced
    current_cube_state = env.cube_state
    
    temp_env = RubiksCube222EnvOneFace()
    temp_env.cube = current_cube.copy()
    temp_env.cube_reduced = current_cube_reduced
    temp_env.cube_state = current_cube_state

    solution = []
    steps = 0
    state = temp_env.cube_state
    done = False
    solution_process = []
    
    while not done and steps <= 10:
        action, q_values = solver_one_face.get_action(state)
        solution_process.append({
            'q_values': q_values,
            'action': int(action),
            'step': steps
        })
        next_state, reward, done, _, _ = temp_env.step(action)
        solution.append(ACTION_MAP[action])
        state = next_state
        steps += 1
    
    if steps <= 10 and done:
        return jsonify({
            'solution': solution, 
            'cube_reduced': current_cube_reduced,
            'solution_process': solution_process
        })
    return jsonify({
        'solution': [], 
        'cube_reduced': current_cube_reduced,
        'solution_process': []
    })

@app.route('/next_step_one_face', methods=['POST'])
def next_step_one_face():
    if env.cube_state == 0:
        return jsonify({'next_move': None})
    
    action, q_values = solver_one_face.get_action(env.cube_state)
    next_move = ACTION_MAP[action]
    
    return jsonify({
        'next_move': next_move,
        'q_values': q_values
    })

if __name__ == "__main__":
    app.run(debug=True)