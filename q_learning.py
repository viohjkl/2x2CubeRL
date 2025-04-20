import json
import gymnasium as gym
import numpy as np
import rubiks_cube_gym
import time
import os

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, q_table_filename=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        if q_table_filename is not None and os.path.exists(q_table_filename):
            self.q_table = np.load(q_table_filename)
        else:
            self.q_table = np.zeros((n_states, n_actions))
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):

        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max * (1 - done))
        self.q_table[state, action] = new_value
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self, filename):
        np.save(filename, self.q_table)

def save_history(avg_rewards, avg_steps, solve_rate, solve_steps, filename):
    history = {
        'avg_rewards': avg_rewards,
        'avg_steps': avg_steps,
        'solve_rate': solve_rate,
        'solve_steps': solve_steps
    }
    
    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            data = json.load(f)
            if isinstance(data, list):
                data.append(history)
            else:
                data = [data, history]
            f.seek(0)
            json.dump(data, f)
    else:
        with open(filename, 'w') as f:
            json.dump([history], f)

def train_agent(n_episodes=1000, max_steps=50, render=False, q_table_filename=None):

    #   env = gym.make('rubiks-cube-222-one-face-v0')
    env = gym.make('rubiks-cube-222-v0') 

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.2,
        gamma=0.99,
        epsilon=0.05,
        epsilon_decay=0.9,
        epsilon_min=0.05,
        q_table_filename=q_table_filename
    )

    rewards_history = []
    steps_history = []
    solve_history = []
    solved_episodes = 0
    start_time = time.time()
    scramble = 12
    
    for episode in range(n_episodes): 
        state, _ = env.reset(scramble_step=scramble)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):

            action = agent.get_action(state)
            
            next_state, reward, done, _, _ = env.step(action)

            steps += 1
            
            total_reward += reward

            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.1)
            
            if done:
                solved_episodes += 1
                solve_history.append(steps)
                break
        
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        
        episodes = 1000
        if (episode + 1) % episodes == 0:
            avg_reward = np.mean(rewards_history[-episodes:])
            avg_steps = np.mean(steps_history[-episodes:])
            solve_rate = solved_episodes / (episodes) * 100
            solve_steps = np.mean(solve_history[-solved_episodes:])
            print(f"Episode: {episode + 1:,}")
            print(f"Scramble Steps: {scramble}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Steps: {avg_steps:.2f}")
            print(f"Average Steps to Solve: {solve_steps:.2f}") 
            print(f"Solve Rate: {solve_rate:.2f}%")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Time: {time.time() - start_time:.2f}s")
            print("------------------------")

            
            if solve_rate >= 90 and scramble != 12:
                q_table_filename = f'q_table_scramble_{scramble}_solve_rate_{solve_rate:.2f}_avg_steps_{solve_steps:.2f}.npy'
                agent.save_q_table(q_table_filename)
                print(f"Saved Q table to {q_table_filename}")

            if scramble == 12 and (episode + 1) % 1000000 == 0:
                q_table_filename = f'q_table_scramble_{scramble}_solve_rate_{solve_rate:.2f}_avg_steps_{solve_steps:.2f}.npy'
                agent.save_q_table(q_table_filename)
                print(f"Saved Q table to {q_table_filename}")

            if solve_rate >= 90 and scramble < 8:
                scramble += 1
                print(f"Increasing scramble steps to {scramble}")
            elif solve_rate >= 90 and scramble < 12:
                scramble += 2
                print(f"Increasing scramble steps to {scramble}")

            solved_episodes = 0

            if episode == n_episodes - 1:
                q_table_filename = f'q_table_scramble_{scramble}_solve_rate_{solve_rate:.2f}_avg_steps_{solve_steps:.2f}.npy'
                agent.save_q_table(q_table_filename)
                print(f"Saved Q table to {q_table_filename}")

        
            save_history(avg_reward, avg_steps, solve_rate, solve_steps, 'q_history.json')
            rewards_history.clear()
            steps_history.clear()
            solve_history.clear()

    env.close()
    return agent, rewards_history, steps_history

if __name__ == "__main__":

    N_EPISODES = 6000000
    MAX_STEPS = 50
    RENDER = False
    Q_TABLE_FILENAME = None

    print("Starting training...")
    agent, rewards, steps = train_agent(
        q_table_filename=Q_TABLE_FILENAME,
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS,
        render=RENDER
    )
    print("Training completed!")