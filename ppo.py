import json
import gymnasium as gym
import numpy as np
import rubiks_cube_gym
import time
import os
import pickle
from rubiks_cube_gym.envs.rubiks_cube_222 import RubiksCube222Env

class PPOAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01,
                 ppo_epochs=4, model_path=None): 

        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.policy_table = np.ones((n_states, n_actions)) / n_actions
            self.value_table = np.zeros(n_states)
        
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        # self.mini_batch_size = 250

    def get_action(self, state, training=True):
        action_probs = self.policy_table[state]
        
        if training:
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            action = np.argmax(action_probs)
        
        log_prob = np.log(action_probs[action] + 1e-10)
        
        return action, log_prob
    
    def update(self, memory):
        states = memory['states']
        actions = memory['actions']
        old_log_probs = memory['log_probs']
        rewards = memory['rewards']
        next_states = memory['next_states']
        dones = memory['dones']
        
        advantages = []
        returns = []
        values = np.array([self.value_table[s] for s in states])
        next_values = np.array([self.value_table[s] for s in next_states])
        
        gae = 0
        success_weight = 1 
        for i in reversed(range(len(rewards))):
            if rewards[i] > 0:
                delta = rewards[i] * success_weight + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            else:
                delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):

            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                old_log_prob = old_log_probs[i]
                advantage = advantages[i]
                ret = returns[i]
                
                action_probs = self.policy_table[state]
                log_prob = np.log(action_probs[action] + 1e-10)
                
                ratio = np.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
                
                policy_loss = -min(surr1, surr2)
                
                entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
                
                grad = np.zeros_like(action_probs)
                grad[action] = policy_loss - self.entropy_coef * entropy
                self.policy_table[state] -= self.lr * grad
                
                self.policy_table[state] = np.clip(self.policy_table[state], 1e-10, 1.0)
                self.policy_table[state] /= np.sum(self.policy_table[state])
                
                value_error = ret - self.value_table[state]
                self.value_table[state] += self.lr * value_error
    
    def save(self, path):
        data = {
            'policy_table': self.policy_table,
            'value_table': self.value_table
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.policy_table = data['policy_table']
        self.value_table = data['value_table']

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

def train_agent(n_episodes=1000000, max_steps=50, update_interval=1000, 
                eval_interval=1000, model_path=None):

    env = gym.make('rubiks-cube-222-v0')
    
    agent = PPOAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        lr=0.1,           
        gamma=0.99,       
        gae_lambda=0.95,   
        clip_epsilon=0.1,
        entropy_coef=0.1,
        ppo_epochs=1,
        # mini_batch_size=250,
        model_path=model_path
    )
    
    memory = {
        'states': [],
        'actions': [],
        'log_probs': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }
    
    rewards_history = []
    steps_history = []
    solve_history = []
    solved_episodes = 0
    scramble = 12
    
    total_steps = 0
    start_time = time.time()
    
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(scramble_step=scramble)
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:

            action, log_prob = agent.get_action(state)
            
            next_state, reward, done, _, _ = env.step(action)
            
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['next_states'].append(next_state)
            memory['dones'].append(done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1
            
            if total_steps % update_interval == 0:
                agent.update(memory)
                for key in memory:
                    memory[key] = []
            
            if done:
                solved_episodes += 1
                solve_history.append(steps)
                break
        
        rewards_history.append(episode_reward)
        steps_history.append(steps)
        
        if episode % eval_interval == 0:
            avg_reward = np.mean(rewards_history[-eval_interval:])
            avg_steps = np.mean(steps_history[-eval_interval:])
            solve_rate = solved_episodes / eval_interval * 100
            solve_steps = np.mean(solve_history[-solved_episodes:]) if solved_episodes > 0 else 0
            
            print(f"Episode: {episode:,}")
            print(f"Scramble Steps: {scramble}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Steps: {avg_steps:.2f}")
            print(f"Average Steps to Solve: {solve_steps:.2f}") 
            print(f"Solve Rate: {solve_rate:.2f}%")
            print(f"Time: {time.time() - start_time:.2f}s")
            print("------------------------")
            

            if solve_rate >= 90 and scramble != 12:
                model_filename = f'ppo_model_scramble_{scramble}_solve_rate_{solve_rate:.2f}_avg_steps_{solve_steps:.2f}.pkl'
                agent.save(model_filename)
                print(f"saved {model_filename}")
            
            if scramble == 12 and episode % 1000000 == 0:
                model_filename = f'ppo_model_scramble_{scramble}_solve_rate_{solve_rate:.2f}_avg_steps_{solve_steps:.2f}.pkl'
                agent.save(model_filename)
                print(f"saved {model_filename}")
            
            if solve_rate >= 90 and scramble < 12:
                scramble += 1
                print(f"scramble: {scramble}")
            
            save_history(avg_reward, avg_steps, solve_rate, solve_steps, 'ppo_history.json')
            
            solved_episodes = 0
            rewards_history = []
            steps_history = []
            solve_history = []

            if episode == n_episodes:
                model_filename = f'ppo_model_final_scramble_{scramble}_solve_rate_{solve_rate:.2f}.pkl'
                agent.save(model_filename)
                print(f"saved {model_filename}")
    
    env.close()
    return agent

if __name__ == "__main__":
    N_EPISODES = 10000000
    MAX_STEPS = 50
    UPDATE_INTERVAL = 1000
    EVAL_INTERVAL = 1000
    MODEL_PATH = None

    print("loading {MODEL_PATH}" if MODEL_PATH else "start training...")
    agent = train_agent(
        n_episodes=N_EPISODES,
        max_steps=MAX_STEPS,
        update_interval=UPDATE_INTERVAL,    
        eval_interval=EVAL_INTERVAL,
        model_path=MODEL_PATH,
    )
    print("training complete!")