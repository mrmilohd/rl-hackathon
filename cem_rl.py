import numpy as np
from environment import MysteryControlEnv

class CEMRLAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
        # --- MLP Neural Network Architecture ---
        # 4 (Input) -> 16 (Hidden) -> 3 (Output)
        self.input_size = observation_space.shape[0]
        self.hidden_size = 16
        self.output_size = action_space.shape[0]
        
        # Calculate sizes to slice the 1D flat weight array later
        self.s_w1 = self.input_size * self.hidden_size
        self.s_b1 = self.hidden_size
        self.s_w2 = self.hidden_size * self.output_size
        self.s_b2 = self.output_size
        
        # Total parameters = (4*16) + 16 + (16*3) + 3 = 131 parameters
        self.num_params = self.s_w1 + self.s_b1 + self.s_w2 + self.s_b2
        
        self.best_weights = np.zeros(self.num_params)

    def unpack_weights(self, flat_weights):
        """Slices the 1D evolved array back into standard Neural Network layers."""
        idx = 0
        
        W1 = flat_weights[idx:idx+self.s_w1].reshape(self.input_size, self.hidden_size)
        idx += self.s_w1
        
        b1 = flat_weights[idx:idx+self.s_b1]
        idx += self.s_b1
        
        W2 = flat_weights[idx:idx+self.s_w2].reshape(self.hidden_size, self.output_size)
        idx += self.s_w2
        
        b2 = flat_weights[idx:idx+self.s_b2]
        
        return W1, b1, W2, b2

    def act(self, observation, weights=None):
        if weights is None:
            weights = self.best_weights
            
        W1, b1, W2, b2 = self.unpack_weights(weights)
        
        # 1. Normalize Observation
        x = observation / 100.0
        
        # 2. Hidden Layer (Uses ReLU Activation)
        hidden = np.dot(x, W1) + b1
        hidden = np.maximum(0, hidden)  # ReLU: max(0, x) prevents negative values
        
        # 3. Output Layer (Uses Sigmoid Activation)
        raw_action = np.dot(hidden, W2) + b2
        
        # Clip to prevent math overflow warnings before sending to EXP
        raw_action = np.clip(raw_action, -10.0, 10.0) 
        
        action = 1.0 / (1.0 + np.exp(-raw_action))
        
        return action

    def reward_function(self, state, action, next_state, terminated, truncated):
        """
        NEW REWARD FUNCTION: Positive Reinforcement!
        Instead of punishing errors, we reward survival and accuracy.
        """
        target_p = state[2]
        target_t = state[3]
        next_p = next_state[0]
        next_t = next_state[1]
        
        p_error = abs(next_p - target_p)
        t_error = abs(next_t - target_t)
        
        # 1. BASE SURVIVAL REWARD (+10 points every step)
        # Reaching 200 steps automatically guarantees 2000 points!
        reward = 10.0
        
        # 2. ACCURACY REWARD (Closer to target = more points. Max +20 combined per step)
        p_score = max(0, (100.0 - p_error) / 10.0)
        t_score = max(0, (100.0 - t_error) / 10.0)
        reward += (p_score + t_score)
        
        # 3. EFFICIENCY PENALTY
        # Small deduction for having both Inlet and Outlet open (wasteful)
        reward -= (action[0] * action[1]) * 2.0
        
        # 4. DEATH PENALTY (Overrides hoarding points if it dies early)
        if terminated:
            reward -= 100.0
            
        return reward


def train_cem():
    print("Starting Deep CEM (MLP) RL Training with Survival Rewards...")
    env = MysteryControlEnv()
    agent = CEMRLAgent(env.action_space, env.observation_space)
    
    # --- Deep CEM Hyperparameters ---
    iterations = 100          # 100 generations
    batch_size = 50           # 50 agents per generation
    elite_frac = 0.2          # Top 20% survive (Top 10 agents)
    elite_size = int(batch_size * elite_frac)
    
    # Initialize Random Distribution Center
    mu = np.random.normal(0, 0.5, agent.num_params)
    sigma = np.ones(agent.num_params) * 2.0 

    for iteration in range(1, iterations + 1):
        candidate_weights = np.random.normal(loc=mu, scale=sigma, size=(batch_size, agent.num_params))
        rewards = np.zeros(batch_size)
        
        for i in range(batch_size):
            obs, _ = env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            
            weights_to_test = candidate_weights[i]
            
            while not (terminated or truncated):
                action = agent.act(obs, weights=weights_to_test)
                next_obs, env_reward, terminated, truncated, _ = env.step(action)
                reward = agent.reward_function(obs, action, next_obs, terminated, truncated)
                total_reward += reward
                obs = next_obs
                
            rewards[i] = total_reward
            
        elite_indices = np.argsort(rewards)[::-1][:elite_size]
        elite_weights = candidate_weights[elite_indices]
        
        # Update Distribution
        mu = np.mean(elite_weights, axis=0)
        # Keep a minimum standard deviation (0.05) so it never stops exploring minor tweaks
        sigma = np.std(elite_weights, axis=0) + 0.05 
        
        best_reward = rewards[elite_indices[0]]
        avg_reward = np.mean(rewards)
        
        # Print progress clearly
        if iteration % 10 == 0 or iteration == 1:
            print(f"Gen {iteration:3d} | Avg Score: {avg_reward:8.2f} | Best Score: {best_reward:8.2f}")
        
    print("\nTraining Complete! Saving best weights to the Agent.")
    agent.best_weights = mu
    
    print("\n--- FINAL EVALUATION of Deep CEM Agent ---")
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    steps = 0
    
    while not (terminated or truncated):
        action = agent.act(obs) # Uses final best_weights
        next_obs, _, terminated, truncated, _ = env.step(action)
        reward = agent.reward_function(obs, action, next_obs, terminated, truncated)
        total_reward += reward
        obs = next_obs
        steps += 1
        
    print(f"Final Agent Score: {total_reward:.2f} | Steps Survived: {steps}")
    print("\n[SUCCESS] The ML Weights have converged! Check your memory/terminal for the array or use agent.best_weights.")
    
    # Save weights to file so you can easily copy them!
    np.save('best_agent_weights.npy', agent.best_weights)
    print("Saved 131 weights to 'best_agent_weights.npy'!")

if __name__ == '__main__':
    train_cem()
