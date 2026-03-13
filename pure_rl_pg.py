import numpy as np

# We import the environment directly (assuming it's named MysteryControlEnv inside environment.py)
from environment import MysteryControlEnv

class NumpyPolicyGradientAgent:
    """
    A 'Proper Deep RL' Agent built entirely from scratch in NumPy using the 
    REINFORCE (Monte Carlo Policy Gradient) algorithm.
    It treats the environment strictly as a Black Box. No physics equations are used.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
        # 1. Initialize Neural Network Weights
        # State space is 4, Action space is 3. We use a simple linear layer.
        self.theta = np.zeros((4, 3)) 
        
        # 2. Exploration rate (standard deviation for our Gaussian action distribution)
        self.std = 0.20 

    def sigmoid(self, x):
        # Activation function to bound actions between 0 and 1
        x = np.clip(x, -50, 50) # prevent overflow
        return 1.0 / (1.0 + np.exp(-x))

    def act(self, observation, explore=True):
        """Forward Pass of our RL model"""
        # 1. Make observation numbers smaller so weights don't explode (Normalization)
        obs_scaled = observation / 100.0
        
        # 2. Calculate Mean Action (Forward pass through Neural Net layer)
        # mu = Sigmoid(Observation * Weights)
        mu = self.sigmoid(np.dot(obs_scaled, self.theta))
        
        # 3. Sample from Gaussian distribution
        if explore:
            action = np.random.normal(mu, self.std)
        else:
            action = mu # During evaluation, we just pick the best predicted action
            
        # Clip the action to physical bounds [0, 1]
        action_clipped = np.clip(action, 0.0, 1.0)
        return action_clipped, mu

    def reward_function(self, state, action, next_state, terminated, truncated):
        """Our RL Reward shaping (Black Box approach)"""
        curr_p, curr_t, target_p, target_t = state
        next_p, next_t = next_state[0], next_state[1]
        
        p_error = abs(next_p - target_p)
        t_error = abs(next_t - target_t)
        
        # Negative reward based on distance to target
        reward = -(p_error + t_error) * 0.1
        
        # Massive penalty for terminating (blowing up)
        if terminated:
            reward -= 100.0
            
        # Stability bonus
        if p_error < 2.0 and t_error < 2.0:
            reward += 10.0
            
        return reward


def train_policy_gradient():
    """
    The main RL Training Loop. It uses backpropagation calculated by hand in NumPy.
    """
    env = MysteryControlEnv()
    agent = NumpyPolicyGradientAgent(env.action_space, env.observation_space)
    
    learning_rate = 0.01
    gamma = 0.99  # Discount factor for future rewards
    num_episodes = 500
    
    print("Starting purely Black-Box Reinforcement Learning (REINFORCE)...\n")
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        
        # Memory variables for the RL trajectory
        states_mem = []
        actions_mem = []
        mus_mem = []
        rewards_mem = []
        
        terminated = False
        truncated = False
        
        # 1. Play the episode
        while not (terminated or truncated):
            action, mu = agent.act(obs, explore=True)
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # Use our custom reward shape
            reward = agent.reward_function(obs, action, next_obs, terminated, truncated)
            
            # Store memory
            states_mem.append(obs / 100.0) # Remember to save scaled obs
            actions_mem.append(action)
            mus_mem.append(mu)
            rewards_mem.append(reward)
            
            obs = next_obs
            
        # 2. End of episode: Calculate Discounted Returns (G_t)
        returns = []
        G = 0
        for r in reversed(rewards_mem):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        
        # Normalize returns (Stabilizes deep learning training)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # 3. BACKPROPAGATION (Manual Gradient Ascent)
        # We manually calculate the derivative of the Log Probability of a Gaussian Distribution
        for t in range(len(states_mem)):
            s_t = states_mem[t] # Shape: (4,)
            a_t = actions_mem[t] # Shape: (3,)
            mu_t = mus_mem[t]    # Shape: (3,)
            G_t = returns[t]     # Scalar
            
            # Derivative of Gaussian Log-Prob: (action - mean) / var
            d_log_prob = (a_t - mu_t) / (agent.std ** 2)
            
            # Derivative of Sigmoid: mu * (1 - mu)
            d_sigmoid = mu_t * (1.0 - mu_t)
            
            # Chain Rule to get Gradient for Weights (Theta)
            gradient = np.outer(s_t, d_log_prob * d_sigmoid)
            
            # Update Weights
            agent.theta += learning_rate * G_t * gradient
            
        # Logging progress
        if episode % 20 == 0:
            total_score = sum(rewards_mem)
            print(f"Episode: {episode:3d} | Total Score: {total_score:7.2f} | Episode Length: {len(states_mem)}")

if __name__ == '__main__':
    train_policy_gradient()
