import numpy as np

class MBRLParticipantAgent:
    """
    Model-Based Reinforcement Learning (MBRL) Agent using Model Predictive Control (MPC).
    This agent learns the physics of the environment on the fly and simulates thousands
    of possible futures "in its head" before picking the best action.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
        # 1. Experience Buffer (to train our dynamics model)
        self.state_history = []
        self.action_history = []
        self.next_state_history = []
        
        # 2. Planning Parameters (Model Predictive Control)
        self.planning_horizon = 5     # How many steps into the future to simulate
        self.num_simulations = 1000   # How many random futures to imagine per step
        self.warmup_steps = 15        # Take random actions at first to collect data
        self.steps_taken = 0
        
        # 3. The "Neural Network" / Dynamics Model Parameters
        # For this simple environment, we will use a fast Linear Regression model
        # next_state = (state * W_s) + (action * W_a) + bias
        # In a highly complex environment, this would be a PyTorch MLP.
        self.W = None 

    def remember(self, state, action, next_state):
        """Store the transition data to train our world model."""
        # We only care about predicting Pressure (idx 0) and Temp (idx 1), 
        # because Targets (idx 2, 3) never change!
        self.state_history.append(state)
        self.action_history.append(action)
        self.next_state_history.append(next_state[:2]) 
        
    def train_dynamics_model(self):
        """
        Trains the probabilistic world model p(s_{t+1} | s_t, a_t).
        We use ordinary least squares (Linear Regression) for lightning-fast training.
        If using Deep Learning, this is where you would do loss.backward() and optimizer.step().
        """
        if len(self.state_history) < self.warmup_steps:
            return # Not enough data yet
            
        # Inputs: [Current Pressure, Current Temp, Target P, Target T, Inlet, Outlet, Heater]
        X = np.hstack([self.state_history, self.action_history])
        
        # Outputs: [Next Pressure, Next Temp]
        Y = np.array(self.next_state_history)
        
        # Fit the model: W = (X^T * X)^-1 * X^T * Y
        # This solves for the exact physics coefficients of the hidden environment!
        self.W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    def predict_next_state(self, states, actions):
        """Uses the learned model to predict the future state in our imagination."""
        X = np.hstack([states, actions])
        predicted_next_pt = X @ self.W # Matrix multiplication
        
        # Reconstruct the full state [Pressure, Temp, Target P, Target T]
        # Targets remain the same
        full_next_states = np.zeros_like(states)
        full_next_states[:, 0:2] = predicted_next_pt
        full_next_states[:, 2:4] = states[:, 2:4] # Copy targets
        
        return full_next_states

    def act(self, observation):
        self.steps_taken += 1
        
        # Phase 1: Exploration (Gather data for the model)
        if self.steps_taken < self.warmup_steps or self.W is None:
            return self.action_space.sample()
            
        # Phase 2: Model Predictive Control (MPC) - Random Shooting Method
        # We imagine 1,000 parallel universes, each with a random sequence of 5 actions.
        
        # Create (1000, 5, 3) shaped array of random actions
        imagined_action_sequences = np.random.uniform(
            low=0.0, high=1.0, 
            size=(self.num_simulations, self.planning_horizon, self.action_space.shape[0])
        )
        
        # Keep track of the total reward for each of the 1000 parallel universes
        trajectory_rewards = np.zeros(self.num_simulations)
        
        # All universes start at the exact current real state
        current_imagined_states = np.tile(observation, (self.num_simulations, 1))
        
        # Simulate 'H' steps into the future
        for t in range(self.planning_horizon):
            actions_at_t = imagined_action_sequences[:, t, :]
            
            # Predict what the environment will do using our trained model
            predicted_next_states = self.predict_next_state(current_imagined_states, actions_at_t)
            
            # Score each predicted future
            rewards_at_t = self.batch_reward_function(current_imagined_states, actions_at_t, predicted_next_states)
            trajectory_rewards += rewards_at_t
            
            current_imagined_states = predicted_next_states
            
        # Find the single universe that resulted in the highest total reward
        best_universe_idx = np.argmax(trajectory_rewards)
        
        # Return the FIRST action of that best sequence. 
        # (We will re-plan again on the very next timestep - this is receding horizon control!)
        best_action = imagined_action_sequences[best_universe_idx, 0, :]
        return best_action

    def batch_reward_function(self, states, actions, next_states):
        """
        Calculates rewards for all 1,000 simulations at the same time using Numpy array operations.
        Matches the logic of our reward shaping!
        """
        target_p = states[:, 2]
        target_t = states[:, 3]
        next_p = next_states[:, 0]
        next_t = next_states[:, 1]
        
        p_error = next_p - target_p
        t_error = next_t - target_t
        
        # Quadratic penalty calculation
        rewards = -(p_error**2 + t_error**2) * 0.1
        
        # Safety penalties
        rewards -= np.maximum(0, next_p - 85) * 5
        rewards -= np.maximum(0, next_t - 85) * 5
        
        # Massive penalty if the simulated model thinks it will explode!
        exploded = (next_p > 95) | (next_t > 95)
        rewards[exploded] -= 200
        
        return rewards
