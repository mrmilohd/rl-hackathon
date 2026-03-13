from agent_template import ParticipantAgent
import numpy as np

class MySmartAgent(ParticipantAgent):
    """
    An example of a participant's submission.
    """
    def act(self, observation):
        pressure, temp, target_pressure, target_temp = observation
        
        # Simple heuristic-based control (just for demonstration)
        # In a real submission, this would be a trained RL model.
        inlet_v = 1.0 if pressure < target_pressure else 0.0
        outlet_v = 1.0 if pressure > target_pressure else 0.0
        heater_p = 1.0 if temp < target_temp else 0.0
        
        return np.array([inlet_v, outlet_v, heater_p], dtype=np.float32)

    def reward_function(self, state, action, next_state, terminated, truncated):
        # Improved reward logic
        pressure, temp, target_pressure, target_temp = state
        next_pressure, next_temp, _, _ = next_state
        
        p_error = abs(next_pressure - target_pressure)
        t_error = abs(next_temp - target_temp)
        
        # Reward for stability and staying close to targets
        reward = - (p_error * 0.5 + t_error * 0.5)
        
        # Heavy penalty for safety violations
        if terminated:
            reward -= 500
            
        return reward
