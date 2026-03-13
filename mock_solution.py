import numpy as np

class MockParticipantAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
        # We can store past errors here to do PID-like control if we want to be advanced!
        self.prev_p_error = 0
        self.prev_t_error = 0

    def act(self, observation):
        """
        MOCK RULE-BASED BRAIN
        Instead of random guessing or a Neural Network, we use our understanding of the physics.
        Observation: [current_pressure, current_temp, target_pressure, target_temp]
        """
        curr_p, curr_t, target_p, target_t = observation
        
        # 1. Control Pressure (Inlet adds pressure, Outlet removes pressure)
        # Max inlet adds 10, max outlet removes 8.
        p_diff = target_p - curr_p
        
        inlet_v = 0.0
        outlet_v = 0.0
        
        if p_diff > 0:
            # We need more pressure. 
            # If the difference is big, open inlet fully. Otherwise, open proportionally.
            inlet_v = np.clip(p_diff / 10.0, 0.0, 1.0) 
        elif p_diff < 0:
            # We have too much pressure.
            # Open outlet proportionally. Note: outlet removes 8 max per step.
            outlet_v = np.clip(abs(p_diff) / 8.0, 0.0, 1.0)
            
        # 2. Control Temperature (Heater adds temp, system naturally cools)
        # Heater adds up to 5. Natural cooling removes up to 2.
        t_diff = target_t - curr_t
        
        # We need to overcome natural cooling AND heat it up to the target
        natural_cooling = 2.0 * (curr_t / 100.0)
        required_heat = t_diff + natural_cooling
        
        if required_heat > 0:
            # Heater can provide max 5 units of heat
            heater_p = np.clip(required_heat / 5.0, 0.0, 1.0)
        else:
            heater_p = 0.0 # Just let it cool naturally
            
        return np.array([inlet_v, outlet_v, heater_p], dtype=np.float32)

    def reward_function(self, state, action, next_state, terminated, truncated):
        """
        IMPROVED REWARD FUNCTION (Reward Shaping)
        This is how you teach a Neural Network if you were training one.
        """
        curr_p, curr_t, target_p, target_t = state
        next_p, next_t, _, _ = next_state
        inlet, outlet, heater = action
        
        # 1. Error Penalties (Quadratic is better than Linear for fine-tuning)
        p_error = next_p - target_p
        t_error = next_t - target_t
        
        # Squaring the error punishes Large mistakes heavily, but is forgiving to tiny errors
        distance_penalty = -( (p_error**2) + (t_error**2) ) * 0.1
        
        # 2. Safety Bounds Penalty (Keep away from 95+ danger zone!)
        safety_penalty = 0
        if next_p > 85:
            safety_penalty -= (next_p - 85) * 5  # Sharp penalty as it nears 95
        if next_t > 85:
            safety_penalty -= (next_t - 85) * 5
            
        # 3. Efficiency Penalty (Don't waste energy/resources)
        # Punish doing inlet and outlet at the same time (wasteful)
        waste_penalty = -(inlet * outlet) * 10 
        
        # 4. Target Bonus (The carrot)
        target_bonus = 0
        if abs(p_error) < 1.0 and abs(t_error) < 1.0:
            target_bonus = 50 # massive reward for stabilizing at the target!
            
        # 5. Terminal condition
        terminal_penalty = 0
        if terminated:
            terminal_penalty = -200 # Exploded/Overheated!
            
        total_reward = distance_penalty + safety_penalty + waste_penalty + target_bonus + terminal_penalty
        return total_reward
