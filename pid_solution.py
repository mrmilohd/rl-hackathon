import numpy as np

class PIDAgent:
    """
    A pure NumPy implementation of a Proportional-Integral-Derivative (PID) Controller.
    This is the industry standard for continuous control systems (like boilers and valves),
    and requires absolutely zero Neural Networks or external libraries.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
        # --- PID Tuning Parameters (These are the "Weights" you tune) ---
        
        # 1. Pressure Controller (Controls Inlet / Outlet)
        # Kp: Proportional gain (reacts to current error)
        # Ki: Integral gain (reacts to accumulated past errors, helps reach exact target)
        # Kd: Derivative gain (reacts to rate of change, prevents overshooting)
        self.Kp_P = 0.5   
        self.Ki_P = 0.05  
        self.Kd_P = 0.1   
        
        # 2. Temperature Controller (Controls Heater)
        self.Kp_T = 0.4
        self.Ki_T = 0.02
        self.Kd_T = 0.1
        
        # --- Memory for Derivative and Integral calculations ---
        self.integral_P_error = 0.0
        self.prev_P_error = 0.0
        
        self.integral_T_error = 0.0
        self.prev_T_error = 0.0

    def act(self, observation):
        curr_p, curr_t, target_p, target_t = observation
        
        # ==========================================
        # 1. PRESSURE CONTROL (PID Logic)
        # ==========================================
        error_P = target_p - curr_p
        
        # Accumulate error over time (Integral)
        self.integral_P_error += error_P
        
        # Calculate rate of change of error (Derivative)
        derivative_P = error_P - self.prev_P_error
        
        # The core PID Formula
        control_signal_P = (self.Kp_P * error_P) + (self.Ki_P * self.integral_P_error) + (self.Kd_P * derivative_P)
        
        # Save current error for the next step
        self.prev_P_error = error_P
        
        # Map the abstract control signal to physical valves (0 to 1)
        inlet_v = 0.0
        outlet_v = 0.0
        
        if control_signal_P > 0:
            # We need positive pressure -> Open inlet, close outlet
            inlet_v = np.clip(control_signal_P, 0.0, 1.0)
        else:
            # We need negative pressure -> Close inlet, open outlet
            # Multiply by -1 because the outlet valve needs a positive 0-1 percentage
            outlet_v = np.clip(-control_signal_P, 0.0, 1.0)

        # ==========================================
        # 2. TEMPERATURE CONTROL (PID Logic)
        # ==========================================
        error_T = target_t - curr_t
        self.integral_T_error += error_T
        derivative_T = error_T - self.prev_T_error
        
        # Advanced trick: Feed-Forward Control
        # We know the tank naturally cools by ~ 2.0 * (curr_temp / 100). 
        # So we PRE-add a little bit of heater power to fight the cooling!
        natural_cooling_compensation = (2.0 * (curr_t / 100.0)) / 5.0 
        
        # Standard PID component
        control_signal_T = (self.Kp_T * error_T) + (self.Ki_T * self.integral_T_error) + (self.Kd_T * derivative_T)
        
        # Total Heater Power = PID + Feed-Forward Compensation
        heater_p = control_signal_T + natural_cooling_compensation
        heater_p = np.clip(heater_p, 0.0, 1.0)
        
        self.prev_T_error = error_T

        return np.array([inlet_v, outlet_v, heater_p], dtype=np.float32)

    def reward_function(self, state, action, next_state, terminated, truncated):
        """
        Our advanced custom reward logic.
        """
        curr_p, curr_t, target_p, target_t = state
        next_p, next_t, _, _ = next_state
        inlet, outlet, heater = action
        
        p_error = next_p - target_p
        t_error = next_t - target_t
        
        # Distance penalty
        reward = -( (p_error**2) + (t_error**2) ) * 0.1
        
        # Safety Buffer (Keep away from 95!)
        if next_p > 85: reward -= (next_p - 85) * 5 
        if next_t > 85: reward -= (next_t - 85) * 5
            
        # Efficiency Penalty
        reward -= (inlet * outlet) * 10 
        
        # Target Bonus
        if abs(p_error) < 1.0 and abs(t_error) < 1.0:
            reward += 50 
            
        # Catastrophic failure penalty
        if terminated:
            reward -= 200 
            
        return reward
