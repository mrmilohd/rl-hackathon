import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MysteryControlEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(MysteryControlEnv, self).__init__()
        
        # Action space: [Inlet Valve Opening (0-1), Outlet Valve Opening (0-1), Heater Power (0-1)]
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Observation space: [Current Pressure, Current Temperature, Target Pressure, Target Temperature]
        # Pressure: 0 to 100 psi, Temperature: 0 to 100 C
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.state = None
        self.steps_beyond_done = None
        self.max_steps = 200
        self.current_step = 0
        
        # System parameters (mystery parameters to be tuned or learned)
        self.inlet_flow_rate = 10.0
        self.outlet_flow_rate = 8.0
        self.heat_coefficient = 5.0
        self.cooling_coefficient = 2.0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random initial state and targets
        pressure = self.np_random.uniform(20, 40)
        temp = self.np_random.uniform(20, 30)
        target_pressure = self.np_random.uniform(50, 70)
        target_temp = self.np_random.uniform(60, 80)
        
        self.state = np.array([pressure, temp, target_pressure, target_temp], dtype=np.float32)
        self.current_step = 0
        
        return self.state, {}

    def step(self, action):
        inlet_v, outlet_v, heater_p = action
        pressure, temp, target_pressure, target_temp = self.state
        
        # Update pressure (simplified dynamics)
        pressure_change = (inlet_v * self.inlet_flow_rate) - (outlet_v * self.outlet_flow_rate)
        new_pressure = np.clip(pressure + pressure_change, 0, 100)
        
        # Update temperature (simplified dynamics)
        temp_change = (heater_p * self.heat_coefficient) - (self.cooling_coefficient * (temp / 100))
        new_temp = np.clip(temp + temp_change, 0, 100)
        
        # Check for safety violations (pressure or temp too high)
        terminated = False
        if new_pressure > 95 or new_temp > 95:
            terminated = True
        
        # Default reward logic (participants will modify this)
        # Reward is higher when pressure and temp are close to targets
        pressure_error = abs(new_pressure - target_pressure)
        temp_error = abs(new_temp - target_temp)
        
        # Basic reward: Negative error
        reward = -(pressure_error + temp_error)
        
        # Penalty for termination (failure)
        if terminated:
            reward -= 100
        
        self.state = np.array([new_pressure, new_temp, target_pressure, target_temp], dtype=np.float32)
        self.current_step += 1
        
        truncated = self.current_step >= self.max_steps
        
        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, State: {self.state}")

    def close(self):
        pass

if __name__ == "__main__":
    # Test the environment
    env = MysteryControlEnv()
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}")
        if terminated or truncated:
            break
    env.close()
