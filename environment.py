import numpy as np
import gymnasium as gym
from gymnasium import spaces

MAX_STEPS = 200
MIN_STATE_VALUE = 0.0
MAX_STATE_VALUE = 100.0
SAFETY_LIMIT = 95.0

class MysteryControlEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=MIN_STATE_VALUE,
            high=MAX_STATE_VALUE,
            shape=(4,),
            dtype=np.float32,
        )
        self.render_mode = render_mode
        self.state = None
        self.max_steps = MAX_STEPS
        self.current_step = 0

        self.inlet_flow_rate = 10.0
        self.outlet_flow_rate = 8.0
        self.heat_coefficient = 5.0
        self.cooling_coefficient = 2.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del options

        pressure = self.np_random.uniform(20, 40)
        temp = self.np_random.uniform(20, 30)
        target_pressure = self.np_random.uniform(50, 70)
        target_temp = self.np_random.uniform(60, 80)

        self.state = np.array([pressure, temp, target_pressure, target_temp], dtype=np.float32)
        self.current_step = 0

        return self.state, {}

    def step(self, action):
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        inlet_v, outlet_v, heater_p = clipped_action
        pressure, temp, target_pressure, target_temp = self.state

        pressure_change = (inlet_v * self.inlet_flow_rate) - (outlet_v * self.outlet_flow_rate)
        new_pressure = np.clip(pressure + pressure_change, MIN_STATE_VALUE, MAX_STATE_VALUE)

        temp_change = (heater_p * self.heat_coefficient) - (self.cooling_coefficient * (temp / 100))
        new_temp = np.clip(temp + temp_change, MIN_STATE_VALUE, MAX_STATE_VALUE)

        terminated = bool(new_pressure > SAFETY_LIMIT or new_temp > SAFETY_LIMIT)

        pressure_error = abs(new_pressure - target_pressure)
        temp_error = abs(new_temp - target_temp)
        reward = -(pressure_error + temp_error)

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
