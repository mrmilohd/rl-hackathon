import numpy as np
from environment import MysteryControlEnv
from pid_solution import PIDAgent

def evaluate_pid(num_episodes=5):
    env = MysteryControlEnv()
    agent = PIDAgent(env.action_space, env.observation_space)
    
    total_scores = []
    
    print("Starting Advanced PID Evaluation...\n")
    
    for episode in range(num_episodes):
        # We MUST reset the agent's PID memory when the environment resets!
        agent.integral_P_error = 0.0
        agent.prev_P_error = 0.0
        agent.integral_T_error = 0.0
        agent.prev_T_error = 0.0
        
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated):
            action = agent.act(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # Use our custom reward shape for scoring
            custom_reward = agent.reward_function(obs, action, next_obs, terminated, truncated)
            episode_reward += custom_reward
            
            obs = next_obs
            step += 1
            
        total_scores.append(episode_reward)
        print(f"Episode {episode + 1}: Score = {episode_reward:.2f} (Steps: {step})")
        
    avg_score = np.mean(total_scores)
    print(f"\nFinal PID Average Score over {num_episodes} episodes: {avg_score:.2f}")

if __name__ == '__main__':
    evaluate_pid(10)