from environment import MysteryControlEnv
from mbrl_solution import MBRLParticipantAgent
import numpy as np

def run_mbrl():
    env = MysteryControlEnv()
    agent = MBRLParticipantAgent(env.action_space, env.observation_space)
    
    total_rewards = []
    
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            action = agent.act(obs)
            next_obs, true_reward, terminated, truncated, _ = env.step(action)
            
            # The agent stores this experience to learn the physics model
            agent.remember(obs, action, next_obs)
            
            # Continuously train the "Neural Network / Differential Equations"
            if step > agent.warmup_steps:
                agent.train_dynamics_model()
                
            obs = next_obs
            episode_reward += true_reward
            step += 1
            done = terminated or truncated
            
        print(f"Episode {episode + 1}: Score = {episode_reward:.2f} (Steps: {step})")
        total_rewards.append(episode_reward)
        
    print(f"\nFinal MBRL Average Score: {np.mean(total_rewards):.2f}")

if __name__ == '__main__':
    run_mbrl()