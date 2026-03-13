import numpy as np
from environment import MysteryControlEnv
from mock_solution import MockParticipantAgent

def evaluate_agent(num_episodes=5):
    env = MysteryControlEnv()
    agent = MockParticipantAgent(env.action_space, env.observation_space)
    
    total_scores = []
    
    print("Starting Evaluation...\n")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated):
            # 1. Agent chooses an action
            action = agent.act(obs)
            
            # 2. Environment takes a step
            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            
            # 3. Calculate custom reward (Hackathon evaluation)
            custom_reward = agent.reward_function(obs, action, next_obs, terminated, truncated)
            episode_reward += custom_reward
            
            obs = next_obs
            step += 1
            
        total_scores.append(episode_reward)
        print(f"Episode {episode + 1}: Score = {episode_reward:.2f} (Steps: {step})")
        
    avg_score = np.mean(total_scores)
    print(f"\nFinal Average Score over {num_episodes} episodes: {avg_score:.2f}")

if __name__ == '__main__':
    evaluate_agent(5)