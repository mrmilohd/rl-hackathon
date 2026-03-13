import numpy as np
from environment import MysteryControlEnv
from cem_rl import CEMRLAgent # Testing our trained agent

def evaluate_agent(num_episodes=5):
    env = MysteryControlEnv()
    agent = CEMRLAgent(env.action_space, env.observation_space)
    # Load our fully trained weights!
    agent.best_weights = np.load("best_agent_weights.npy")
    
    total_scores = []
    total_steps = []
    
    print("Starting Objective Evaluation...\n")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step = 0
        
        episode_score = 0.0
        
        while not (terminated or truncated):
            # 1. Agent chooses an action
            action = agent.act(obs)
            
            # 2. Environment takes a step
            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            
            # 3. Objective Error Calculation:
            # obs = [Pressure, Temp, Target_Pressure, Target_Temp]
            press_err = abs(next_obs[0] - next_obs[2])
            temp_err = abs(next_obs[1] - next_obs[3])
            
            # Standard evaluation penalty
            step_score = -(press_err + temp_err)
            if terminated:
                step_score -= 100
                
            episode_score += step_score
            
            obs = next_obs
            step += 1
            
        total_steps.append(step)
        total_scores.append(episode_score)
        
        print(f"Episode {episode + 1}: Survived = {step}/200 | Score: {episode_score:.2f}")
        
    print(f"\n--- Final Objective Averages over {num_episodes} episodes ---")
    print(f"Avg Survival: {np.mean(total_steps):.1f} / 200 steps")
    print(f"Avg Score:    {np.mean(total_scores):.2f}")

if __name__ == '__main__':
    evaluate_agent(5)