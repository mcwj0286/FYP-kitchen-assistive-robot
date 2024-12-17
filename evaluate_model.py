#%%
import torch
import numpy as np
from sim_env.LIBERO.BAKU.baku.agent.baku import BCAgent
import gym
import json
import os

def load_agent(checkpoint_path, device='cuda'):
    """Load a trained BAKU agent from checkpoint."""
    # Load checkpoint
    payload = torch.load(checkpoint_path, map_location=device)
    
    # Define agent configuration (match your training config)
    agent_config = {
        'obs_shape': {
            'pixels': (3, 128, 128),
            'pixels_egocentric': (3, 128, 128),
            'proprioceptive': (9,),
            'features': (51,)
        },
        'action_shape': (7,),
        'device': device,
        'lr': 1e-4,
        'hidden_dim': 256,  # from suite.hidden_dim
        'stddev_schedule': 0.1,
        'stddev_clip': 0.3,
        'use_tb': True,  # from root config
        'augment': True,
        'obs_type': 'pixels',
        'encoder_type': 'resnet',
        'policy_type': 'gpt',
        'policy_head': 'deterministic',
        'pixel_keys': ['pixels', 'pixels_egocentric'],  # from suite config
        'proprio_key': 'proprioceptive',
        'feature_key': 'features',
        'use_proprio': True,
        'train_encoder': True,
        'norm': False,
        'history': False,  # from suite config
        'history_len': 10,  # from suite config
        'eval_history_len': 5,  # from suite config
        'separate_encoders': False,
        'temporal_agg': True,
        'max_episode_len': 252,  # from suite.task_make_fn # adjust based on the task
        'num_queries': 10,
        'use_language': True,
        'prompt': 'text',  # from root config
        'film': True
    }
    
    # Create agent
    agent = BCAgent(**agent_config)
    
    # Load state from checkpoint
    agent.load_snapshot(payload, eval=True)
    return agent

def evaluate_episode(env, agent, episode_length=252):  # Updated episode length
    """Run one evaluation episode."""
    obs = env.reset()
    episode_reward = 0
    for step in range(episode_length):
        # Format observation for agent
        # Adjust this based on your environment's observation space
        agent_obs = {
            'pixels': obs['pixels'] if isinstance(obs, dict) else obs,
            'proprio': np.zeros(7)  # Adjust proprio dimension as needed
        }
        
        # Create dummy prompt (adjust based on your setup)
        prompt = {
            'task_emb': np.zeros(384)  # Adjust embedding dimension as needed
        }
        
        # Get action from agent
        action = agent.act(
            obs=agent_obs,
            prompt=prompt,
            norm_stats=None,
            step=step,
            global_step=0,
            eval_mode=True
        )
        
        # Step environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    return episode_reward

def main():
    # Configuration
    root_dir = '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO/BAKU'
    checkpoint_path = os.path.join(root_dir, 'baku/weights/weights/libero/baku.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # num_episodes = 10
    
    # # Create environment
    # env = gym.make('your-env-id')  # Replace with your environment
    
    # Load agent
    agent = load_agent(checkpoint_path, device)
    print(agent)
    # # Evaluate
    # episode_rewards = []
    # for episode in range(num_episodes):
    #     reward = evaluate_episode(env, agent)
    #     episode_rewards.append(reward)
    #     print(f"Episode {episode+1}: Reward = {reward:.2f}")
    
    # # Print summary
    # print("\nEvaluation Summary:")
    # print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

if __name__ == '__main__':
    main()
