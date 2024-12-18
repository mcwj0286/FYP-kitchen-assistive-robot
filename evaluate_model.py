#%%
import torch
import numpy as np
from sim_env.LIBERO.BAKU.baku.agent.baku import BCAgent
import gym
import json
import os
import sys
# sys.path.insert(0, '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO')
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.benchmark import get_benchmark
from libero.lifelong.main import get_task_embs
from libero.lifelong.metric import raw_obs_to_tensor_obs
from sentence_transformers import SentenceTransformer
from libero.libero import benchmark, get_libero_path, set_libero_default_path

sentence_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
class Config:
    def __init__(self):
        self.task_embedding_format = "gpt2"
        
        # Data config
        self.data = type('DataConfig', (), {
            'max_word_len': 512,
        })()
        
        # Policy config structure
        policy_config = {
            'language_encoder': type('LanguageEncoder', (), {
                'network_kwargs': type('NetworkKwargs', (), {
                    'input_size': None  # This will be set dynamically
                })()
            })()
        }
        self.policy = type('PolicyConfig', (), policy_config)()

# Create config instance 
cfg = Config()

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
        'augment': False,
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

def evaluate_episode(env, agent, task_emb, max_steps=252):
    # Update evaluate_episode to interact with the environment and use task embeddings
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        # Prepare data for the agent
        data = raw_obs_to_tensor_obs(obs[0], task_emb)
        # Get action from the agent
        action = agent.act(data, eval_mode=True)
        # Step the environment
        obs, reward, done, info = env.step([action])
        steps += 1
    # Return success indicator
    return info[0].get('success', False)

def main():
    # Configuration
    root_dir = '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO/BAKU'
    checkpoint_path = os.path.join(root_dir, 'baku/weights/weights/libero/baku.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets_default_path = get_libero_path("datasets")
    init_states_default_path = get_libero_path("init_states")
    bddl_files_default_path = get_libero_path("bddl_files")
    # set_libero_default_path(datasets_default_path, bddl_files_default_path)

    # Set up the benchmark and get task embeddings
    benchmark_dict = benchmark.get_benchmark_dict()
    print(benchmark_dict)
    benchmark_instance = benchmark_dict["libero_10"]()

    descriptions = [benchmark_instance.get_task(i).language for i in range(benchmark_instance.get_num_tasks())]
    # print(descriptions)
    task_embs = sentence_encoder.encode(descriptions) # [90,384]
    # print(task_embs.shape)
    task_id = 0  # Specify the task ID for evaluation
    task = benchmark_instance.get_task(task_id)
    task_emb = task_embs[task_id]

    init_states_path = os.path.join(init_states_default_path, task.problem_folder, task.init_states_file)
    if not os.path.exists(init_states_path):
        print(f"[error] the init states {init_states_path} cannot be found. Check your paths", "red")
    
    init_states = benchmark_instance.get_task_init_states(task_id)
    # Set up environment arguments
    env_args = {
        'bddl_file_name': os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
        'camera_heights': 128,
        'camera_widths': 128,
    }
    print(env_args)
    env_num = 5
    try:
        # env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
        env = OffScreenRenderEnv(**env_args)
        env.reset()
    except Exception as e:
        print(f"Error initializing environments: {e}")
        return
    
    env.seed(0)  # Set seed for reproducibility
    for eval_index in range(len(init_states)):
        env.set_init_state(init_states[eval_index])

    for _ in range(5):
        obs, _, _, _ = env.step([0.] * 7)
    # Load the agent
    agent = load_agent(checkpoint_path, device)
    # # Evaluation loop
    # num_episodes = 10
    # success_count = 0
    # for episode in range(num_episodes):
    #     success = evaluate_episode(env, agent, task_emb)
    #     if success:
    #         success_count += 1
    #     print(f"Episode {episode + 1}: Success = {success}")
    # # Report evaluation metrics
    # print(f"Success rate: {success_count / num_episodes}")

if __name__ == '__main__':
    main()
