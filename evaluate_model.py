#%%
import torch
import numpy as np
from sim_env.LIBERO.BAKU.baku.agent.baku import BCAgent
import gym
import json
import os
import sys
# sys.path.insert(0, '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO')
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark
from libero.lifelong.main import get_task_embs
from libero.lifelong.metric import raw_obs_to_tensor_obs
from sentence_transformers import SentenceTransformer
from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.utils.video_utils import VideoWriter
from libero.libero.utils.time_utils import Timer
import robomimic.utils.obs_utils as ObsUtils

sentence_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
class Config:
    def __init__(self):
        # self.task_embedding_format = "gpt2"
        
        # Data config
        data_config = {
            'max_word_len': 512,
            'obs': type('ObsConfig', (), {
                'modality': {
                    'rgb': ['agentview_rgb', 'eye_in_hand_rgb'],
                    'low_dim': ['gripper_states', 'joint_states']
                }
            })(),
            'obs_key_mapping': {
                'agentview_rgb': 'agentview_image',
                'eye_in_hand_rgb': 'robot0_eye_in_hand_image',
                'gripper_states': 'robot0_gripper_qpos',
                'joint_states': 'robot0_joint_pos'
            },
            'img_h': 128,
            'img_w': 128
        }
        self.data = type('DataConfig', (), data_config)()
        self.device = 'cuda'

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

def evaluate_episode(env, agent, task_emb, max_steps=252):
    # Update evaluate_episode to interact with the environment and use task embeddings
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        # Prepare data for the agent
        data = raw_obs_to_tensor_obs(obs[0], task_emb)
        # Get action from the agent with required arguments
        action = agent.act(
            obs=data["obs"],
            prompt={"task_emb": task_emb},
            norm_stats=None,  # Set to None if normalization not needed
            step=steps,
            global_step=steps,
            eval_mode=True
        )
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

    # Get task embeddings
    descriptions = [benchmark_instance.get_task(i).language for i in range(benchmark_instance.get_num_tasks())]
    task_embs = sentence_encoder.encode(descriptions) # [90,384]
    task_id = 0  # Specify the task ID for evaluation
    task = benchmark_instance.get_task(task_id)
    

    init_states_path = os.path.join(init_states_default_path, task.problem_folder, task.init_states_file)
    if not os.path.exists(init_states_path):
        print(f"[error] the init states {init_states_path} cannot be found. Check your paths", "red")
    
    init_states = benchmark_instance.get_task_init_states(task_id)
    
    # # Configure evaluation parameters
    # env_num = 20
    max_steps = 252
    save_videos = True
    video_folder = "evaluation_videos"

    # Add this before using raw_obs_to_tensor_obs
    # Initialize observation utils with the modality specs from config
    obs_modality_specs = {
        "obs": {
            "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
            "low_dim": ["gripper_states", "joint_states"]
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
    
    with Timer() as t, VideoWriter(video_folder, save_videos) as video_writer:
        # Set up environment arguments
        env_args = {
            'bddl_file_name': os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
            'camera_heights': 128,
            'camera_widths': 128,
        }
        
        # Initialize single environment
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        env.seed(0)
        
        init_states_path = os.path.join(
            init_states_default_path, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        init_state = init_states[0]  # Take first init state

        done = False
        steps = 0
        obs = env.set_init_state(init_state)
        task_emb = task_embs[task_id]

        # Initialize agent
        agent = load_agent(checkpoint_path, device)

        success = False
        # Initial physics stabilization
        for _ in range(5):
            env.step(np.zeros(7))

        with torch.no_grad():
            while steps < max_steps and not done:
                steps += 1
                
                # Prepare data and get action with all required arguments
                task_emb_tensor = torch.from_numpy(task_emb).unsqueeze(0).to(device)
                data = raw_obs_to_tensor_obs([obs], task_emb_tensor, cfg)
                action = agent.act(
                    obs=data["obs"],
                    prompt={"task_emb": task_emb_tensor[0]},
                    norm_stats=None,  # Set to None if normalization not needed
                    step=steps,
                    global_step=steps,
                    eval_mode=True
                )
                print(action.shape)
                # Step environment
                obs, reward, done, info = env.step(action)
                video_writer.append_obs(obs, done, camera_name="agentview_image")

                success = info.get('success', False)

        env.close()

        # Save evaluation stats
        eval_stats = {
            "success": success,
            "total_steps": steps,
            "time_taken": t.get_elapsed_time()
        }
        
        save_path = f"eval_stats_task{task_id}.json"
        with open(save_path, 'w') as f:
            json.dump(eval_stats, f)

        print(f"Evaluation completed in {t.get_elapsed_time()} seconds")
        print(f"Success: {success}")
        print(f"Results saved to {save_path}")

if __name__ == '__main__':
    main()
