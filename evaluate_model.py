#%%
import torch
import numpy as np
from sim_env.LIBERO.BAKU.baku.agent.baku import BCAgent
import gym
import json
import os
import sys
# sys.path.insert(0, '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO')
from libero.libero.envs import OffScreenRenderEnv , DummyVectorEnv
from libero.libero.benchmark import get_benchmark
from libero.lifelong.main import get_task_embs
from utils import raw_obs_to_tensor_obs, encode_task
from sentence_transformers import SentenceTransformer
from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.utils.video_utils import VideoWriter
from libero.libero.utils.time_utils import Timer
from sim_env.LIBERO.BAKU.baku.suite.libero import RGBArrayAsObservationWrapper
import robomimic.utils.obs_utils as ObsUtils
from libero.lifelong.utils import (
    # control_seed,
    # safe_device,
    torch_load_model,
    # NpEncoder,
    # compute_flops,
)
#   evaluation_videos/
#    └── {model_type}/
#        └── {weight_name}/
#            └── {benchmark_name}/
#                └── {task_name}/
#                    └── attempt_{N}.mp4
# sentence_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
from models.bc_baku_policy import BCBakuPolicy

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
        self.max_state_dim = 123
        self.max_episode_len = 600
        self.model_type = 'bc_baku'

# Create config instance 
cfg = Config()

def load_agent(checkpoint_path, model_type='baku', device='cuda'):
    """Load a trained agent from checkpoint."""
    # Load checkpoint
    payload = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'baku':
        # Define agent configuration (match your training config)
        agent_config = {
            'obs_shape': {
                'pixels': (3, 128, 128),
                'pixels_egocentric': (3, 128, 128),
                'proprioceptive': (9,),
                'features': (123,)
            },
            'action_shape': (7,),
            'device': device,
            'lr': 1e-4,
            'hidden_dim': 256,
            'stddev_schedule': 0.1,
            'stddev_clip': 0.3,
            'use_tb': True,
            'augment': True,
            'obs_type': 'pixels',
            'encoder_type': 'resnet',
            'policy_type': 'gpt',
            'policy_head': 'deterministic',
            'pixel_keys': ['pixels', 'pixels_egocentric'],
            'proprio_key': 'proprioceptive',
            'feature_key': 'features',
            'use_proprio': True,
            'train_encoder': True,
            'norm': False,
            'history': False,
            'history_len': 10,
            'eval_history_len': 5,
            'separate_encoders': False,
            'temporal_agg': True,
            'max_episode_len': 600,
            'num_queries': 10,
            'use_language': True,
            'prompt': 'text',
            'film': True
        }
        agent = BCAgent(**agent_config)
        agent.load_snapshot(payload, eval=True)
    
    elif model_type == 'bc_baku':
        agent_config = {
            'repr_dim': 512,
            'act_dim': 7,
            'hidden_dim': 256,
            'policy_head': 'deterministic',
            'obs_type': 'pixels',
            'obs_shape': {
                'pixels': (3, 128, 128),
                'pixels_egocentric': (3, 128, 128),
                'proprioceptive': (9,),
            },
            'device': device,
            'history': True,
            'history_len': 10,
            'temporal_agg': True,
            'max_episode_len': 650,
            'use_proprio': True,
            'num_queries': 10
        }
        agent = BCBakuPolicy(**agent_config).to(device)
        agent.load_state_dict(payload['model_state_dict'])
        agent.eval()
    
    else:  # transformer
        agent = torch_load_model(checkpoint_path, device=device)
        agent.eval()
    
    return agent

def get_model_weights(model_type):
    """Get all model weights from the appropriate directory."""
    if model_type == 'baku':
        root_dir = '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO/BAKU'
        return [os.path.join(root_dir, 'baku/weights/weights/libero/baku.pt')]
    elif model_type == 'transformer':
        checkpoint_path = '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/outputs/2024-12-21/19-27-15/experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_001/multitask_model_ep15.pth'
        return [checkpoint_path]
    elif model_type == 'bc_baku':
        weights_dir = '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/outputs/2025-01-13/01-11-20/experiments/baku_20250113-011120'
        return sorted([os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.pth')])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_evals', type=int, default=5, help='Number of evaluation runs')
    parser.add_argument('--model_type', type=str, default='bc_baku', choices=['baku', 'transformer', 'bc_baku'], help='Type of model to evaluate')
    args = parser.parse_args()

    cfg.model_type = args.model_type
    # Get all model weights for the specified type
    model_weights = get_model_weights(args.model_type)
    print(f"Found {len(model_weights)} weights for model type {args.model_type}")

    datasets_default_path = get_libero_path("datasets")
    init_states_default_path = get_libero_path("init_states")
    bddl_files_default_path = get_libero_path("bddl_files")

    # Set up the benchmark and get task embeddings
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict["libero_spatial"]()
    descriptions = [benchmark_instance.get_task(i).language for i in range(benchmark_instance.get_num_tasks())]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Evaluate each weight file
    for weight_path in model_weights:
        weight_name = os.path.basename(weight_path)
        print(f"\nEvaluating weight file: {weight_name}")

        # Load agent for this weight file
        agent = load_agent(weight_path, model_type=args.model_type, device=device)
        
        # Create results directory for this weight
        results_dir = os.path.join('evaluation_results', args.model_type, weight_name.replace('.pth', ''))
        os.makedirs(results_dir, exist_ok=True)
        
        # Create video directory for this weight
        video_folder = os.path.join('evaluation_videos', args.model_type, weight_name.replace('.pth', ''))
        
        # Initialize overall results dictionary
        all_tasks_results = {}
        
        # Iterate through all tasks
        num_tasks = benchmark_instance.get_num_tasks()
        for task_id in range(num_tasks):
            task = benchmark_instance.get_task(task_id)
            task_emb = encode_task(task.language)

            # Get task name for folder organization
            Benchmark_name = task.problem_folder
            task_name = descriptions[task_id]
            base_video_folder = os.path.join(video_folder, Benchmark_name, task_name)
            os.makedirs(base_video_folder, exist_ok=True)
            
            print(f"\nEvaluating Task {task_id}/{num_tasks-1}: {task_name}")
            
            # Get fixed init states to control the experiment randomness
            init_states_path = os.path.join(init_states_default_path, task.problem_folder, task.init_states_file)
            if not os.path.exists(init_states_path):
                print(f"[error] the init states {init_states_path} cannot be found. Check your paths", "red")
                continue
            init_states = torch.load(init_states_path)
            agent.reset()
            successes = []
            all_eval_stats = []
            
            for eval_idx in range(args.num_evals):
                print(f"Starting evaluation {eval_idx + 1}/{args.num_evals}")
                video_path = base_video_folder
                
                with Timer() as t, VideoWriter(video_path, True, video_filename=f"attempt_{eval_idx}.mp4") as video_writer:
                    # Set up environment arguments
                    env_args = {
                        'bddl_file_name': os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
                        'camera_heights': 128,
                        'camera_widths': 128,
                    }
                    
                    # Initialize environment using DummyVectorEnv
                    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
                    env.reset()
                    env.seed(eval_idx)
                    
                    # Get init state for this evaluation
                    init_state = init_states[eval_idx % len(init_states)]
                    obs = env.set_init_state([init_state])  # No need to index [0] here as raw_obs_to_tensor_obs handles it
                    
                    # Reset model's internal state if needed
                    agent.reset()
                    
                    success = False
                    steps = 0
                    max_steps = 550
                    
                    # Initial physics stabilization
                    dummy_action = np.zeros(7)
                    for _ in range(5):
                        obs, _, _, _ = env.step([dummy_action])
                    
                    with torch.no_grad():
                        while steps < max_steps:
                            steps += 1
                            
                            # Convert observation to tensor format based on model type
                            if args.model_type == 'baku':
                                tensor_obs = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                                _norm_stats = {'actions': {'min': 0, 'max': 1}, 'proprioceptive': {'min': 0, 'max': 1}}
                                action = agent.act(
                                    obs=tensor_obs,
                                    prompt={"task_emb": task_emb},
                                    norm_stats=_norm_stats,
                                    step=steps, 
                                    global_step=steps,
                                    eval_mode=True
                                )
                            elif args.model_type == 'bc_baku':
                                # Prepare observation for bc_baku
                                tensor_obs = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                                action = agent.get_action(tensor_obs, steps)
                            else:  # transformer
                                tensor_obs = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                                action = agent.get_action(tensor_obs)

                            # Step the environment
                            obs, reward, done, info = env.step(action)
                            
                            # Record video (use first environment's observation)
                            video_writer.append_obs(obs[0], done[0], camera_name="agentview_image")
                            
                            success = info[0].get('success', False)
                            if done[0]:
                                break

                    env.close()
                    
                    successes.append(success)
                    eval_stats = {
                        "attempt": eval_idx + 1,
                        "success": success,
                        "total_steps": steps,
                    }
                    all_eval_stats.append(eval_stats)
                    print(f"Attempt {eval_idx + 1} completed: Success = {success}, Steps = {steps}")
            
            # Calculate per-task statistics
            success_rate = sum(successes) / len(successes)
            task_stats = {
                "individual_attempts": all_eval_stats,
                "success_rate": success_rate,
                "total_evaluations": args.num_evals
            }
            
            all_tasks_results[f"task_{task_id}_{task_name}"] = task_stats
            print(f"Task {task_id} Success Rate: {success_rate * 100:.2f}%")

        # Save results for this weight file
        final_stats = {
            "weight_file": weight_name,
            "tasks_results": all_tasks_results,
            "overall_success_rate": np.mean([task["success_rate"] for task in all_tasks_results.values()]),
            "total_tasks": num_tasks,
            "evaluations_per_task": args.num_evals
        }
        
        save_path = os.path.join(results_dir, f"eval_stats_{weight_name}.json")
        with open(save_path, 'w') as f:
            json.dump(final_stats, f, indent=4)

        print(f"\nEvaluation completed for {weight_name}!")
        print(f"Overall Success Rate: {final_stats['overall_success_rate'] * 100:.2f}%")
        print(f"Results saved to {save_path}")

if __name__ == '__main__':
    main()
