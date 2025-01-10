import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# OpenGL and GPU settings
os.environ["MUJOCO_GL"] = "osmesa"  # Use EGL for headless rendering
os.environ["EGL_DEVICE_ID"] = "-1"  # Use first GPU
import json
import hydra
import torch
import numpy as np
from easydict import EasyDict
from omegaconf import OmegaConf
import yaml
import argparse
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from models.bc_baku_policy import BCBakuPolicy
from libero.lifelong.models.base_policy import get_policy_class
from utils import evaluate_multitask_training_success
from libero.lifelong.utils import control_seed, get_task_embs, torch_load_model
import json
def get_model(model_type, cfg, shape_meta=None):
    if model_type == "baku":
        return BCBakuPolicy(
            repr_dim=512,
            act_dim=7,
            hidden_dim=cfg.hidden_dim,
            policy_head=cfg.policy_head,
            obs_type=cfg.obs_type,
            history=cfg.history,
            history_len=cfg.history_len,
            temporal_agg=cfg.temporal_agg,
            use_proprio=cfg.use_proprio,
            language_fusion="film" if cfg.film else None,
            num_queries=cfg.num_queries,
            max_episode_len=cfg.max_episode_len,
            device=cfg.device
        ).to(cfg.device)
    elif model_type == "transformer":
        BCTransformerPolicy = get_policy_class("bctransformerpolicy")
        return BCTransformerPolicy(cfg, shape_meta).to(cfg.device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

@hydra.main(config_path="sim_env/LIBERO/libero/configs", config_name="config")
def main(hydra_cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='transformer', choices=['baku', 'transformer'],
                      help='Type of model to evaluate (baku or transformer)')
    args = parser.parse_args()

    # Convert hydra config to regular config
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # Override with evaluation settings
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model_type = args.model_type
    cfg.seed = 2
    cfg.eval.n_eval = 10  # Number of evaluation episodes per task
    cfg.eval.max_steps = 650  # Maximum steps per episode
    cfg.eval.num_procs = 1  # Reduce number of processes to avoid GPU memory issues
    cfg.eval.use_mp = False  # Disable multiprocessing to avoid OpenGL context issues
    cfg.eval.eval = True
    cfg.eval.save_sim_states = False
    cfg.eval.render_gpu_device_id = 0  # Specify GPU device for rendering

    # Set paths
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    cfg.benchmark_name = "libero_spatial"

    # Update observation key mapping for robosuite environment
    if args.model_type == "transformer":
        # Get base directory for configs
        base_dir = Path("/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO/libero/configs/policy")
        
        # Load default transformer policy config
        transformer_cfg = OmegaConf.load(base_dir / "bc_transformer_policy.yaml")
        # Load all default configs
        color_aug_cfg = OmegaConf.load(base_dir / "data_augmentation/batch_wise_img_color_jitter_group_aug.yaml")
        translation_aug_cfg = OmegaConf.load(base_dir / "data_augmentation/translation_aug.yaml")
        image_encoder_cfg = OmegaConf.load(base_dir / "image_encoder/resnet_encoder.yaml")
        language_encoder_cfg = OmegaConf.load(base_dir / "language_encoder/mlp_encoder.yaml")
        position_encoding_cfg = OmegaConf.load(base_dir / "position_encoding/sinusoidal_position_encoding.yaml")
        policy_head_cfg = OmegaConf.load(base_dir / "policy_head/gmm_head.yaml")
        
        # Update image encoder config to match checkpoint
        image_encoder_cfg.network_kwargs.update({
            "backbone_type": "resnet18",
            "pool_type": "none",
            "use_group_norm": True,
            "use_film": True,
            "input_channel": 3,
            "image_size": 128
        })
        
        # Merge all configs
        cfg.policy = OmegaConf.merge(transformer_cfg)
        cfg.policy.color_aug = color_aug_cfg
        cfg.policy.translation_aug = translation_aug_cfg
        cfg.policy.image_encoder = image_encoder_cfg
        cfg.policy.language_encoder = language_encoder_cfg
        cfg.policy.temporal_position_encoding = position_encoding_cfg
        cfg.policy.policy_head = policy_head_cfg
        
        # Update observation modalities
        cfg.data.obs.modality = {
            "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
            "depth": [],
            "low_dim": ["robot0_gripper_qpos", "robot0_joint_pos"]
        }
        # Update key mapping to match environment keys
        cfg.data.obs.key_mapping = {
            "agentview_image": "agentview_image",
            "robot0_eye_in_hand_image": "robot0_eye_in_hand_image",
            "robot0_gripper_qpos": "robot0_gripper_qpos",
            "robot0_joint_pos": "robot0_joint_pos"
        }
        # Update shape meta to match environment keys
        shape_meta = {
            "all_shapes": {
                "agentview_image": (3, 128, 128),
                "robot0_eye_in_hand_image": (3, 128, 128)
            },
            "ac_dim": 7
        }

    # Get benchmark and task embeddings
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_tasks = benchmark.n_tasks
    
    # Get task descriptions and embeddings
    descriptions = []
    for i in range(n_tasks):
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
    
    # Get task embeddings
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # Get shape meta for transformer model
    shape_meta = None
    if args.model_type == "transformer":
        shape_meta = {
            "all_shapes": {
                "agentview_image": (3, 128, 128),
                "robot0_eye_in_hand_image": (3, 128, 128)
            },
            "ac_dim": 7
        }

    # Path to model checkpoints
    model_dir = "/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/outputs/2024-12-28/full_train/experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_001"
    
    # Get all model checkpoints
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    model_files.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]) if 'ep' in x else float('inf'))

    # Load config.json
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        cfg_dict = json.load(f)
        cfg = EasyDict(cfg_dict)
    shape_meta = cfg.shape_meta
    cfg.eval.num_procs = 1  # Reduce number of processes to avoid GPU memory issues
    cfg.eval.use_mp = False  # Disable multiprocessing to avoid OpenGL context issues
    cfg.model_type = args.model_type
    print("\n=================== Evaluation Information ===================")
    print(f"Model type: {args.model_type}")
    print(f"Benchmark: {cfg.benchmark_name}")
    print(f"Number of tasks: {n_tasks}")
    print(f"Device: {cfg.device}")
    print(f"Number of evaluation episodes per task: {cfg.eval.n_eval}")
    print("==========================================================\n")

    
    # Initialize model
    model = get_model(args.model_type, cfg, shape_meta)

    # Evaluate each model checkpoint
    for model_file in model_files:
        print(f"\nEvaluating model: {model_file}")
        
        # Load model checkpoint using torch_load_model
        model_path = os.path.join(model_dir, model_file)
        state_dict, loaded_cfg, _ = torch_load_model(model_path, map_location=cfg.device)
        
        # Update config with loaded config
        cfg.update(loaded_cfg)
        
        # Load state dict into model
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)
        model.eval()
        
        # Evaluate on all tasks
        task_ids = list(range(n_tasks))
        success_rates = evaluate_multitask_training_success(cfg, benchmark, task_ids, model)
        
        # Print results
        print("\nTask-wise success rates:")
        for task_id, success_rate in enumerate(success_rates):
            print(f"Task {task_id}: {success_rate:.4f}")
        print(f"Average success rate: {np.mean(success_rates):.4f}")
        print("==========================================================")

if __name__ == "__main__":
    main()
