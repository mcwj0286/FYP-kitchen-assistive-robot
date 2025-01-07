import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# OpenGL and GPU settings
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for headless rendering
os.environ["EGL_DEVICE_ID"] = "0"  # Use first GPU
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
from libero.lifelong.utils import control_seed, get_task_embs

def load_transformer_config():
    """Load and merge all necessary config files for transformer policy"""
    # Get absolute path to config directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(current_dir, "sim_env/LIBERO/libero/configs")
    
    # Load base transformer policy config
    with open(os.path.join(config_dir, "policy/bc_transformer_policy.yaml"), 'r') as f:
        policy_cfg = yaml.safe_load(f)
    
    # Load data augmentation configs
    with open(os.path.join(config_dir, "policy/data_augmentation/default.yaml"), 'r') as f:
        aug_cfg = yaml.safe_load(f)
    
    # Load image encoder config
    with open(os.path.join(config_dir, "policy/image_encoder/default.yaml"), 'r') as f:
        img_cfg = yaml.safe_load(f)
    
    # Load language encoder config
    with open(os.path.join(config_dir, "policy/language_encoder/default.yaml"), 'r') as f:
        lang_cfg = yaml.safe_load(f)
    
    # Load policy head config
    with open(os.path.join(config_dir, "policy/policy_head/default.yaml"), 'r') as f:
        head_cfg = yaml.safe_load(f)
    
    # Load position encoding config
    with open(os.path.join(config_dir, "policy/position_encoding/default.yaml"), 'r') as f:
        pos_cfg = yaml.safe_load(f)
    
    # Merge all configs
    cfg = {
        "policy": {
            **policy_cfg,
            "color_aug": aug_cfg["color_aug"],
            "translation_aug": aug_cfg["translation_aug"],
            "image_encoder": img_cfg["image_encoder"],
            "language_encoder": lang_cfg["language_encoder"],
            "policy_head": head_cfg["policy_head"],
            "temporal_position_encoding": pos_cfg["temporal_position_encoding"],
        }
    }
    
    return EasyDict(cfg)

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
    cfg.eval = EasyDict()
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

    # Additional config for transformer model
    if args.model_type == "transformer":
        cfg.data = EasyDict()
        cfg.data.task_order_index = 0
        cfg.data.max_word_len = 25
        cfg.data.img_h = 128  # Add image height
        cfg.data.img_w = 128  # Add image width
        cfg.data.obs = EasyDict()
        cfg.data.obs.modality = EasyDict()
        cfg.data.obs.modality.rgb = ["agentview_rgb", "robot0_eye_in_hand_rgb"]
        cfg.data.obs.key_mapping = {
            "agentview_rgb": "agentview_image",
            "robot0_eye_in_hand_rgb": "robot0_eye_in_hand_image"
        }
        cfg.data.use_joint = True
        cfg.data.use_gripper = True
        cfg.data.use_ee = False
    else:  # baku model
        cfg.data = EasyDict()
        cfg.data.img_h = 128
        cfg.data.img_w = 128
        cfg.data.task_order_index = 0

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
                "agentview_rgb": (3, 128, 128),
                "robot0_eye_in_hand_rgb": (3, 128, 128)
            },
            "ac_dim": 7
        }

    # Initialize model based on type
    model = get_model(args.model_type, cfg, shape_meta)

    # Path to model checkpoints
    model_dir = "/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/outputs/2024-12-28/full_train/experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_001"
    
    # Get all model checkpoints
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    model_files.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]) if 'ep' in x else float('inf'))

    print("\n=================== Evaluation Information ===================")
    print(f"Model type: {args.model_type}")
    print(f"Benchmark: {cfg.benchmark_name}")
    print(f"Number of tasks: {n_tasks}")
    print(f"Device: {cfg.device}")
    print(f"Number of evaluation episodes per task: {cfg.eval.n_eval}")
    print("==========================================================\n")

    # Evaluate each model checkpoint
    for model_file in model_files:
        print(f"\nEvaluating model: {model_file}")
        
        # Load model checkpoint
        checkpoint = torch.load(os.path.join(model_dir, model_file), map_location=cfg.device)
        
        # Load model state based on model type
        if args.model_type == "transformer":
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:  # baku model
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        
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
