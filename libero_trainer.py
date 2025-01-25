import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing
import pprint
import time
from pathlib import Path
import logging
import json

import hydra
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from torch.utils.data import DataLoader, RandomSampler
from dataset import LIBERODataset
from models.bc_baku_policy import BCBakuPolicy
from models.bc_transformer_policy import bc_transformer_policy
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from utils import evaluate_multitask_training_success
from libero.lifelong.utils import control_seed, get_task_embs
from models.act_policy import ACTPolicy

def get_model(model_type,cfg):
    """Initialize model based on type and return both model and config.
    
    Args:
        model_type (str): Type of model to initialize ('baku', 'transformer', or 'act')
        
    Returns:
        tuple: (model, cfg) where model is the initialized model and cfg is its configuration
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == "bc_baku":
        # Default configuration for BCBaku
        cfg.device = device
        cfg.seed = 2
        cfg.train.batch_size = 64
        cfg.train.lr = 1e-5
        cfg.train.n_epochs = 50
        cfg.train.optimizer.name = 'torch.optim.Adam'
        cfg.train.optimizer.kwargs.lr = 1e-5
        cfg.train.optimizer.kwargs.betas = [0.9, 0.999]
        cfg.data.task_order_index = 0
        cfg.data.seq_len = 10
        cfg.obs_type = "pixels"
        cfg.policy_type = "gpt"
        cfg.policy_head = "deterministic"
        cfg.use_proprio = True
        cfg.use_language = True
        cfg.temporal_agg = True
        cfg.num_queries = 10
        cfg.hidden_dim = 256
        cfg.history = True
        cfg.history_len = 10
        cfg.eval_history_len = 10
        cfg.film = True
        cfg.max_episode_len = 650
        cfg.eval.n_eval = 10
        cfg.eval.eval_every = 5
        cfg.eval.max_steps = 650
        cfg.eval.use_mp = True
        cfg.eval.num_procs = 5
        cfg.eval.n_eval = 5
        cfg.seq_length = 10
        model = BCBakuPolicy(
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
        # Load transformer config from json
        transformer_cfg_path = "/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/outputs/2024-12-28/full_train/experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_001/config.json"
        with open(transformer_cfg_path, 'r') as f:
            cfg = EasyDict(json.load(f))
        
        # Update device and training settings
        cfg.device = device
        if not hasattr(cfg.train, 'lr'):
            cfg.train.lr = cfg.train.optimizer.kwargs.lr
        cfg.seq_length = 10
        # Initialize transformer model
        model = BCTransformerPolicy(cfg, cfg.shape_meta).to(cfg.device)
        
    elif model_type == "act":
        # Default configuration for ACT
        cfg.device = device
        cfg.seed = 2
        cfg.train.batch_size = 64
        cfg.train.n_epochs = 100
        
        # ACT optimizer settings
        cfg.optimizer = EasyDict()
        cfg.optimizer.name = 'torch.optim.AdamW'
        cfg.lr = 1e-4  # Main learning rate
        cfg.lr_backbone = 1e-5  # Backbone learning rate
        cfg.weight_decay = 1e-4
        
        # Core model parameters required by detr_vae.py
        cfg.hidden_dim = 512  # Transformer hidden dimension
        cfg.nheads = 8  # Number of attention heads
        cfg.enc_layers = 4  # Encoder layers
        cfg.dec_layers = 7  # Decoder layers
        cfg.dim_feedforward = 2048
        cfg.dropout = 0.1
        cfg.pre_norm = False
        cfg.num_queries = 10  # Chunk size
        cfg.state_dim = 9  # 7 DoF joints + 2 gripper
        cfg.action_dim = 7   # 6 DoF + 1 gripper
        cfg.camera_names = ['pixels', 'pixels_egocentric']
        cfg.multitask = True
        cfg.obs_type = "pixels"
        cfg.max_episode_len = 650
        # Backbone settings required by backbone.py
        cfg.backbone = 'resnet18'
        cfg.position_embedding = 'sine'
        cfg.dilation = False
        cfg.masks = False
        cfg.seq_length = 1
        cfg.get_pad_mask = True
        cfg.get_action_padding = True
        # Initialize ACT model
        model = ACTPolicy(
            cfg=cfg,
            kl_weight=10.0,  # From official ACT implementation
            device=cfg.device,
        )
    elif model_type == "bc_transformer": #custom implementation from libero transformer
        # Configuration for our bc_transformer_policy
        cfg.device = device
        cfg.seed = 2
        cfg.train.batch_size = 32
        cfg.train.lr = 1e-5
        cfg.train.n_epochs = 100
        cfg.obs_type = "pixels"
        cfg.policy_head = "deterministic"
        cfg.use_proprio = True
        cfg.temporal_agg = True
        cfg.num_queries = 10
        cfg.hidden_dim = 256
        cfg.history = True
        cfg.history_len = 10
        cfg.film = True
        cfg.max_episode_len = 650
        cfg.seq_length = 10
        cfg.get_pad_mask = False
        cfg.get_action_padding = False
        # Initialize our transformer model
        model = bc_transformer_policy(
            repr_dim=512,
            act_dim=7,
            hidden_dim=cfg.hidden_dim,
            policy_head=cfg.policy_head,
            obs_type=cfg.obs_type,
            obs_shape={
                'pixels': (3, 128, 128),
                'pixels_egocentric': (3, 128, 128),
                'proprioceptive': (9,),
            },
            language_dim=768,
            lang_repr_dim=512,
            language_fusion="film" if cfg.film else None,
            pixel_keys=['pixels', 'pixels_egocentric'],
            proprio_key='proprioceptive',
            device=cfg.device,
            history=cfg.history,
            history_len=cfg.history_len,
            num_queries=cfg.num_queries,
            temporal_agg=cfg.temporal_agg,
            max_episode_len=cfg.max_episode_len,
            use_proprio=cfg.use_proprio
        ).to(cfg.device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model, cfg

def convert_to_transformer_format(data_dict):
    """Convert dataset output format to transformer input format.
    
    Args:
        data_dict (dict): Dataset output with keys:
            - pixels: [B, T, 3, 128, 128]
            - pixels_egocentric: [B, T, 3, 128, 128]
            - proprioceptive: [B, T, 9]
            - task_emb: [B, 768]
            
    Returns:
        dict: Transformer format with:
            obs: {
                agentview_rgb: [B, T, 3, 128, 128],
                eye_in_hand_rgb: [B, T, 3, 128, 128],
                gripper_states: [B, T, 2],
                joint_states: [B, T, 7]
            },
            task_emb: [B, 768]
    """
    # Split proprioceptive into gripper and joint states
    gripper_states = data_dict["proprioceptive"][..., :2]  # First 2 dimensions
    joint_states = data_dict["proprioceptive"][..., 2:9]   # Next 7 dimensions
    
    return {
        "obs": {
            "agentview_rgb": data_dict["pixels"],
            "eye_in_hand_rgb": data_dict["pixels_egocentric"],
            "gripper_states": gripper_states,
            "joint_states": joint_states
        },
        "task_emb": data_dict["task_emb"]
    }

#loss function 
def loss_fn(dist, target, reduction="mean", model_type="bc_baku", **kwargs):
    """Compute loss based on model type"""
    if model_type == "act":
        # ACT returns a loss dictionary with l1, kl, and total loss
        return dist  # dist is actually the loss_dict for ACT
    else:
        # Original loss computation for other models
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * 1.0
        else:
            raise NotImplementedError

        return loss

@hydra.main(config_path="sim_env/LIBERO/libero/configs", config_name="config")
def main(hydra_cfg):
    # Add model type selection
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='bc_transformer', choices=['bc_baku', 'transformer', 'act', 'bc_transformer'],
                      help='Type of model to train (baku, transformer, or act)')
    args = parser.parse_args()
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('libero_trainer.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Get model and its configuration
    model, cfg = get_model(args.model_type,cfg)
    cfg.model_type = args.model_type  # Add model type to config

    # Print configs
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # Control seed
    control_seed(cfg.seed)

    # Create experiment directory for saving checkpoints
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cfg.experiment_dir = os.path.join("experiments", f"{cfg.model_type}_{timestamp}")
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    logger.info(f"Created experiment directory at: {cfg.experiment_dir}")

    # Save experiment config
    config_path = os.path.join(cfg.experiment_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    logger.info(f"Saved experiment config to: {config_path}")

    # Prepare paths
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder =get_libero_path("init_states")
    
    # you can specify the benchmark name here
    cfg.benchmark_name = "libero_spatial" #{"libero_spatial", "libero_object", "libero_goal", "libero_10"}

    # Get benchmark and task information
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # Print benchmark information
    print("\n=================== Benchmark Information ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks}")
    for i in range(n_manip_tasks):
        print(f"    - Task {i+1}: {benchmark.get_task(i).language}")
    print("=======================================================================\n")
    # Set default num_queries if not specified in config
    if not hasattr(cfg, 'num_queries'):
        cfg.num_queries = 1
    if not hasattr(cfg, 'get_pad_mask'):
        cfg.get_pad_mask = False
    # Create dataset
    train_dataset = LIBERODataset(
        data_path=cfg.folder,
        benchmark=cfg.benchmark_name,
        device=cfg.device,
        load_task_emb=True,
        num_queries=cfg.num_queries,
        seq_length=cfg.seq_length,
        get_pad_mask = cfg.get_pad_mask,
        get_action_padding=cfg.get_action_padding,
    )

    # Create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        sampler=RandomSampler(train_dataset),
        pin_memory=True,
    )

    # Initialize optimizer based on model type
    if args.model_type == "act":
        # ACT uses different learning rates for backbone and other parameters
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr": cfg.lr,
            },
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=cfg.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    
    # Training loop
    logger.info("Starting training...")
    best_val_success = 0.0
    task_ids = list(range(n_manip_tasks))
    best_task_success_rates = np.zeros(n_manip_tasks)
    
    for epoch in range(cfg.train.n_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, obs_dict in enumerate(train_dataloader):
            # Move tensors to device
            for key, value in obs_dict.items():
                if isinstance(value, torch.Tensor):
                    obs_dict[key] = value.to(cfg.device)
                    
            # Get ground truth actions
            gt_actions = obs_dict["actions"]

            # Forward pass based on model type
            if args.model_type == "act":
                # Check and reshape actions if needed
                if obs_dict['actions'].shape[-1] != cfg.action_dim:
                    obs_dict['actions'] = obs_dict['actions'].view(-1, cfg.num_queries, cfg.action_dim)
                # ACT forward pass returns loss dictionary
                loss_dict = model.forward(obs_dict)
                loss = loss_dict["loss"]  # Combined L1 and KL loss
                
                # Log individual losses
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch} Batch {batch_idx} Losses: "
                              f"L1={loss_dict['l1']:.4f}, "
                              f"KL={loss_dict['kl']:.4f}, "
                              f"Total={loss_dict['loss']:.4f}")
            else:
                # Convert format for transformer if needed
                if args.model_type == "transformer":
                    obs_dict = convert_to_transformer_format(obs_dict)
                
                # Original models' forward pass
                pred_actions = model.forward(obs_dict)
                loss = loss_fn(pred_actions, gt_actions, model_type=args.model_type)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if isinstance(loss, torch.Tensor):
                total_loss += loss.item()
            else:
                total_loss += loss_dict["loss"].item()  # For ACT
        
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation step every eval_every epochs
        if epoch > 0 and epoch % cfg.eval.eval_every == 0:
            logger.info(f"Running validation at epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                success_rates = evaluate_multitask_training_success(cfg, benchmark, task_ids, model)
                avg_success = np.mean(success_rates)
                logger.info(f"Validation Success Rates: {success_rates}")
                logger.info(f"Average Success Rate: {avg_success:.4f}")
                
                # Save checkpoint if improved
                if avg_success > best_val_success:
                    best_val_success = avg_success
                    save_path = os.path.join(cfg.experiment_dir, f'model_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'success_rates': success_rates,
                        'avg_success': avg_success,
                    }, save_path)
                    logger.info(f"Saved best model checkpoint with success rate {avg_success:.4f}")
                
                # Save model checkpoint for every eval epoch
                # eval_save_path = os.path.join(cfg.experiment_dir, f'model_epoch_{epoch}.pth')
                # torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'success_rates': success_rates,
                #     'avg_success': avg_success,
                # }, eval_save_path)
                # logger.info(f"Saved evaluation checkpoint at epoch {epoch}")


    # Save final model checkpoint regardless of performance
    final_save_path = os.path.join(cfg.experiment_dir, f'model_final.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_save_path)
    logger.info(f"Saved final model checkpoint at {final_save_path}")

    logger.info("Training completed")
    train_dataset.close()

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
