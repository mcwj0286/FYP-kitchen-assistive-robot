import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing
import pprint
import time
from pathlib import Path
import logging
import json
import wandb
from dotenv import load_dotenv

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from easydict import EasyDict

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
from models.bc_act_policy import bc_act_policy
from models.moe_policy import moe_policy, ModelArgs
from models.bc_moe_policy import bc_moe_policy


def get_model(model_type,cfg):
    """Initialize model based on type and return both model and config.
    
    Args:
        model_type (str): Type of model to initialize ('baku', 'transformer', or 'act')
        
    Returns:
        tuple: (model, cfg) where model is the initialized model and cfg is its configuration
    """
    cfg.overlap = 0
    # Set default parameters
    cfg.get_pad_mask = False
    cfg.get_action_padding = False
   
    cfg.num_queries = 1
    # cfg.eval.num_procs = 5
    # cfg.eval.use_mp = True
    # cfg.eval.n_eval = 5
    cfg.seq_length = 10
    if model_type == "bc_baku":
        # Default configuration for BCBaku
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
        cfg.policy_head = cfg.policy_head
        cfg.use_proprio = True
        cfg.use_language = True
        cfg.temporal_agg = True
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
        cfg.seed = 32
        # cfg.train.batch_size = 64
        cfg.train.lr = 1e-4
        cfg.train.optimizer.name = 'torch.optim.Adam'
        cfg.train.optimizer.kwargs.lr = 1e-4
        cfg.train.optimizer.kwargs.betas = [0.9, 0.999]
        cfg.train.n_epochs = 100
        cfg.obs_type = "pixels"
        cfg.policy_head = cfg.policy_head
        cfg.use_proprio = True
        cfg.temporal_agg = True
        # num_queries from command line
        cfg.hidden_dim = 256
        cfg.history = False
        cfg.history_len = 10
        cfg.film = True
        cfg.max_episode_len = 650
        cfg.seq_length = 10

        cfg.overlap =3

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
            use_proprio=cfg.use_proprio,
            use_mpi_pixels_egocentric =False

        ).to(cfg.device)
    elif model_type == "bc_act":
        # Configuration for BC-ACT

        cfg.action_dim = 7
        # Optimizer settings
        cfg.train.optimizer = EasyDict()
        cfg.train.optimizer.name = 'torch.optim.AdamW'
        cfg.train.lr = 1e-4
        cfg.train.optimizer.kwargs = EasyDict()
        cfg.train.optimizer.kwargs.lr = cfg.train.lr
        cfg.train.optimizer.kwargs.weight_decay = 1e-4
        
        # Model parameters
        cfg.obs_type = "pixels"
        cfg.policy_head = cfg.policy_head
        cfg.use_proprio = True
        # cfg.num_queries is now set from command line args
   
        cfg.film = True
        cfg.max_episode_len = 650
        cfg.seq_length = 10
        cfg.get_pad_mask = True
        cfg.get_action_padding = True
        cfg.overlap = 0

      
        
        # Initialize BC-ACT model
        model = bc_act_policy(
            repr_dim=cfg.repr_dim,
            act_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            policy_head=cfg.policy_head,
            obs_type=cfg.obs_type,
            obs_shape={
                'pixels': (3, 128, 128),
                'pixels_egocentric': (3, 128, 128),
                'proprioceptive': (9,),
            },
            language_dim=768,
            lang_repr_dim=cfg.repr_dim,
            language_fusion="film" if cfg.film else None,
            pixel_keys=['pixels', 'pixels_egocentric'],
            proprio_key='proprioceptive',
            device=cfg.device,
            num_queries=cfg.num_queries,
            max_episode_len=cfg.max_episode_len,
            use_proprio=cfg.use_proprio,
            learnable_tokens=False,
            n_layer=cfg.n_layer,
            use_moe=cfg.use_moe,
            use_mpi=cfg.use_mpi,
            mpi_root_dir=cfg.mpi_root_dir,
            benchmark_name=cfg.benchmark_name
        ).to(cfg.device)
    elif model_type == "moe":
        # Configuration for MoE policy
        cfg.seed = 2
        cfg.train.batch_size = 16
        cfg.train.lr = 1e-4
        cfg.train.optimizer.name = 'torch.optim.Adam'
        cfg.train.optimizer.kwargs.lr = 1e-4
        cfg.train.optimizer.kwargs.betas = [0.9, 0.999]
        cfg.train.n_epochs = 100
        cfg.obs_type = "pixels"
        cfg.policy_head = cfg.policy_head
        cfg.use_proprio = True
        cfg.temporal_agg = False
        cfg.num_queries = 1
        cfg.hidden_dim = 256
        cfg.history = False
        cfg.history_len = 1
        cfg.film = True
        cfg.max_episode_len = 650
        cfg.seq_length = 10
        
        # MoE specific configurations
        moe_args = ModelArgs(
            max_batch_size=cfg.train.batch_size,
            max_seq_len=cfg.max_episode_len,
            dim=512,  # Model dimension
            inter_dim=cfg.hidden_dim,
            moe_inter_dim=cfg.hidden_dim,
            n_layers=8,
            n_dense_layers=2,
            n_heads=8,
            n_routed_experts=64,
            n_shared_experts=2,
            n_activated_experts=6,
            action_dim=7,
            dtype="bf16",
            score_func = 'softmax'
        )

        # Initialize MoE model
        config = {
        'repr_dim': 512,
        'act_dim': 7,
        'hidden_dim': 256,
        'policy_head': cfg.policy_head,
        'obs_type': 'pixels',
        'obs_shape': {
            'pixels': (3, 128, 128),
            'pixels_egocentric': (3, 128, 128),
            'proprioceptive': (9,),
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'history_len': 10,
        'temporal_agg': cfg.temporal_agg,  # Set to False for simplicity in this test
        'num_queries': cfg.num_queries,
        'moe_args': moe_args
    }

        # Initialize the model
        model = moe_policy(**config).to(config['device'])
    elif model_type == "bc_moe":
        # Configuration for BC-MoE policy
        cfg.seed = 2
        cfg.train.batch_size = 16
        cfg.train.lr = 1e-4
        cfg.train.optimizer.name = 'torch.optim.Adam'
        cfg.train.optimizer.kwargs.lr = 1e-4
        cfg.train.optimizer.kwargs.betas = [0.9, 0.999]
        cfg.train.n_epochs = 100
        cfg.obs_type = "pixels"
        cfg.policy_head = cfg.policy_head
        cfg.use_proprio = True
        cfg.temporal_agg = True  # Set to False to avoid dimension mismatch
        # Use num_queries from command line 
        cfg.hidden_dim = 256
        cfg.history = False       # Simplify training initially
        cfg.history_len = 1
        cfg.film = True
        cfg.max_episode_len = 650
        cfg.seq_length = 10

        # Initialize BC-MoE model
        model = bc_moe_policy(
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
            use_proprio=cfg.use_proprio,
            n_shared_experts=1
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

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Add model type selection
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='bc_act', 
                      choices=['bc_baku', 'transformer', 'act', 'bc_transformer', 'bc_act', 'moe', 'bc_moe'],
                      help='Type of model to train')
    parser.add_argument('--eval_sample_size', type=int, default=10,
                      help='Number of tasks to sample for evaluation. If None, all tasks will be evaluated.')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                      help='Enable Weights & Biases logging')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run the model on')
    parser.add_argument('--n_layer', type=int, default=8,
                      help='Number of layers in the model')
    parser.add_argument('--repr_dim', type=int, default=512,
                      help='Representation dimension for the model')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension for the model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_queries', type=int, default=10,
                      help='Number of action tokens/queries for models that use them')
    parser.add_argument('--use_moe', action='store_true', default=False, 
                      help='Use MoE in bc_act model')
    parser.add_argument('--use_mpi', action='store_true', default=False,
                      help='Use MPI vision encoder instead of ResNet18')
    parser.add_argument('--mpi_root_dir', type=str, 
                      default="/home/john/project/FYP-kitchen-assistive-robot/models/networks/utils/MPI/mpi/checkpoints/mpi-small",
                      help='Path to MPI model checkpoints')
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                      choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                      help='Which benchmark to use for training')
    parser.add_argument('--policy_head', type=str, default='deterministic',
                      choices=['deterministic', 'task_specific_head'],
                      help='Type of policy head to use')
    
    args = parser.parse_args()
    
    # Create base config with all required components
    cfg = EasyDict({
        'device': args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu',
        'n_layer': args.n_layer,
        'repr_dim': args.repr_dim,
        'hidden_dim': args.hidden_dim,
        'use_moe': args.use_moe,
        'use_mpi': args.use_mpi,
        'mpi_root_dir': args.mpi_root_dir,
        'benchmark_name': args.benchmark,
        'model_type': args.model_type,
        'policy_head': args.policy_head,
        'train': {
            'batch_size': args.batch_size,
            'n_epochs': 100,
            'accumulation_steps': 4,  # Add gradient accumulation steps
            'optimizer': {
                'name': 'torch.optim.Adam',
                'kwargs': {
                    'lr': 1e-4,
                    'betas': [0.9, 0.999]
                }
            }
        },
        'eval': {
            'eval_every': 5,
            'max_steps': 650,
            'use_mp': True,
            'num_procs': 5,
            'n_eval': 5
        },
        'data': {
            'task_order_index': 0,
            'seq_len': 10,
            'img_h': 128,
            'img_w': 128,
            'img_c': 3
        },
        'seed': 2,
        'overlap': 0,
        'get_pad_mask': False,
        'get_action_padding': False,
        'num_queries': args.num_queries,
        'seq_length': 10
    })

    # Create experiment directory for saving checkpoints
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cfg.experiment_dir = os.path.join("experiments", f"{cfg.model_type}_{timestamp}")
    os.makedirs(cfg.experiment_dir, exist_ok=True)

    # Setup logging after creating experiment directory
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(cfg.experiment_dir, 'libero_trainer.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Created experiment directory at: {cfg.experiment_dir}")

    # Get model and its configuration
    model, cfg = get_model(args.model_type, cfg)
    cfg.model_type = args.model_type

    # Print total number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_params/1e6:.2f}M")

    # Initialize wandb if enabled - moved after model configuration
    if args.use_wandb:
        # Get WANDB API key from environment
        wandb_key = os.getenv('WANDB_API_KEY')
        if wandb_key is None:
            logger.error("WANDB_API_KEY not found in .env file")
            raise ValueError("WANDB_API_KEY not found in .env file")
        
        # Login to wandb
        try:
            wandb.login(key=wandb_key)
            logger.info("Successfully logged in to Weights & Biases")
        except Exception as e:
            logger.error(f"Failed to login to Weights & Biases: {str(e)}")
            raise

        # Initialize wandb with all available config parameters
        wandb.init(
            project="fyp-kitchen-assistive-robot",
            name=f"{args.model_type}_{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "model_type": args.model_type,
                "seed": cfg.seed,
                "batch_size": cfg.train.batch_size,
                "learning_rate": cfg.train.optimizer.kwargs.lr,
                "num_epochs": cfg.train.n_epochs,
                "eval_sample_size": args.eval_sample_size,
                "benchmark_name": cfg.benchmark_name,
                "seq_length": cfg.seq_length,
                "num_queries": cfg.num_queries,
                "hidden_dim": cfg.hidden_dim,
                "trainable_parameters": num_params/1e6,
                "device": cfg.device,
                "n_layer": cfg.n_layer,
                "repr_dim": cfg.repr_dim,
                "use_moe": cfg.use_moe,
                "use_mpi": cfg.use_mpi,
            }
        )

    
    # Print configs
    pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(cfg)

    # Control seed
    control_seed(cfg.seed)

    # Save experiment config
    config_path = os.path.join(cfg.experiment_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    logger.info(f"Saved experiment config to: {config_path}")

    # Prepare paths
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder =get_libero_path("init_states")
    
    # Get benchmark and task information
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # Setup evaluation task IDs based on sampling option
    if args.eval_sample_size is not None and args.eval_sample_size < n_manip_tasks:
        np.random.seed(cfg.seed)  # Use same seed for reproducibility
        eval_task_ids = np.random.choice(n_manip_tasks, size=args.eval_sample_size, replace=False)
        logger.info(f"Selected {len(eval_task_ids)} tasks for evaluation: {eval_task_ids}")
    else:
        eval_task_ids = list(range(n_manip_tasks))  # Use all tasks
        logger.info("Using all tasks for evaluation")

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
        overlap=cfg.overlap
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
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Reduce LR by half when plateauing
        patience=5,   # Number of epochs to wait before reducing LR
        verbose=True,
        min_lr=1e-6  # Minimum learning rate
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_success = 0.0
    task_ids = list(range(n_manip_tasks))  # Full task list for training
    best_task_success_rates = np.zeros(len(eval_task_ids))  # Track best success rates for eval tasks
    
    for epoch in range(cfg.train.n_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, obs_dict in enumerate(train_dataloader):
            # Move tensors in obs_dict to cfg.device
            for key, value in obs_dict.items():
                if isinstance(value, torch.Tensor):
                    obs_dict[key] = value.to(cfg.device)
                    
            # Use the train_step function with optimizer and scheduler
            loss_output = model.train_step(obs_dict, optimizer=optimizer)
            
            if isinstance(loss_output, dict):
                loss = loss_output.get("loss", None)
                if loss is None:
                    raise KeyError("train_step returned a dict without key 'loss'")
            else:
                loss = loss_output
            

            if isinstance(loss, torch.Tensor):
                total_loss += loss.item()
            else:
                total_loss += float(loss)
                
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}")
        
        # Log training metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "train/learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Step the scheduler based on average training loss
        scheduler.step(avg_train_loss)
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation step every eval_every epochs
        if epoch > 0 and epoch % cfg.eval.eval_every == 0:
            logger.info(f"Running validation at epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                success_rates = evaluate_multitask_training_success(cfg, benchmark, eval_task_ids, model)
                avg_success = np.mean(success_rates)
                logger.info(f"Validation Success Rates for sampled tasks: {success_rates}")
                logger.info(f"Average Success Rate: {avg_success:.4f}")
                
                # Log validation metrics to wandb
                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "val/avg_success_rate": avg_success,
                        **{f"val/task_{task_id}_success": rate 
                           for task_id, rate in zip(eval_task_ids, success_rates)}
                    })
                
                # Save checkpoint if improved
                if avg_success > best_val_success:
                    best_val_success = avg_success
                    save_path = os.path.join(cfg.experiment_dir, f'model_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                        'success_rates': success_rates,
                        'avg_success': avg_success,
                    }, save_path)
                    logger.info(f"Saved best model checkpoint with success rate {avg_success:.4f}")
                    
                    # Log best model metrics to wandb
                    if args.use_wandb:
                        wandb.run.summary["best_success_rate"] = avg_success
                        wandb.run.summary["best_epoch"] = epoch

    # Save final model checkpoint
    final_save_path = os.path.join(cfg.experiment_dir, f'model_final.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_save_path)
    logger.info(f"Saved final model checkpoint at {final_save_path}")

    # Close wandb run
    if args.use_wandb:
        wandb.finish()

    logger.info("Training completed")
    train_dataset.close()

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
