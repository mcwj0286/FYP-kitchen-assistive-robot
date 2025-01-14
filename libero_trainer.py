import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing
import pprint
import time
from pathlib import Path
import logging

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
from utils import evaluate_multitask_training_success
from libero.lifelong.utils import control_seed , get_task_embs
from libero.lifelong.datasets import get_dataset

#loss function 
def loss_fn( dist, target, reduction="mean", **kwargs):
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * 1.0
        else:
            raise NotImplementedError

        return loss
@hydra.main(config_path="sim_env/LIBERO/libero/configs", config_name="config")
def main(hydra_cfg):
    # preprocessing
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

    

    # Common config overrides
    cfg.device = "cuda"
    cfg.seed = 2
    cfg.train.batch_size = 32
    cfg.train.lr = 5e-4
    cfg.obs_type = "pixels"
    cfg.policy_type = "gpt"
    cfg.policy_head = "deterministic"
    cfg.use_proprio = True
    cfg.use_language = True
    cfg.temporal_agg = False
    cfg.num_queries = 1
    cfg.hidden_dim = 256
    cfg.history = True
    cfg.history_len = 10
    cfg.eval_history_len = 10
    cfg.film = True
    cfg.train.n_epochs = 50  # Adjust as needed
    cfg.max_episode_len = 650 # has to greater than the max episode length in the evaluation
    # print configs
    pp = pprint.PrettyPrinter(indent=2)
    cfg.model_type = "baku"
    pp.pprint(cfg)

    # control seed
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
    # prepare paths
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    
    # you can specify the benchmark name here
    cfg.benchmark_name = "libero_spatial" #{"libero_spatial", "libero_object", "libero_goal", "libero_10"}

    manip_datasets = []
    descriptions = []
    shape_meta = None
    # get benchmark and number of tasks
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks
    # for i in range(n_manip_tasks):
    #     try:
    #         task_i_dataset, shape_meta = get_dataset(
    #             dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
    #             obs_modality=cfg.data.obs.modality,
    #             initialize_obs_utils=(i == 0),
    #             seq_len=cfg.data.seq_len,
    #         )
    #     except Exception as e:
    #         print(f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}")
    #         print(f"[error] {e}")
    #         continue

    #     task_description = benchmark.get_task(i).language
    #     descriptions.append(task_description)
    #     # manip_datasets.append(task_i_dataset)
    # # Get task embeddings
    # task_embs = get_task_embs(cfg, descriptions)
    # benchmark.set_task_embs(task_embs)

    # Print benchmark information
    print("\n=================== Benchmark Information ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks}")
    for i in range(n_manip_tasks):
        print(f"    - Task {i+1}: {benchmark.get_task(i).language}")
    print("=======================================================================\n")

    # Create dataset directly using LIBERODataset
    train_dataset = LIBERODataset(
        data_path=cfg.folder,
        benchmark=cfg.benchmark_name,
        device=cfg.device,
        load_task_emb=cfg.use_language,
        num_queries=cfg.num_queries
    )

    # Create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,  # Set to 0 to avoid deadlock
        sampler=RandomSampler(train_dataset),
        pin_memory=True,
    )

    # Initialize model with BAKU config
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

    # Initialize optimizer with BAKU learning rate
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    
    # Loss function
    criterion = loss_fn

    # Training loop
    print("[info] Starting training...")
    best_val_success = 0.0
    task_ids = list(range(n_manip_tasks))
    best_task_success_rates = np.zeros(n_manip_tasks)  # Track best success rate for each task
    
    for epoch in range(cfg.train.n_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (obs_dict, gt_actions) in enumerate(train_dataloader):
            # Move all tensors in obs_dict to GPU
            for key, value in obs_dict.items():
                if isinstance(value, torch.Tensor):
                    obs_dict[key] = value.to(cfg.device)
            gt_actions = gt_actions.to(cfg.device)
            # Forward pass
            pred_actions = model.forward(obs_dict)
            
            # Compute loss
            loss = criterion(pred_actions, gt_actions)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # if batch_idx % 10 == 0:
            #     print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation step every 5 epochs, but skip epoch 0
        if epoch > 0 and epoch % 5 == 0:
            logger.info(f"Running validation at epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                success_rates = evaluate_multitask_training_success(cfg, benchmark, task_ids, model)
                avg_success = np.mean(success_rates)
                logger.info(f"Validation Success Rates: {success_rates}")
                logger.info(f"Average Success Rate: {avg_success:.4f}")
                
                # Check for improvements in any task
                improved = False
                improved_tasks = []
                for task_idx, (curr_rate, best_rate) in enumerate(zip(success_rates, best_task_success_rates)):
                    if curr_rate > best_rate:
                        improved = True
                        improved_tasks.append(task_idx)
                        best_task_success_rates[task_idx] = curr_rate
                
                # Update best average success if improved
                if improved:
                    best_val_success = max(best_val_success, avg_success)
                    logger.info(f"Improvements in tasks {improved_tasks}")
                    logger.info("Task-wise improvements:")
                    for task_idx in improved_tasks:
                        logger.info(f"    Task {task_idx}: {best_task_success_rates[task_idx]:.4f}")
                    
                    # Save model checkpoint every 5 epochs regardless of improvement
                save_path = os.path.join(cfg.experiment_dir, f'model_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'task_success_rates': success_rates,
                    'best_task_success_rates': best_task_success_rates,
                    'avg_success': avg_success,
                    'improved_tasks': improved_tasks if improved else [],
                    'is_best': improved,
                }, save_path)
                logger.info(f"Saved model checkpoint at epoch {epoch}")

    logger.info("Training completed")
    # Clean up dataset
    train_dataset.close()

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
