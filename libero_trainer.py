import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing
import pprint
import time
from pathlib import Path

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
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    create_experiment_dir,
    get_task_embs,
)
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from dataset import TransformedDataset
from models.bc_baku_policy import BCBakuPolicy

@hydra.main(config_path="sim_env/LIBERO/libero/configs", config_name="config")
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # BAKU config overrides
    cfg.device = "cuda"
    cfg.seed = 2
    cfg.train.batch_size = 64
    cfg.train.lr = 1e-4
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
    cfg.train.n_epochs = 100  # Adjust as needed

    # print configs
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    # prepare paths
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    # you can specify the benchmark name here
    cfg.benchmark_name = "libero_object" #{"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    # get benchmark and number of tasks
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets
    manip_datasets = []
    descriptions = []
    shape_meta = None

    # Load datasets for each task
    for i in range(n_manip_tasks):
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}")
            print(f"[error] {e}")
            continue

        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    # Get task embeddings
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # Create datasets
    datasets = [
        SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
    ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]

    # Print benchmark information
    print("\n=================== Benchmark Information ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks}")
    for i in range(n_manip_tasks):
        print(f"    - Task {i+1}: {benchmark.get_task(i).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    # print(ConcatDataset(manip_datasets)[0])
    # Create combined dataset
    concat_dataset = TransformedDataset(ConcatDataset(datasets))
    
    # Create data loader
    train_dataloader = DataLoader(
        concat_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,  # Set to 0 to avoid deadlock
        sampler=RandomSampler(concat_dataset),
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
        device=cfg.device
    ).to(cfg.device)

    # Initialize optimizer with BAKU learning rate
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    
    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print("[info] Starting training...")
    best_loss = float('inf')
    
    for epoch in range(cfg.train.n_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, gt_actions) in enumerate(train_dataloader):
            # Move data to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(cfg.device)
            gt_actions = gt_actions.to(cfg.device)
            # Print shapes of data and ground truth actions
            print("Data shapes:")
            for k, v in data.items():
                print(f"{k}: {v.shape}")
            print(f"Ground truth actions shape: {gt_actions.shape}")
            # Forward pass
            pred_actions = model.forward(data)
            
            # Compute loss
            loss = criterion(pred_actions, gt_actions)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(cfg.experiment_dir, 'best_model.pth'))
            print(f"Saved new best model with loss: {best_loss:.4f}")

    print("[info] Training completed")

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
