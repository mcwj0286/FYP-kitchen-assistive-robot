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
from utils import evaluate_multitask_training_success

@hydra.main(config_path="sim_env/LIBERO/libero/configs", config_name="config")
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # BAKU config overrides
    cfg.device = "cuda"
    cfg.seed = 2
    cfg.train.batch_size = 16
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

    
    # Create combined dataset
    concat_dataset = TransformedDataset(ConcatDataset(datasets))
    # Create data loader
    train_dataloader = DataLoader(
        concat_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,  # Set to 0 to avoid deadlock
        sampler=RandomSampler(concat_dataset),
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
        device=cfg.device
    ).to(cfg.device)

    # Initialize optimizer with BAKU learning rate
    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    
    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print("[info] Starting training...")
    best_train_loss = float('inf')
    best_val_success = 0.0
    task_ids = list(range(n_manip_tasks))
    best_task_success_rates = np.zeros(n_manip_tasks)  # Track best success rate for each task
    
    for epoch in range(cfg.train.n_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, gt_actions) in enumerate(train_dataloader):
            # Move data to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(cfg.device)
            gt_actions = gt_actions.to(cfg.device)
            
            # Forward pass
            pred_actions = model.forward(data, action=gt_actions)
            
            # Compute loss
            loss = criterion(pred_actions, gt_actions)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"[info] Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation step every 5 epochs
        if epoch % 5 == 0:
            print(f"[info] Running validation at epoch {epoch}...")
            model.eval()
            with torch.no_grad():
                success_rates = evaluate_multitask_training_success(cfg, benchmark, task_ids, model)
                avg_success = np.mean(success_rates)
                print(f"[info] Validation Success Rates: {success_rates}")
                print(f"[info] Average Success Rate: {avg_success:.4f}")
                
                # Check for improvements in any task
                improved = False
                improved_tasks = []
                for task_idx, (curr_rate, best_rate) in enumerate(zip(success_rates, best_task_success_rates)):
                    if curr_rate > best_rate:
                        improved = True
                        improved_tasks.append(task_idx)
                        best_task_success_rates[task_idx] = curr_rate
                
                # Save model if there's improvement in any task
                if improved:
                    best_val_success = max(best_val_success, avg_success)  # Update best average success
                    save_path = os.path.join(cfg.experiment_dir, f'model_epoch_{epoch}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'task_success_rates': success_rates,
                        'best_task_success_rates': best_task_success_rates,
                        'avg_success': avg_success,
                        # 'train_loss': avg_train_loss,
                        'improved_tasks': improved_tasks,
                    }, save_path)
                    print(f"[info] Saved model at epoch {epoch} due to improvements in tasks {improved_tasks}")
                    print(f"[info] Task-wise improvements:")
                    for task_idx in improved_tasks:
                        print(f"    Task {task_idx}: {best_task_success_rates[task_idx]:.4f}")
        
        # Save model if we get better training loss
        # if avg_train_loss < best_train_loss:
        #     best_train_loss = avg_train_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_loss': best_train_loss,
        #         'task_success_rates': success_rates if epoch % 5 == 0 else None,
        #         'avg_success': avg_success if epoch % 5 == 0 else None,
        #     }, os.path.join(cfg.experiment_dir, 'best_model_train.pth'))
        #     print(f"[info] Saved new best model with training loss: {best_train_loss:.4f}")

    print("[info] Training completed")

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
