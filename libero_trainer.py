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
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

@hydra.main(config_path="sim_env/LIBERO/libero/configs", config_name="config")
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # Add BAKU-specific configurations
    if not hasattr(cfg, 'policy'):
        cfg.policy = EasyDict()
        
    # Set BAKU policy configuration 
    cfg.policy.update({
        'policy_type': 'bcbakupolicy',  # Changed to match registered name (lowercase)
        'obs_type': 'pixels',
        'encoder_type': 'resnet',
        'policy_head': 'deterministic',
        'hidden_dim': 256,
        'language_fusion': 'film',
        'use_proprio': True,
        'temporal_agg': True,
        'language_encoder': EasyDict({
            "network": "MLP",
            "network_kwargs": {
                "input_size": 768,
                "output_size": 512,
                "hidden_channels": [512, 512]
            }
        })
    })

    # print configs
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning paths
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    # get benchmark and number of tasks
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
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

    # prepare experiment and update config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    # Initialize result summary
    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "L_fwd": np.zeros((n_manip_tasks,)),
        "S_fwd": np.zeros((n_manip_tasks,)),
    }

    # define multitask algorithm
    algo = safe_device(get_algo_class("Multitask")(n_manip_tasks, cfg), cfg.device)
    
    # Load pretrained model if specified
    if cfg.pretrain_model_path:
        try:
            algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0])
        except:
            print(f"[error] cannot load pretrained model from {cfg.pretrain_model_path}")
            return

    print(f"[info] start multitask learning")
    
    # Combine all datasets for multitask training
    concat_dataset = ConcatDataset(datasets)
    
    # Create data loader
    train_dataloader = DataLoader(
        concat_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,  # Set to 0 to avoid deadlock
        sampler=RandomSampler(concat_dataset),
    )

    # Training state tracking
    prev_success_rate = -1.0
    best_state_dict = algo.policy.state_dict()
    cumulated_counter = 0.0
    idx_at_best_succ = 0
    successes = []
    losses = []
    
    # Model checkpoint paths
    model_checkpoint_name = os.path.join(cfg.experiment_dir, f"multitask_model.pth")
    print(f'experiment_dir: {cfg.experiment_dir}')
    print("[info] Starting multitask training...")
    
    # Training loop
    for epoch in range(0, cfg.train.n_epochs + 1):
        t0 = time.time()
        
        # Training or zero-shot evaluation
        if epoch > 0 or cfg.pretrain:
            algo.policy.train()
            training_loss = 0.0
            for data in train_dataloader:
                loss = algo.observe(data)
                training_loss += loss
            training_loss /= len(train_dataloader)
        else:
            training_loss = 0.0
            for data in train_dataloader:
                loss = algo.eval_observe(data)
                training_loss += loss
            training_loss /= len(train_dataloader)
            
        t1 = time.time()
        print(f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}")

        # Evaluate every epoch
        t0 = time.time()
        algo.policy.eval()
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            model_checkpoint_name_ep = os.path.join(cfg.experiment_dir, f"multitask_model_ep{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': algo.policy.state_dict(),
                'loss': training_loss,
                'cfg': cfg
            }, model_checkpoint_name_ep)
        
        losses.append(training_loss)

        # Evaluate success
        success_rates = evaluate_success(
            cfg=cfg,
            algo=algo,
            benchmark=benchmark,
            task_ids=list(range(n_manip_tasks)),
            result_summary=None
        )
        success_rate = np.mean(success_rates)
        successes.append(success_rate)

        # Save best model if there's improvement
        if prev_success_rate < success_rate and not cfg.pretrain:
            torch.save({
                'epoch': epoch,
                'model_state_dict': algo.policy.state_dict(),
                'loss': training_loss,
                'success_rate': success_rate,
                'cfg': cfg
            }, model_checkpoint_name)
            best_checkpoint_name = os.path.join(cfg.experiment_dir, f"multitask_model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': algo.policy.state_dict(),
                'loss': training_loss,
                'success_rate': success_rate,
                'cfg': cfg
            }, best_checkpoint_name)
            prev_success_rate = success_rate
            idx_at_best_succ = len(losses) - 1
            print(f"[info] New best model saved with success rate: {success_rate:4.2f}")

        t1 = time.time()

        # Update metrics
        cumulated_counter += 1.0
        ci = confidence_interval(success_rate, cfg.eval.n_eval)
        tmp_successes = np.array(successes)
        tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

        print(
            f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | "
            f"best succ: {prev_success_rate:4.2f} | "
            f"succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | "
            f"time: {(t1-t0)/60:4.2f}",
            flush=True,
        )

        # Step learning rate scheduler
        if algo.scheduler is not None and epoch > 0:
            algo.scheduler.step()

    # Load best model for final evaluation
    best_model_state = torch.load(os.path.join(cfg.experiment_dir, "multitask_model_best.pth"))['model_state_dict']
    algo.policy.load_state_dict(best_model_state)

    # Save training curves
    auc_checkpoint_name = os.path.join(cfg.experiment_dir, f"multitask_auc.log")
    torch.save({
        "success": np.array(successes),
        "loss": np.array(losses),
    }, auc_checkpoint_name)

    # Final evaluation
    if cfg.eval.eval:
        print("[info] Running final evaluation...")
        L = evaluate_loss(cfg, algo, benchmark, datasets)
        S = evaluate_success(
            cfg=cfg,
            algo=algo,
            benchmark=benchmark,
            task_ids=list(range(n_manip_tasks)),
            result_summary=result_summary if cfg.eval.save_sim_states else None,
        )

        result_summary["L_conf_mat"][-1] = L
        result_summary["S_conf_mat"][-1] = S

        print(("[All task loss ] " + " %4.2f |" * n_manip_tasks) % tuple(L))
        print(("[All task succ.] " + " %4.2f |" * n_manip_tasks) % tuple(S))

        # Save results
        torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))

    print("[info] Training completed")

def confidence_interval(success_rate, n_eval, z=1.96):
    """Calculate confidence interval for success rate"""
    if n_eval == 0:
        return 0
    return z * np.sqrt(success_rate * (1 - success_rate) / n_eval)

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    main()
