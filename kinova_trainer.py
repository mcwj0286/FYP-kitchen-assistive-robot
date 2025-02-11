#!/usr/bin/env python3
import os
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

# Import the real-world demonstration dataset
from dataset import Kinova_Dataset

# Import models for behavior cloning.
# Here we support two options: "bc_act" and "bc_transformer".
from models.bc_act_policy import bc_act_policy
# bc_transformer_policy is imported later if needed.

def parse_args():
    parser = argparse.ArgumentParser(description="Train a robot manipulation model on real-world demonstration (Kinova) data")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing the real demonstration HDF5 files")
    parser.add_argument("--model_type", type=str, default="bc_act", choices=["bc_act", "bc_transformer"],
                        help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="kinova_experiments",
                        help="Directory to save training checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader worker processes")
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting training on real-world demonstration data with model type: %s", args.model_type)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create experiment directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the Kinova dataset.
    dataset = Kinova_Dataset(
        data_path=args.data_path,
        seq_length=10,
        frame_stack=4,
        overlap=2,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        get_action_padding=True,
        num_queries=10,
        image_shape=(128, 128),         # Target image size: 128x128
        action_velocity_scale=30.0,       # Joint velocities in [-30, 30]
        gripper_scale=3000.0,             # Gripper velocity in [-3000, 3000]
        load_task_emb=True,
        max_word_len=25
    )
    logger.info("Kinova_Dataset loaded %d segments from %d tasks", len(dataset), len(dataset.task_files))
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset),
        pin_memory=True,
        num_workers=args.num_workers
    )

    # Initialize model based on model type.
    if args.model_type == "bc_act":
        # Define a simple configuration dictionary for bc_act_policy.
        config = {
            "obs_shape": {
                "pixels": (3, 128, 128),
                "pixels_egocentric": (3, 128, 128),
                "proprioceptive": (9,)
            },
            "action_dim": 7,
            "policy_head": "deterministic",
            "hidden_dim": 256,
            "device": device,
            "history": True,
            "history_len": 10,
            "num_queries": 10,
            "temporal_agg": True,
            "max_episode_len": 650,
            "use_proprio": True,
        }
        model = bc_act_policy(**config).to(device)
    elif args.model_type == "bc_transformer":
        from models.bc_transformer_policy import bc_transformer_policy
        config = {  
            "obs_shape": {
                "pixels": (3, 128, 128),
                "pixels_egocentric": (3, 128, 128),
                "proprioceptive": (9,)
            },
            "action_dim": 7,
            "policy_head": "mtdh",
            "hidden_dim": 256,
            "device": device,
            "history": False,
            "history_len": 10,
            "num_queries": 10,
            "temporal_agg": True,
            "max_episode_len": 650,
            "use_proprio": True,
        }
        model = bc_transformer_policy(**config).to(device)
    else:
        raise ValueError("Unknown model type: {}".format(args.model_type))
    
    # Set up optimizer and scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # Begin training loop.
    logger.info("Starting training loop for %d epochs", args.epochs)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # Move batch data to device.
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            # Assumes the model has a train_step(batch, optimizer=...) method that returns a loss tensor.
            loss = model.train_step(batch, optimizer=optimizer)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # if (batch_idx + 1) % 10 == 0:
            #     logger.info("Epoch %d, Batch %d/%d, Loss: %.4f", epoch+1, batch_idx+1, len(dataloader), loss.item())
        avg_loss = total_loss / len(dataloader)
        logger.info("Epoch %d completed, Average Loss: %.4f", epoch+1, avg_loss)
        scheduler.step(avg_loss)
    
        # Optionally, save a checkpoint every 5 epochs.
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

    # Save the final model.
    final_path = os.path.join(args.output_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info("Training completed. Final model saved to %s", final_path)

    # (Validation and evaluation code can be added later as required)

if __name__ == "__main__":
    main() 