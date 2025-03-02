#!/usr/bin/env python3
import os
import time
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from dotenv import load_dotenv
load_dotenv()
# Import the real-world demonstration dataset
from dataset import Kinova_Dataset

# Import models for behavior cloning.
# Here we support two options: "bc_act" and "bc_transformer".
# from models.bc_act_policy import bc_act_policy
# from models.bc_transformer_policy import bc_transformer_policy
# bc_transformer_policy is imported later if needed.

def parse_args():
    parser = argparse.ArgumentParser(description="Train a robot manipulation model on real-world demonstration (Kinova) data")
    # parser.add_argument("--data_path", type=str, required=True,
    #                     help="Directory containing the real demonstration HDF5 files")
    parser.add_argument("--model_type", type=str, default="bc_transformer", choices=["bc_act", "bc_transformer"],
                        help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="kinova_experiments",
                        help="Directory to save training checkpoints")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader worker processes")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of data to use for training vs validation")
    return parser.parse_args()

def create_experiment_dir(base_dir, model_type):
    """
    Create a unique experiment directory with timestamp.
    
    Parameters:
        base_dir (str): Base directory for all experiments
        model_type (str): Type of model being trained
    
    Returns:
        str: Path to the created experiment directory
    """
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create experiment directory name
    exp_dir_name = f"{model_type}_{timestamp}"
    # Create full path
    exp_dir = os.path.join(base_dir, exp_dir_name)
    # Create directory
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def create_datasets(data_path, train_ratio, **dataset_kwargs):
    """
    Create training and validation datasets.
    
    Parameters:
        data_path (str): Path to the dataset
        train_ratio (float): Proportion of data to use for training
        **dataset_kwargs: Additional arguments to pass to the dataset constructor
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Create training dataset
    train_dataset = Kinova_Dataset(
        data_path=data_path,
        train_ratio=train_ratio,
        is_train=True,
        **dataset_kwargs
    )
    
    # Create validation dataset
    val_dataset = Kinova_Dataset(
        data_path=data_path,
        train_ratio=train_ratio,
        is_train=False,
        **dataset_kwargs
    )
    
    return train_dataset, val_dataset

def validate(model, dataloader, device, logger,model_type):
    """
    Validate the model on the validation dataset.
    
    Parameters:
        model: The model to validate
        dataloader: DataLoader for the validation dataset
        device: Device to perform validation on
        logger: Logger for logging validation information
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_val_loss = 0.0
    mse_loss = nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch data to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            predicted_actions = model(batch)
            target_actions = batch['actions']
            
            if model_type == "bc_act":
                # Extract mean from distribution and reshape to match target shape
                predicted_actions = predicted_actions.mean
                B, S, an = target_actions.shape  # batch_size, seq_len, act_dim*num_queries
                predicted_actions = predicted_actions.reshape(B, S, an)
            else:
                predicted_actions = predicted_actions.mean
                
            # Calculate MSE loss
            loss = mse_loss(predicted_actions, target_actions)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(dataloader)
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting training on real-world demonstration data with model type: %s", args.model_type)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create experiment directory with timestamp
    exp_dir = create_experiment_dir(args.output_dir, args.model_type)
    logger.info("Created experiment directory: %s", exp_dir)

    # Create log file in experiment directory
    log_file = os.path.join(exp_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Save training arguments
    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Common dataset parameters
    dataset_kwargs = {
        "seq_length": 10,
        "frame_stack": 1,
        "overlap": 2,
        "pad_frame_stack": True,
        "pad_seq_length": True,
        "get_pad_mask": True,
        "get_action_padding": True,
        "num_queries": 10,
        "image_shape": (128, 128),
        "action_velocity_scale": 30.0,
        "gripper_scale": 3000.0,
        "load_task_emb": True,
        "max_word_len": 25
    }

    # Initialize train and validation datasets
    train_dataset, val_dataset = create_datasets(
        data_path=os.getenv('KINOVA_DATASET'),
        train_ratio=args.train_ratio,
        **dataset_kwargs
    )
    
    logger.info("Training dataset: %d segments", len(train_dataset))
    logger.info("Validation dataset: %d segments", len(val_dataset))
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # Initialize model based on model type.
    if args.model_type == "bc_act":
        from models.bc_act_policy import bc_act_policy
        # Define a simple configuration dictionary for bc_act_policy.
        config = {
            "obs_shape": {
                "pixels": (3, 128, 128),
                "pixels_egocentric": (3, 128, 128),
                "proprioceptive": (9,)
            },
            "act_dim": 7,
            "policy_head": "deterministic",
            "hidden_dim": 256,
            "device": device,
            "num_queries": 10,
            "max_episode_len": 650,
            "use_proprio": True,
            'n_layer': 4,
        }
        model = bc_act_policy(**config).to(device)
    elif args.model_type == "bc_transformer":
        from models.bc_transformer_policy import bc_transformer_policy
        
        model = bc_transformer_policy(
            history=False,
            max_episode_len=1000,
            use_mpi_pixels_egocentric=False,
            device=device,
        ).to(device)
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

    # Initialize tracking of best validation loss
    best_val_loss = float('inf')
    
    # Prepare for tracking losses
    train_losses = []
    val_losses = []

    # Begin training loop.
    logger.info("Starting training loop for %d epochs", args.epochs)
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to device.
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            # Assumes the model has a train_step(batch, optimizer=...) method that returns a loss tensor.
            loss = model.train_step(batch, optimizer=optimizer)
            
            total_loss += loss.item()

            # if (batch_idx + 1) % 10 == 0:
            #     logger.info("Epoch %d, Batch %d/%d, Loss: %.4f", epoch+1, batch_idx+1, len(train_loader), loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info("Epoch %d completed, Training Loss: %.4f", epoch+1, avg_train_loss)
        
        # Validation phase
        val_loss = validate(model, val_loader, device, logger,args.model_type)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(exp_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            logger.info("New best model saved with validation loss: %.4f", val_loss)
    
        # Save regular checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(exp_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

    # Save the final model in the experiment directory
    final_path = os.path.join(exp_dir, "model_final.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
    }, final_path)
    logger.info("Training completed. Final model saved to %s", final_path)

    # Clean up datasets to close HDF5 files
    train_dataset.close()
    val_dataset.close()

if __name__ == "__main__":
    main() 