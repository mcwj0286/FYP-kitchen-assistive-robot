#!/usr/bin/env python3
import os
import time
import logging
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from dotenv import load_dotenv
load_dotenv()
# Import the real-world demonstration dataset
from dataset import Kinova_Dataset

# Try to import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import models for behavior cloning.
# Here we support two options: "bc_act" and "bc_transformer".
# from models.bc_act_policy import bc_act_policy
# from models.bc_transformer_policy import bc_transformer_policy
# bc_transformer_policy is imported later if needed.

def parse_args():
    parser = argparse.ArgumentParser(description="Train a robot manipulation model on real-world demonstration (Kinova) data")
    # Model selection and configuration
    parser.add_argument("--model_type", type=str, default="bc_transformer", choices=["bc_act", "bc_transformer"],
                        help="Type of model to train")
    parser.add_argument("--n_layer", type=int, default=4,
                        help="Number of layers in the model")
    parser.add_argument("--repr_dim", type=int, default=512,
                        help="Representation dimension for the model")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for the model")
    parser.add_argument("--num_queries", type=int, default=10,
                        help="Number of action tokens/queries for models that use them")
    parser.add_argument("--policy_head", type=str, default="deterministic",
                        choices=["deterministic", "task_specific_head"],
                        help="Type of policy head to use")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of data to use for training vs validation")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Number of batches to accumulate gradients over")
    
    # Data processing parameters
    parser.add_argument("--seq_length", type=int, default=10,
                        help="Sequence length for temporal data")
    parser.add_argument("--frame_stack", type=int, default=1,
                        help="Number of frames to stack")
    parser.add_argument("--overlap", type=int, default=2,
                        help="Overlap between consecutive sequences")
    
    # Infrastructure parameters
    parser.add_argument("--output_dir", type=str, default="kinova_experiments",
                        help="Directory to save training checkpoints")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader worker processes")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model-specific options
    parser.add_argument("--use_mpi", action="store_true", default=False,
                        help="Use MPI vision encoder instead of ResNet18 (if available)")
    parser.add_argument("--use_moe", action="store_true", default=False,
                        help="Use Mixture of Experts (if supported by model)")
    # Device parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
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

def validate(model, dataloader, device, logger, model_type):
    """
    Validate the model on the validation dataset.
    
    Parameters:
        model: The model to validate
        dataloader: DataLoader for the validation dataset
        device: Device to perform validation on
        logger: Logger for logging validation information
        model_type: Type of model being validated
        
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

def set_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting training on real-world demonstration data with model type: %s", args.model_type)

    # Device
    device = args.device
    logger.info(f"Using device: {device}")

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

    # Initialize wandb if enabled
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_key = os.getenv('WANDB_API_KEY')
        if wandb_key:
            try:
                wandb.login(key=wandb_key)
                logger.info("Successfully logged in to Weights & Biases")
                wandb.init(
                    project="kinova-robot-training",
                    name=f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=vars(args)
                )
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
                args.use_wandb = False
        else:
            logger.warning("WANDB_API_KEY not found in environment, disabling wandb")
            args.use_wandb = False
    elif args.use_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb package not available, disabling wandb")
        args.use_wandb = False

    # Common dataset parameters
    dataset_kwargs = {
        "seq_length": args.seq_length,
        "frame_stack": args.frame_stack,
        "overlap": args.overlap,
        "pad_frame_stack": False,
        "pad_seq_length": False,
        "get_pad_mask": False,
        "get_action_padding": False,
        "num_queries": args.num_queries,
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

    # Initialize model based on model type with customized parameters
    if args.model_type == "bc_act":
        from models.bc_act_policy import bc_act_policy
        # Define a configuration dictionary for bc_act_policy with customizable parameters
        config = {
            "obs_shape": {
                "pixels": (3, 128, 128),
                "pixels_egocentric": (3, 128, 128),
                "proprioceptive": (9,)
            },
            "act_dim": 7,
            "policy_head": args.policy_head,
            "hidden_dim": args.hidden_dim,
            "repr_dim": args.repr_dim,
            "device": device,
            "num_queries": args.num_queries,
            "max_episode_len": 650,
            "use_proprio": True,
            'n_layer': args.n_layer,
            'use_moe': args.use_moe,
            'use_mpi': args.use_mpi,
        }
        model = bc_act_policy(**config).to(device)
    elif args.model_type == "bc_transformer":
        from models.bc_transformer_policy import bc_transformer_policy
        
        model = bc_transformer_policy(
            repr_dim=args.repr_dim,
            hidden_dim=args.hidden_dim,
            policy_head=args.policy_head,
            history=False,
            max_episode_len=1000,
            use_mpi_pixels_egocentric=args.use_mpi,
            device=device,
            num_queries=args.num_queries,
        ).to(device)
    else:
        raise ValueError("Unknown model type: {}".format(args.model_type))
    
    # Count and log trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_params/1e6:.2f}M")

    # Set up optimizer and scheduler
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
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

    # Begin training loop
    logger.info("Starting training loop for %d epochs", args.epochs)
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)

            # Forward pass and compute loss
            if args.accumulation_steps > 1:
                # Gradient accumulation for larger effective batch size
                optimizer.zero_grad()
                loss = model.train_step(batch, optimizer=None)  # Don't step optimizer here
                loss = loss / args.accumulation_steps  # Scale loss
                loss.backward()
                
                # Step optimizer after accumulation or at end of epoch
                if ((batch_idx + 1) % args.accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
            else:
                # Standard training step
                optimizer.zero_grad()
                loss = model.train_step(batch, optimizer=optimizer)
            
            total_loss += loss.item() * (args.accumulation_steps if args.accumulation_steps > 1 else 1)
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info("Epoch %d completed, Training Loss: %.4f", epoch+1, avg_train_loss)
        
        # Validation phase
        val_loss = validate(model, val_loader, device, logger, args.model_type)
        val_losses.append(val_loss)
        
        # Log to wandb if enabled
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
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
                'args': vars(args),  # Save configuration
            }, best_model_path)
            logger.info("New best model saved with validation loss: %.4f", val_loss)
            
            # Update wandb best model tracking
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
    
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
                'args': vars(args),  # Save configuration
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
        'args': vars(args),  # Save configuration
    }, final_path)
    logger.info("Training completed. Final model saved to %s", final_path)

    # Clean up datasets to close HDF5 files
    train_dataset.close()
    val_dataset.close()
    
    # Clean up wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main() 