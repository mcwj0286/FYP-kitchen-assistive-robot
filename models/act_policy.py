"""Action Chunking Transformer (ACT) Policy Implementation.

This module implements a wrapper around the original ACT implementation with training and inference methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import torchvision.transforms as transforms
# from networks.utils.act.policy import ACTPolicy as OriginalACTPolicy
from .networks.utils.act.detr.models.detr_vae import build as build_ACT_model

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class ACTPolicy(nn.Module):
    """ACT Policy wrapper with training and inference methods.
    
    Implements the training and inference algorithms from the ACT paper:
    "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
    """
    
    def __init__(self, 
                 cfg,
                 kl_weight=10,
                 lang_dim=768,
                 device='cuda'):
        """Initialize ACT Policy.
        
        Args:
            cfg: Configuration object containing:
                Required for backbone:
                    - hidden_dim (int): Hidden dimension size
                    - position_embedding (str): Type of position embedding ('sine' or 'learned')
                    - backbone (str): Type of backbone ('resnet18', 'resnet34', etc.)
                    - dilation (bool): Whether to use dilated convolutions
                    
                Required for transformer:
                    - nheads (int): Number of attention heads
                    - num_encoder_layers (int): Number of encoder layers
                    - num_decoder_layers (int): Number of decoder layers
                    - dim_feedforward (int): Dimension of feedforward network
                    - dropout (float): Dropout rate
                    - pre_norm (bool): Whether to use pre-normalization
                    
                Required for model:
                    - state_dim (int): Dimension of robot state
                    - action_dim (int): Dimension of action space
                    - num_queries (int): Number of action queries (chunk size)
                    - camera_names (list): List of camera names
                    - multitask (bool): Whether using multitask learning
                    - obs_type (str): Type of observation ('pixels' or 'features')
                    - max_episode_len (int): Maximum episode length
                    
                Required for training:
                    - lr (float): Learning rate
                    - lr_backbone (float): Learning rate for backbone
                    - weight_decay (float): Weight decay
                    
            kl_weight (float): Weight for KL divergence loss (β in paper)
            device (str): Device to run model on
            inference_weight (float): Weight for exponential averaging during inference
        """
        super(ACTPolicy, self).__init__()  # Initialize the parent nn.Module
        
        self.device = torch.device(device)
        self.num_queries = cfg.num_queries
        self.kl_weight = kl_weight
       
        self.max_episode_len = cfg.max_episode_len
        self.act_dim = cfg.action_dim
        self.temporal_agg = True # Default to True if not specified
        
        # Initialize the ACT model
        self.model = build_ACT_model(cfg).to(self.device)
        self.lang_proj = nn.Linear(lang_dim, cfg.hidden_dim).to(self.device)
        
        # Image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # Setup optimizer with all parameters including lang_proj
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr": cfg.lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.lr_backbone,
            },
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, weight_decay=cfg.weight_decay)
        
        # Remove image normalization if data is already normalized
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225]
        # )
        self.reset()
    
    def reset(self):
        """Reset history buffers"""
        self.all_time_actions = torch.zeros(
            1,  # initial batch size
            self.max_episode_len,
            self.max_episode_len + self.num_queries,
            self.act_dim
        ).to(self.device)
        self.step = 0
        
    def forward(self, data ,is_training=True):
        """Forward pass through the policy network"""
        # Reshape proprioceptive data from (B,1,D) to (B,D)
        if data["proprioceptive"].dim() == 3:
            data["proprioceptive"] = data["proprioceptive"].squeeze(1)
            
        # # Check temporal dimension of input images
        # if "pixels" in data:
        #     if data["pixels"].shape[1] == 1:
        #         raise ValueError("Expected pixels temporal dimension > 1, got shape: " + str(data["pixels"].shape))
                
        # if "pixels_egocentric" in data:
        #     if data["pixels_egocentric"].shape[1] == 1:
        #         raise ValueError("Expected pixels_egocentric temporal dimension > 1, got shape: " + str(data["pixels_egocentric"].shape))

        # Normalize images and ensure consistent dimensions
        if "pixels" in data:
            B, T, C, H, W = data["pixels"].shape
            data["pixels"] = self.normalize(data["pixels"].reshape(-1, C, H, W)).reshape(B, T, C, H, W)
            
        if "pixels_egocentric" in data:
            images = torch.cat([data["pixels"], data["pixels_egocentric"]], dim=1)
        else:
            images = data["pixels"]
            
        # Move task embedding to correct device and reshape
        if data["task_emb"].dim() == 3:
            task_emb = data["task_emb"].to(self.device)
            task_emb = self.lang_proj(task_emb)
            task_emb = task_emb.squeeze(1)  # Remove temporal dimension if present
        else:
            task_emb = data["task_emb"].to(self.device)
            task_emb = task_emb.unsqueeze(1)
            task_emb = self.lang_proj(task_emb)
            task_emb = task_emb.squeeze(1)

        # Process inputs through model
        if is_training:
            # Training mode
            data["action_padding_mask"] = data["action_padding_mask"].to(self.device).squeeze(1)
            data["actions"] = data["actions"].to(self.device)
            data["proprioceptive"] = data["proprioceptive"].to(self.device)
            images = images.to(self.device)
            
            # Ensure actions and padding mask have same sequence length
            if data["actions"].shape[1] != data["action_padding_mask"].shape[1]:
                raise ValueError(f"Actions shape {data['actions'].shape} and padding mask shape {data['action_padding_mask'].shape} must have same sequence length")
                
            pred_actions, is_pad_hat, (mu, logvar) = self.model(
                qpos=data["proprioceptive"],  # [B, state_dim]
                image=images,  # [B, num_cameras, C, H, W]
                env_state=None,
                actions=data["actions"],  # [B, seq_len, action_dim]
                is_pad=data["action_padding_mask"],  # [B, seq_len]
                task_emb=task_emb  # [B, hidden_dim]
            )
            
            # Compute losses
            total_kld, _, _ = kl_divergence(mu, logvar)
            
            loss_dict = {}
            all_l1 = F.l1_loss(data["actions"], pred_actions, reduction='none')
            l1_loss = all_l1.mean()
            
            loss_dict["l1"] = l1_loss
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + self.kl_weight * loss_dict["kl"]
            
            return loss_dict
            
        else:
            # Inference mode
            data["proprioceptive"] = data["proprioceptive"].to(self.device)
            images = images.to(self.device)
            
            pred_actions, _, _ = self.model(
                qpos=data["proprioceptive"],
                image=images,
                env_state=None,
                actions=None,
                is_pad=None,
                task_emb=task_emb
            )
            return pred_actions
    
    def get_action(self, data):
        """ 
        Args:
            data (dict): Dictionary containing:
                - pixels: Agent view RGB images (B, 1, C, H, W) 
                - pixels_egocentric: Eye-in-hand RGB images (B, 1, C, H, W)
                - proprioceptive: Robot state features (B, 1, D_proprio)
                - task_emb: Task embedding (B,1, E)
        Get action for evaluation/inference
        """
        pred_actions = self.forward(data, is_training=False)  # [B, seq_len, act_dim]
            
        # action chunking with batch dimension
        if self.temporal_agg:
            B = pred_actions.shape[0]  # batch dimension
            # pred_actions = pred_actions[:, -1]
            
            # Resize all_time_actions if batch size changes
            if self.all_time_actions.shape[0] != B:
                self.all_time_actions = torch.zeros(
                    B,  # batch dimension
                    self.max_episode_len,
                    self.max_episode_len + self.num_queries,
                    self.act_dim
                ).to(self.device)
            
            actions = []
            for i in range(B):
                action = pred_actions[i]
                action = action.view(-1, self.num_queries, self.act_dim)
                self.all_time_actions[i, self.step, self.step : self.step + self.num_queries] = action[-1:]
                actions_for_curr_step = self.all_time_actions[i,:, self.step]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                actions.append(action)
            action = torch.cat(actions, dim=0)  # [B,7]
            self.step += 1         
            
            return action.detach().cpu().numpy()
        
        return pred_actions[:, -1].detach().cpu().numpy()
    
    
    

if __name__ == "__main__":
    # Example config
    from easydict import EasyDict
    
    cfg = EasyDict(
        # Backbone config
        hidden_dim=512,
        position_embedding='sine',
        backbone='resnet18',
        dilation=False,
        masks=False,  # Required by backbone.py for intermediate layers
        return_interm_layers=False,  # Whether to return intermediate layers
        
        # Transformer config
        nheads=8,
        # num_encoder_layers=4,
        # num_decoder_layers=7,
        dim_feedforward=256,
        dropout=0.1,
        pre_norm=False,
        enc_layers=4,  # Required by transformer
        dec_layers=7,  # Required by transformer
        
        # Model config
        state_dim=14,
        action_dim=7,  # [x, y, z, rx, ry, rz, gripper]
        num_queries=10,  # chunk size
        camera_names=['pixels', 'pixels_egocentric'],
        multitask=True,
        obs_type='pixels',
        max_episode_len=1000,
        temporal_agg=True,
        
        # Training config
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        
        # Additional required params from detr_vae.py
        num_feature_levels=1,
        num_pos_feats=256,  # hidden_dim//2
        # temperature=10000.0,
        # normalize=True,
        # scale=None,
        # aux_loss=False,
        # with_box_refine=False,
        # two_stage=False,
        # activation='relu',
        # batch_first=True
    )
    
    # Initialize policy
    policy = ACTPolicy(
        cfg=cfg,
        kl_weight=0.1,
        device='cuda',
       
    )
    
    # Print detailed parameter counts
    def count_parameters(model):
        total_params = 0
        param_counts = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                
                # Group parameters by component
                component = name.split('.')[0]
                if component not in param_counts:
                    param_counts[component] = 0
                param_counts[component] += num_params
                
        #         # Print individual layer details
        #         print(f"{name}: {param.shape}, params: {num_params:,}")
        
        # Print summary by component
        print("\nParameters by component:")
        for component, count in param_counts.items():
            print(f"{component}: {count:,} parameters ({count/total_params*100:.2f}%)")
            
        print(f"\nTotal trainable parameters: {total_params:,}")
        return total_params
    
    print("\nDetailed model architecture:")
    total_params = count_parameters(policy.model)
    
    # Example training data with correct shapes
    batch_size = 2
    seq_len = cfg.num_queries  # Match num_queries
    data = {
        'pixels': torch.randn(batch_size, seq_len, 3, 224, 224).to('cuda'),  # [B, T, C, H, W]
        'pixels_egocentric': torch.randn(batch_size, seq_len, 3, 224, 224).to('cuda'),  # [B, T, C, H, W]
        'proprioceptive': torch.randn(batch_size, cfg.state_dim).to('cuda'),  # [B, state_dim]
        'task_emb': torch.randn(batch_size, cfg.hidden_dim).to('cuda'),  # [B, hidden_dim]
        'pad_mask': torch.ones(batch_size, cfg.num_queries).bool().to('cuda'),  # [B, num_queries]
        'actions': torch.randn(batch_size, cfg.num_queries, cfg.action_dim).to('cuda')  # [B, num_queries, action_dim]
    }
    
    # Test training with backpropagation
    print("\nTesting training with backpropagation...")
    
    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        # Zero gradients
        policy.optimizer.zero_grad()
        
        # Forward pass
        loss_dict = policy.forward(data)
        print(f"Epoch {epoch + 1} Losses:", loss_dict)
        
        # Backward pass
        total_loss = loss_dict['loss']
        total_loss.backward()
        
        # Print gradients for debugging
        # for name, param in policy.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient norm for {name}: {param.grad.norm().item()}")
        
        # Update parameters
        policy.optimizer.step()
    
    # Remove actions for inference test
    inference_data = {k:v for k,v in data.items() if k != 'actions' or k != 'pad_mask'}
    
    # Test inference
    print("\nTesting inference...")
    pred_actions = policy.get_action(inference_data)
    print("Predicted action shape:", pred_actions.shape)
    
    print("\nTest completed successfully!")
