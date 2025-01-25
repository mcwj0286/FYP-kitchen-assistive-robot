import sys

sys.path.append('/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot')
# from models.networks.gpt import GPT, GPTConfig
from models.networks.mlp import MLP

from models.networks.rgb_modules import BaseEncoder, ResnetEncoder
import torch
import torch.nn as nn
import utils
from collections import deque
import torchvision.transforms as T
import einops
import numpy as np
from models.networks.policy_head import DeterministicHead
from models.networks.transformer_modules import TransformerDecoder, SinusoidalPositionEncoding
import robomimic.utils.tensor_utils as TensorUtils

class bc_transformer_policy(nn.Module):
    def __init__(self, repr_dim=512, act_dim=7, hidden_dim=256,
                 policy_head="deterministic", obs_type='pixels',
                 obs_shape={
                    'pixels': (3, 128, 128),
                    'pixels_egocentric': (3, 128, 128),
                    'proprioceptive': (9,),
                    # 'features': (123,)
                }
                , language_dim=768, lang_repr_dim=512, language_fusion="film",
                 pixel_keys=['pixels', 'pixels_egocentric'],proprio_key='proprioceptive', device="cuda",
                 history=True, history_len=10, num_queries=10,
                 temporal_agg=True,
                 max_episode_len=200,
                 use_proprio=True,
                 num_feat_per_step=3):
        super().__init__()  # Call parent class constructor
        
        self.device = device
        self.language_dim = language_dim
        self.lang_repr_dim = lang_repr_dim
        self.language_fusion = language_fusion
        self.pixel_keys = pixel_keys if pixel_keys else []
        self.proprio_key = proprio_key
        self.repr_dim = repr_dim
        self._policy_head = policy_head
        self.history = history
        self.history_len = history_len if history else 1
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.use_proprio = use_proprio
        self.observation_buffer = {}
        # self.num_queries = num_queries
        self.act_dim = act_dim
        self.num_prompt_feats = num_feat_per_step
        self.step = 0

        if self.temporal_agg:
            self.num_queries = num_queries
        else: 
            self.num_queries = 1
 
        self.action_dim = (
            self.act_dim * self.num_queries
        )
        if obs_type == "pixels":
            if use_proprio:
                proprio_shape = obs_shape[self.proprio_key]
            obs_shape = obs_shape[self.pixel_keys[0]]
        # Initialize transformer decoder
        self.transformer = TransformerDecoder(
            input_size=repr_dim,
            num_layers=8,
            num_heads=4,
            head_output_size=64,
            mlp_hidden_size=hidden_dim,
            dropout=0.1
        )
        
        # Initialize temporal position encoding
        self.temporal_position_encoding = SinusoidalPositionEncoding(
            input_size=repr_dim,
            inv_freq_factor=10000
        )
        
        # Action head for final prediction
        if policy_head == "deterministic":
            self.action_head = DeterministicHead(
                self.repr_dim, self.action_dim, num_layers=2
            )

        # initialize the vision encoder
        self.vision_encoder = nn.ModuleDict()
        for key in self.pixel_keys:
            self.vision_encoder[key] = ResnetEncoder(
                obs_shape,
                512,
                language_dim=self.lang_repr_dim,
                language_fusion=self.language_fusion,
            )

        # Initialize language encoder with proper configuration
        self.language_projector = MLP(
            self.language_dim,
            hidden_channels=[self.lang_repr_dim, self.lang_repr_dim],
        )

        # projector for proprioceptive features
        self.proprio_projector = MLP(
            proprio_shape[0], hidden_channels=[self.repr_dim, self.repr_dim]
        )
        
        
        # augmentations
        # MEAN = torch.tensor([0.485, 0.456, 0.406])
        # STD = torch.tensor([0.229, 0.224, 0.225])
        # self.customAug = T.Compose([T.Normalize(mean=MEAN, std=STD)])
        # self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])
        
        self.latent_queue = []
        self.reset()

    def preprocess_input(self, data, train_mode=True):
        """
        Preprocess the input data for both training and inference
        """
        processed_data = {}
        processed_data["obs"] = {}
        
        # Process image observations
        for key in self.pixel_keys:
            if key in data:
                processed_data["obs"][key] = data[key].float()
                
        # Process proprioceptive data
        if self.use_proprio and self.proprio_key in data:
            processed_data["obs"][self.proprio_key] = data[self.proprio_key].float()
            
        # Process task embedding
        if "task_emb" in data:
            processed_data["task_emb"] = data["task_emb"].float()
            
        return processed_data

    def spatial_encode(self, data):
        """
        Encode spatial features from different modalities
        """
        encoded = []
        
        # 1. Process language embedding for FiLM
        lang_features = data["task_emb"].float()  # (B, 1, E)
        lang_features = self.language_projector(lang_features)  # (B, 1, lang_repr_dim)
        
        # Get batch and time dimensions from first image
        B, T = data["obs"][self.pixel_keys[0]].shape[:2]
        
        # Expand language features
        lang_token = lang_features.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (B, T, 1, E)
        encoded.append(lang_token)
        
        # 2. Process vision features
        for key in self.pixel_keys:
            pixel = data["obs"][key]  # (B, T, C, H, W)
            B, T, C, H, W = pixel.shape
            
            # Process each timestep
            pixel = pixel.reshape(B * T, C, H, W)
            lang = lang_features.repeat_interleave(T, dim=0) if self.language_fusion == "film" else None

            pixel = self.vision_encoder[key](
                pixel,
                lang=lang  # Reshape [B,E] -> [B*T,E]
            )
            pixel = pixel.view(B, T, 1, -1)
            encoded.append(pixel)
            
        # 3. Process proprioceptive features if used
        if self.use_proprio:
            proprio = data["obs"][self.proprio_key].float()
            proprio = self.proprio_projector(proprio)
            proprio = proprio.unsqueeze(2)  # Add modality dimension
            encoded.append(proprio)
            
        # Combine all features
        encoded = torch.cat(encoded, dim=2)  # (B, T, num_modalities, E)
        return encoded

    def temporal_encode(self, x):
        """
        Apply temporal encoding and transformer processing
        """
        # Add positional encoding
        pos_emb = self.temporal_position_encoding(x)  # (T, E)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        
        # Compute mask for transformer
        self.transformer.compute_mask(x.shape)
        
        # Reshape for transformer: (B, T, num_modalities, E) -> (B, T*num_modalities, E)
        sh = x.shape
        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Reshape back
        x = x.reshape(*sh)
        return x[:, :, -1]  # Return last modality features (B, T, E)

    def reset(self):
        """Reset history buffers"""
 
        self.latent_queue = []
        self.step = 0
        
        # temporal aggregation
        if self.temporal_agg:
            # Initialize with batch dimension of 1, will be resized as needed
            self.all_time_actions = torch.zeros(
                1,  # initial batch size
                self.max_episode_len,
                self.max_episode_len + self.num_queries,
                self.act_dim
            ).to(self.device)

    def forward(self, data, action=None):
        """
        Forward pass through the policy network
        
        Args:
            data (dict): Dictionary containing:
                - pixels: Agent view RGB images (B, T, C, H, W) 
                - pixels_egocentric: Eye-in-hand RGB images (B, T, C, H, W)
                - proprioceptive: Robot state features (B, T, D_proprio)
                - task_emb: Task embedding (B, E)
            action (torch.Tensor, optional): Ground truth actions for training (B, T, D_action)
                where D_action is typically 7 (position, rotation, gripper)
                
        Returns:
            pred_actions: Predicted action distribution
            loss_dict (optional): Dictionary of losses if action is provided
        """
        # Preprocess input data
        data = self.preprocess_input(data)
        
        # Encode spatial features
        x = self.spatial_encode(data)
        
        # Apply temporal encoding
        x = self.temporal_encode(x)
        
        # Get action predictions from action head
        pred_actions = self.action_head(x)

        return pred_actions

    def get_action(self, data):
        """ 
            Args:
            data (dict): Dictionary containing:
                - pixels: Agent view RGB images (B, 1, C, H, W) 
                - pixels_egocentric: Eye-in-hand RGB images (B, 1, C, H, W)
                - proprioceptive: Robot state features (B, 1, D_proprio)
                - task_emb: Task embedding (B, E)
        Get action for evaluation/inference"""
        self.eval()
        with torch.no_grad():
            # Preprocess input
            data = self.preprocess_input(data, train_mode=False)
            
            # Encode spatial features
            x = self.spatial_encode(data)
            
            # Manage latent queue for temporal history
            self.latent_queue.append(x)
            if len(self.latent_queue) > self.history_len:
                self.latent_queue.pop(0)
            x = torch.cat(self.latent_queue, dim=1)
            
            # Apply temporal encoding
            x = self.temporal_encode(x)
            
            # Get action prediction
            pred_actions = self.action_head(x[:, -1])
            pred_actions = pred_actions.mean
            
            # Process for temporal aggregation if needed
            if self.temporal_agg:
                B = pred_actions.shape[0]
                pred_actions = pred_actions.view(-1, self.num_queries, self.act_dim)
                
                # Resize all_time_actions if batch size changes
                if self.all_time_actions.shape[0] != B:
                    self.all_time_actions = torch.zeros(
                        B,
                        self.max_episode_len,
                        self.max_episode_len + self.num_queries,
                        self.act_dim
                    ).to(self.device)
                
                actions = []
                for i in range(B):
                    action = pred_actions[i]
                    self.all_time_actions[i, self.step, self.step : self.step + self.num_queries] = action
                    actions_for_curr_step = self.all_time_actions[i,:, self.step]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                    action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    actions.append(action)
                action = torch.cat(actions, dim=0)
                self.step += 1
                return action.detach().cpu().numpy()
            
            return pred_actions.detach().cpu().numpy()



if __name__ == "__main__":

    config = {
        'repr_dim': 512,
        'act_dim': 7,
        'hidden_dim': 256,
        'policy_head': 'deterministic',
        'obs_type': 'pixels',
        'obs_shape': {
            'pixels': (3, 128, 128),
            'pixels_egocentric': (3, 128, 128),
            'proprioceptive': (9,),
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Initialize model
    policy = bc_transformer_policy(**config).to(config['device'])
    policy.eval()  # Set to evaluation mode

    # Create dummy input data
    batch_size = 1
    time_steps = 10
    dummy_data = {
        'pixels': torch.randn(batch_size, time_steps, 3, 128, 128).to(config['device']),
        'pixels_egocentric': torch.randn(batch_size, time_steps, 3, 128, 128).to(config['device']),
        'proprioceptive': torch.randn(batch_size, time_steps, 9).to(config['device']),
        'task_emb': torch.randn(batch_size, 768).to(config['device']),  # Language embedding
        'step': 0  # For temporal aggregation
    }

    # Reset policy
    # policy.reset()

    # Get action
    try:
        action = policy.get_action(dummy_data)
        print(f"Predicted action shape: {action.shape}")
        print(f"Predicted action: {action}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
    
