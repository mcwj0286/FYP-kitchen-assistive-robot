import sys

sys.path.append('/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic')
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

class ACT_TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers=8,
        num_heads=4,
        head_output_size=64,
        mlp_hidden_size=256,
        dropout=0.1,
        num_queries=10,
        learnable_tokens=True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_output_size = head_output_size
        self.dropout = dropout
        self.num_queries = num_queries
        
        # Action token embeddings - either learnable or fixed
        if learnable_tokens:
            self.action_tokens = nn.Parameter(torch.randn(1, num_queries, input_size))
        else:
            self.register_buffer('action_tokens', torch.randn(1, num_queries, input_size))
            
        # Positional embeddings for action tokens
        self.pos_encoding = SinusoidalPositionEncoding(input_size, inv_freq_factor=10000)
        
        # Create causal mask for self attention
        mask = torch.triu(torch.ones(num_queries, num_queries), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        
        # Layer components
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # Cross attention
                'cross_attention': nn.MultiheadAttention(
                    embed_dim=input_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'cross_norm1': nn.LayerNorm(input_size),
                'cross_norm2': nn.LayerNorm(input_size),
                'cross_mlp': nn.Sequential(
                    nn.Linear(input_size, mlp_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_size, input_size),
                    nn.Dropout(dropout)
                ),
                
                # Self attention
                'self_attention': nn.MultiheadAttention(
                    embed_dim=input_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'self_norm1': nn.LayerNorm(input_size),
                'self_norm2': nn.LayerNorm(input_size),
                'self_mlp': nn.Sequential(
                    nn.Linear(input_size, mlp_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_size, input_size),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings (B, num_features, input_size)
            
        Returns:
            output: Processed action tokens (B, num_queries, input_size)
        """
        batch_size = x.shape[0]
        
        # Expand action tokens to batch size and add positional encoding
        queries = self.action_tokens.expand(batch_size, -1, -1)
        pos_enc = self.pos_encoding(queries)
        queries = queries + pos_enc
        
        # Store attention weights for visualization if needed
        self.cross_attention_weights = []
        self.self_attention_weights = []
        
        # Process through layers
        for layer in self.layers:
            # Cross attention
            attended_queries = layer['cross_norm1'](queries)
            cross_out, cross_weights = layer['cross_attention'](
                query=attended_queries,
                key=x,
                value=x
            )
            self.cross_attention_weights.append(cross_weights)
            queries = queries + cross_out
            
            # Cross attention FFN
            cross_ff = layer['cross_norm2'](queries)
            cross_ff = layer['cross_mlp'](cross_ff)
            queries = queries + cross_ff
            
            # Self attention with causal mask
            attended_queries = layer['self_norm1'](queries)
            self_out, self_weights = layer['self_attention'](
                query=attended_queries,
                key=attended_queries,
                value=attended_queries,
                attn_mask=self.causal_mask,
                need_weights=True
            )
            self.self_attention_weights.append(self_weights)
            queries = queries + self_out
            
            # Self attention FFN
            self_ff = layer['self_norm2'](queries)
            self_ff = layer['self_mlp'](self_ff)
            queries = queries + self_ff
            
        return queries
    
    def get_attention_weights(self):
        """Return stored attention weights for visualization"""
        return {
            'cross_attention': self.cross_attention_weights,
            'self_attention': self.self_attention_weights
        }

class bc_act_policy(nn.Module):
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
                 num_queries=10,
                 max_episode_len=200,
                 use_proprio=True,
                 num_feat_per_step=3,
                 learnable_tokens=False):
        super().__init__()  # Call parent class constructor
        
        self.device = device
        self.language_dim = language_dim
        self.lang_repr_dim = lang_repr_dim
        self.language_fusion = language_fusion
        self.pixel_keys = pixel_keys if pixel_keys else []
        self.proprio_key = proprio_key
        self.repr_dim = repr_dim
        self._policy_head = policy_head
        self.max_episode_len = max_episode_len
        self.use_proprio = use_proprio
        self.observation_buffer = {}
        self.act_dim = act_dim
        self.num_prompt_feats = num_feat_per_step
        self.step = 0

        self.num_queries = num_queries
 
        self.action_dim = (
            self.act_dim * self.num_queries
        )
        if obs_type == "pixels":
            if use_proprio:
                proprio_shape = obs_shape[self.proprio_key]
            obs_shape = obs_shape[self.pixel_keys[0]]
        # Initialize transformer decoder
        self.transformer = ACT_TransformerDecoder(
            input_size=repr_dim,
            num_layers=8,
            num_heads=4,
            head_output_size=64,
            mlp_hidden_size=hidden_dim,
            dropout=0.1,
            num_queries=self.num_queries,
            learnable_tokens=learnable_tokens
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
       
        This function encodes spatial features from different modalities and arranges them into tokens.
        The output tensor has shape (B, T, num_modalities, E) where:
        - B: Batch size
        - T: Number of timesteps
        - num_modalities: Number of different input modalities concatenated together:
            1. Language token (1 token)
            2. Vision tokens (Total 2 token ,1 token per vision input type - pixels, pixels_egocentric)
            3. Proprioceptive token (1 token if proprioceptive features used)
        - E: Embedding dimension (self.repr_dim)

        The tokens are arranged in the following order:
        1. Language token from task embedding
        2. Vision tokens from each image input
        3. Proprioceptive token (if used)

        Each modality's features are projected to the same embedding dimension E
        before being concatenated along the modality dimension.
      
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
        x: (B, T, num_modalities, E)
        """
        B,T,num_modalities,E = x.shape
        outputs = []
        for i in range(T):
            input = x[:,i,:,:] # (B, num_modalities, E)
            output = self.transformer(input) # (B, num_queries, E)

            # Unsqueeze output to [B,1,num_queries,E] and store
            output = output.unsqueeze(1)  # [B,1,num_queries,E]
            
            
            outputs.append(output)
        
        # Stack all outputs along time dimension
        x = torch.cat(outputs, dim=1)  # [B,T,num_queries,E]
        
        
        return x

    def reset(self):
        """Reset history buffers"""
 
        self.latent_queue = []
        self.step = 0
        
        
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
        B,T,num_queries,E = x.shape
        # reshape x to [B,T*num_queries,E]
        x = x.reshape(B, -1, E)

        # Get action predictions from action head
        pred_actions = self.action_head(x) # (B, T*num_queries, act_dim)
        # reshape pred_actions to [B,T,num_queries,act_dim]
        pred_actions = pred_actions.reshape(B, T, num_queries, self.act_dim)
        # reshape pred_actions to [B,T,act_dim] for training loss
        pred_actions = pred_actions.reshape(B, T, self.action_dim)

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
            x = self.spatial_encode(data) # (B, 1, num_modalities, E)
            
            # Apply temporal encoding
            x = self.temporal_encode(x) # (B, 1, num_queries, E)
            
            x = x.squeeze(1) # (B, num_queries, E)

            # Get action prediction
            pred_actions = self.action_head(x) # (B, num_queries, act_dim)
            
            pred_actions= pred_actions.mean
            
            
            B = pred_actions.shape[0]
            
            
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
            
            

    # NEW: train_step method for bc_act_policy
    def train_step(self, data):
        """
        Performs a training step for BC-ACT.
        Expects data to contain ground truth actions under the key "actions".
        Returns:
            loss (torch.Tensor): the negative log likelihood loss.
        """
        gt_actions = data.get("actions", None)
        if gt_actions is None:
            raise ValueError("Ground truth actions missing in training batch")
        # Forward pass returns an output distribution.
        pred_dist = self.forward(data)
        # Compute the negative log likelihood loss from the output distribution.
        loss = -pred_dist.log_prob(gt_actions).mean()
        return loss

