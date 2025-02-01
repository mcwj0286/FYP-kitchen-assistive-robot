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
                 history=True, history_len=10, num_queries=10,
                 temporal_agg=True,
                 max_episode_len=200,
                 use_proprio=True,
                 num_feat_per_step=3,
                 learnable_tokens=True):
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
        self.transformer = ACT_TransformerDecoder(
            input_size=repr_dim,
            num_layers=8,
            num_heads=4,
            head_output_size=64,
            mlp_hidden_size=hidden_dim,
            dropout=0.1,
            num_queries=self.num_queries,
            learnable_tokens=False
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

if __name__ == "__main__":
    # Test CustomTransformerDecoder
    print("Testing CustomTransformerDecoder...")
    
    # Initialize decoder
    decoder = CustomTransformerDecoder(
        input_size=512,
        num_layers=8,
        num_heads=4,
        head_output_size=64,
        mlp_hidden_size=256,
        dropout=0.1,
        num_queries=10,
        learnable_tokens=True
    ).cuda()
    
    # Create random input tensor
    batch_size = 32
    num_features = 15  # Number of input features/tokens
    input_dim = 512
    
    x = torch.randn(batch_size, num_features, input_dim).cuda()
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = decoder(x)
            print(f"Output shape: {output.shape}")
            
            # Get attention weights
            attention_weights = decoder.get_attention_weights()
            
            # Print attention weights shapes
            print("\nAttention Weights Shapes:")
            print("Cross Attention:")
            for layer_idx, weights in enumerate(attention_weights['cross_attention']):
                print(f"Layer {layer_idx}: {weights.shape}")
            
            print("\nSelf Attention:")
            for layer_idx, weights in enumerate(attention_weights['self_attention']):
                print(f"Layer {layer_idx}: {weights.shape}")
                
            # Verify causal masking
            print("\nVerifying causal masking in self attention...")
            last_layer_self_attn = attention_weights['self_attention'][-1]
            upper_triangle = last_layer_self_attn[:, :, torch.triu_indices(10, 10, offset=1)[0], torch.triu_indices(10, 10, offset=1)[1]]
            is_masked = torch.all(upper_triangle == 0)
            print(f"Causal masking verified: {is_masked}")
            
            print("\nTest completed successfully!")
            
    except Exception as e:
        print(f"Error during test: {e}")

    # Test bc_act_policy
    print("\nTesting bc_act_policy...")
    
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
    policy = bc_act_policy(**config).to(config['device'])
    policy.eval()

    # Create dummy input data
    batch_size = 1
    time_steps = 10
    dummy_data = {
        'pixels': torch.randn(batch_size, time_steps, 3, 128, 128).to(config['device']),
        'pixels_egocentric': torch.randn(batch_size, time_steps, 3, 128, 128).to(config['device']),
        'proprioceptive': torch.randn(batch_size, time_steps, 9).to(config['device']),
        'task_emb': torch.randn(batch_size, 768).to(config['device']),
    }

    try:
        action = policy.get_action(dummy_data)
        print(f"Predicted action shape: {action.shape}")
        print(f"Predicted action: {action}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
    
