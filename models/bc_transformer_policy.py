import sys

sys.path.append('/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot')
# from models/networks/gpt import GPT, GPTConfig
from models.networks.mlp import MLP

from models.networks.rgb_modules import BaseEncoder, ResnetEncoder
import torch
import torch.nn as nn
import utils
from collections import deque
import torchvision.transforms as T
import einops
import numpy as np
from models.networks.policy_head import DeterministicHead, MultiTokenDeterministicHead
from models.networks.transformer_modules import TransformerDecoder, SinusoidalPositionEncoding
import robomimic.utils.tensor_utils as TensorUtils


# New: MPIVisionEncoder definition
class MPIVisionEncoder(nn.Module):
    def __init__(self, mpi_root_dir, device, output_dim=512):
        super().__init__()
        import sys
        sys.path.append('/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/models/networks/vison_encoder/MPI')
        from models.networks.vison_encoder.MPI.mpi import load_mpi
        # Load the MPI model; freeze=True ensures weights are fixed
        self.mpi_model = load_mpi(mpi_root_dir, device, freeze=True)
        # Projection layer: assume MPI output dimension is 384 (from example), project to output_dim
        self.proj = nn.Linear(384, output_dim)
        self.device = device

    def forward(self, x, lang=None):
        # x is expected to be (N, C, H, W)
        # Duplicate the input along a new dimension to match expected shape: (N, 2, C, H, W)
        # Create transform to resize input to 224x224
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224)
        ])
        
        # x is (N, C, 128, 128)
        N, C, H, W = x.shape
        
        # Reshape to (N*C, H, W) for transform
        x= x.reshape(-1, H, W)
        
        # Apply transform and reshape back
        x = transform(x.unsqueeze(1))  # Add channel dim for transform
        x = x.reshape(N, C, 224, 224)  # Back to (N, C, 224, 224)
        x_dual = torch.stack((x, x), dim=1)
        with torch.no_grad():
            # Get visual representations without language tokens
            x = self.mpi_model.get_representations(x_dual, None, with_lang_tokens=False) # (1, 197, 384)
        # repr shape assumed to be (N, T, 384), aggregate over token dimension
        # x = torch.mean(x, dim=1).unsqueeze(1) # (N, 1, 384)

        # take aggregated token
        x = x[:,-1,:].unsqueeze(1) # (N, 1, 384)
        out = self.proj(x)  # (N, 1, output_dim)
        return out


class bc_transformer_policy(nn.Module):
    def __init__(self, repr_dim=512, act_dim=7, hidden_dim=256,
                 policy_head="deterministic", obs_type='pixels',
                 obs_shape={
                    'pixels': (3, 128, 128),
                    'pixels_egocentric': (3, 128, 128),
                    'proprioceptive': (9,),
                    # 'features': (123,)
                },
                language_dim=768, lang_repr_dim=512, language_fusion="film",
                 pixel_keys=['pixels', 'pixels_egocentric'], proprio_key='proprioceptive', device="cuda",
                 history=True, history_len=10, num_queries=10,
                 temporal_agg=True,
                 max_episode_len=200,
                 use_proprio=True,
                 num_feat_per_step=3,
                 use_mpi_pixels_egocentric=True,  # New flag
                 mpi_root_dir="/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/models/networks/vison_encoder/MPI/mpi/checkpoints"
                 ):
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
        elif policy_head == "mtdh":
            self.action_head = MultiTokenDeterministicHead(
                self.repr_dim, self.act_dim, num_tokens=self.num_queries, num_layers=2 , hidden_size=1024
            )
        else:
            raise ValueError(f"Invalid policy head: {policy_head}")

        # initialize the vision encoder
        self.vision_encoder = nn.ModuleDict()
        for key in self.pixel_keys:
            if key == "pixels_egocentric" and use_mpi_pixels_egocentric:
                self.vision_encoder[key] = MPIVisionEncoder(mpi_root_dir, device, output_dim=repr_dim)
            else:
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
        num_params = self.count_parameters()
        print(f"Total trainable parameters: {num_params:,}")
    def count_parameters(self):
        """
        Count total trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



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
        
        
        # 2. Process vision features
        for key in self.pixel_keys:
            pixel = data["obs"][key]  # (B, T, C, H, W)
            B, T, C, H, W = pixel.shape
            
            # Process each timestep
            pixel = pixel.reshape(B * T, C, H, W)
            if key == "pixels_egocentric" and isinstance(self.vision_encoder[key], MPIVisionEncoder):
                # For MPI encoder, handle its output shape (B*T, 197, repr_dim)
                pixel = self.vision_encoder[key](pixel, lang=None)
                
                _,tk,_ = pixel.shape
                pixel = pixel.reshape(B,T,tk,-1)
                
                
            else:
                # For ResNet encoder, process as before
                lang = lang_features.repeat_interleave(T, dim=0) if self.language_fusion == "film" else None
                pixel = self.vision_encoder[key](pixel, lang=lang)
                pixel = pixel.view(B, T, 1, -1)
            encoded.append(pixel)
        
        encoded.append(lang_token)
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
        Get action for evaluation/inference
        """
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
            
            # Get action prediction based on policy head type
            if self._policy_head == "deterministic":
                pred_dist = self.action_head(x[:, -1])
                pred_actions = pred_dist.mean
            elif self._policy_head == "mtdh":
                pred_dists = self.action_head(x[:, -1])
                # For MTDH, concatenate all token predictions
                pred_actions = torch.cat([dist.mean for dist in pred_dists], dim=-1)
            else:
                raise ValueError(f"Unsupported policy head: {self._policy_head}")
            
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
        
    def train_step(self, data, optimizer, scheduler=None):
        """
        Performs a training step for BC Transformer Policy.
        Expects data to contain ground truth actions under the key "actions".
        Args:
            data: Dictionary containing training data
            optimizer: Optimizer to use for parameter updates
            scheduler: Learning rate scheduler (optional)
        Returns:
            loss (torch.Tensor): the negative log likelihood loss.
        """
        gt_actions = data.get("actions", None)
        if gt_actions is None:
            raise ValueError("Ground truth actions missing in training batch")
            
        
        optimizer.zero_grad()
            
        if self._policy_head == "deterministic":
            pred_dist = self.forward(data)
            # loss = -pred_dist.log_prob(gt_actions).mean()
            log_probs = pred_dist.log_prob(gt_actions)
            loss = -log_probs
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        elif self._policy_head == "mtdh":
            # Reshape ground truth actions from [B, T, act_dim * num_queries] to [B, T, num_queries, act_dim]
            gt_actions = einops.rearrange(gt_actions, 'b t (q a) -> b t q a', q=self.num_queries, a=self.act_dim)
            # Split ground truth actions into list of [B, T, act_dim] tensors
            gt_actions_list = [gt_actions[..., i, :] for i in range(self.num_queries)]
            data = self.preprocess_input(data)
            # Encode spatial features
            x = self.spatial_encode(data)
            # Apply temporal encoding
            z = self.temporal_encode(x)
            pred_actions_probs = self.action_head(z)
            loss = self.action_head.multi_token_loss(pred_actions_probs, gt_actions_list)
            loss.backward()
            optimizer.step()
            
        return loss


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
    policy = bc_transformer_policy(**config, use_mpi_pixels_egocentric=True).to(config['device'])
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
    
