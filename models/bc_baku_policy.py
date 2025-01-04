import sys

sys.path.append('/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic')
from models.networks.gpt import GPT, GPTConfig
from models.networks.mlp import MLP
from models.networks.policy_head import (
    DeterministicHead,
    GMMHead,
    BeTHead,
    VQBeTHead,
    DiffusionHead,
)
from models.networks.rgb_modules import BaseEncoder, ResnetEncoder
import torch
import torch.nn as nn
import utils
from collections import deque
import torchvision.transforms as T
import einops
import numpy as np
class Obersvation_trunk(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        policy_head="deterministic",
        num_feat_per_step=1,
        device="cuda",
    ):
        super().__init__()

        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step
        self._policy_type = "gpt"  # added to enable GPT branch in forward

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        
        self._policy = GPT(
            GPTConfig(
                block_size=65,
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=hidden_dim,
                dropout=0.1,
            )
        )
        

        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "vqbet":
            self._action_head = VQBeTHead(
                hidden_dim,
                self._act_dim,
                hidden_size=hidden_dim,
                device=device,
            )
        

        self.apply(utils.weight_init)

    def forward(self, obs, num_prompt_feats, stddev, action=None, cluster_centers=None):
        B, T, D = obs.shape
        if self._policy_type == "mlp":
            if T * D < self._repr_dim:
                gt_num_time_steps = (
                    self._repr_dim // D - num_prompt_feats
                ) // self._num_feat_per_step
                num_repeat = (
                    gt_num_time_steps
                    - (T - num_prompt_feats) // self._num_feat_per_step
                )
                initial_obs = obs[
                    :, num_prompt_feats : num_prompt_feats + self._num_feat_per_step
                ]
                initial_obs = initial_obs.repeat(1, num_repeat, 1)
                obs = torch.cat(
                    [obs[:, :num_prompt_feats], initial_obs, obs[:, num_prompt_feats:]],
                    dim=1,
                )
                B, T, D = obs.shape
            obs = obs.view(B, 1, T * D)
            features = self._policy(obs)
        elif self._policy_type == "gpt":
            # insert action token at each self._num_feat_per_step interval
            prompt = obs[:, :num_prompt_feats]
            obs = obs[:, num_prompt_feats:]
            obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
            action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
            obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)
            obs = torch.cat([prompt, obs], dim=1)

            # get action features
            features = self._policy(obs)
            features = features[:, num_prompt_feats:]
            num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
            features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        # action head
        pred_actions = self._action_head(
            features,
            stddev,
            
        )
        if isinstance(pred_actions, torch.distributions.Distribution):
                pred_actions = pred_actions.mean
        elif isinstance(pred_actions, dict) and "predicted_action" in pred_actions:
                pred_actions = pred_actions["predicted_action"]
        
        return pred_actions




class BCBakuPolicy(nn.Module):
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
                 use_proprio=True):
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
        self.num_queries = num_queries
        self.act_dim = act_dim
        # number of inputs per time step
        if obs_type == "features":
            num_feat_per_step = 1
        elif obs_type == "pixels":
            num_feat_per_step = len(self.pixel_keys)
            if use_proprio:
                num_feat_per_step += 1

        # observation params
        if obs_type == "pixels":
            if use_proprio:
                proprio_shape = obs_shape[self.proprio_key]
            obs_shape = obs_shape[self.pixel_keys[0]]

        action_dim = (
            self.act_dim * self.num_queries if self.temporal_agg else self.act_dim
        )
        # observation trunk
        self.obs_trunk = Obersvation_trunk(
            repr_dim=repr_dim,
            act_dim=action_dim,
            hidden_dim=hidden_dim,
            policy_head=policy_head,
            num_feat_per_step=num_feat_per_step,
            device=device,
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
        
        self.reset()

    def reset(self):
        """Reset history buffers"""
        self.observation_buffer = {}
        for key in self.pixel_keys:
            self.observation_buffer[key] = deque(maxlen=self.history_len)
        if self.use_proprio:
            self.proprio_buffer = deque(maxlen=self.history_len)

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
        # 1. Process language embedding for FiLM
        lang_features = data["task_emb"].float()  # (B, E)
        lang_features = self.language_projector(lang_features)  # (B, lang_repr_dim)

        # 2. Process vision features
        features = []
        for key in self.pixel_keys:
            pixel = data[key].float()  # (B, T, C, H, W)
            shape = pixel.shape
            # rearrange to (B*T, C, H, W)
            pixel = einops.rearrange(pixel, "b t c h w -> (b t) c h w")
            
            # Repeat language features for each timestep
            lang = lang_features.repeat_interleave(shape[1], dim=0) if self.language_fusion == "film" else None
            
            # encode
            pixel = self.vision_encoder[key](pixel, lang=lang)
            # reshape back to (B, T, D)
            pixel = einops.rearrange(pixel, "(b t) d -> b t d", b=shape[0], t=shape[1])
            features.append(pixel)
            
        # 3. Process proprioceptive features if used
        if self.use_proprio:
            proprio = data["proprioceptive"].float()
            proprio = self.proprio_projector(proprio)
            features.append(proprio)
            
        # 4. Combine all features
        features = torch.cat(features, dim=-1).view(
            shape[0], -1, self.repr_dim
        )  # (B, T * num_feat_per_step, D)

        # 5. Forward through GPT trunk
        pred_actions = self.obs_trunk(
            features, 
            num_prompt_feats=0,
            stddev=1.0,
            action=action  # Pass ground truth actions if provided
        )

        return pred_actions

    def get_action(self, data, step):
        """ 
            Args:
            data (dict): Dictionary containing:
                - pixels: Agent view RGB images (B, 1, C, H, W) 
                - pixels_egocentric: Eye-in-hand RGB images (B, 1, C, H, W)
                - proprioceptive: Robot state features (B, 1, D_proprio)
                - task_emb: Task embedding (B, E)
        Get action for evaluation/inference"""
        # Add current observations to buffer
        for key in self.pixel_keys:
            self.observation_buffer[key].append(data[key])
        
        if self.use_proprio:
            self.proprio_buffer.append(data["proprioceptive"])

        # Stack buffered observations into tensors
        stacked_data = {}
        
        # Stack image observations
        for key in self.pixel_keys:
            stacked_data[key] = torch.cat(list(self.observation_buffer[key]), dim=1)
            
        # Stack proprioceptive features if used    
        if self.use_proprio:
            stacked_data["proprioceptive"] = torch.cat(list(self.proprio_buffer), dim=1)
            
        # Add task embedding
        stacked_data["task_emb"] = data["task_emb"]

        # Get predicted actions from forward pass
        with torch.no_grad():
            pred_actions = self.forward(stacked_data) # [2,1,70]
            
            # action chunking with batch dimension
            if self.temporal_agg:
                B = pred_actions.shape[0]  # batch dimension
                # Reshape to (B, num_queries, act_dim)
                action = pred_actions.view(B, -1, self.num_queries, self.act_dim)[:, -1]
                
                # Resize all_time_actions if batch size changes
                if self.all_time_actions.shape[0] != B:
                    self.all_time_actions = torch.zeros(
                        B,  # batch dimension
                        self.max_episode_len,
                        self.max_episode_len + self.num_queries,
                        self.act_dim
                    ).to(self.device)
                
                # Store predictions for each batch
                self.all_time_actions[:, step, step:step + self.num_queries] = action
                
                # Get actions for current step across all batches
                actions_for_curr_step = self.all_time_actions[:, :, step]  # (B, max_episode_len, act_dim)
                
                # Find valid (non-zero) actions for each batch
                actions_populated = torch.all(actions_for_curr_step != 0, dim=-1)  # (B, max_episode_len)
                
                # Calculate exponential weights
                k = 0.01
                max_populated = int(actions_populated.sum(dim=1).max().item())
                if max_populated > 0:
                    exp_weights = np.exp(-k * np.arange(max_populated))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).to(self.device)
                    
                    # Initialize final actions
                    final_actions = torch.zeros(B, self.act_dim).to(self.device)
                    
                    # Process each batch separately
                    for b in range(B):
                        valid_actions = actions_for_curr_step[b, actions_populated[b]]
                        if len(valid_actions) > 0:
                            # Use only as many weights as we have valid actions
                            b_weights = exp_weights[:len(valid_actions)]
                            b_weights = b_weights / b_weights.sum()
                            final_actions[b] = (valid_actions * b_weights.unsqueeze(-1)).sum(dim=0)
                    
                    action = final_actions
                else:
                    # If no valid actions, use the latest prediction
                    action = pred_actions[:, -1]
                
                
                return action.detach().cpu().numpy()
            
            return pred_actions[:, -1].detach().cpu().numpy()



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
    policy = BCBakuPolicy(**config).to(config['device'])
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
    
