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
        pred_action = self._action_head(
            features,
            stddev,
            
        )
        return pred_action




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
                 history=True, history_len=10,
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

        # observation trunk
        self.obs_trunk = Obersvation_trunk(
            repr_dim=repr_dim,
            act_dim=act_dim,
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
        
        self.history = history
        self.history_len = history_len if history else 1
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.use_proprio = use_proprio
        
        # Initialize observation buffers
        self.observation_buffer = {}
        for key in self.pixel_keys:
            self.observation_buffer[key] = deque(maxlen=self.history_len)
        if self.use_proprio:
            self.proprio_buffer = deque(maxlen=self.history_len)
            
        # For temporal aggregation if enabled
        if self.temporal_agg:
            self.register_buffer('all_time_actions', 
                torch.zeros([self.max_episode_len, self.max_episode_len + 1, act_dim])
            )

    def forward(self, data):
        """Process observations and predict actions"""
        # 1. Process language embedding for FiLM
        lang = self.language_projector(data["task_emb"])
        
        # 2. Process vision features
        vision_feats = []
        for key in self.pixel_keys:
            pixels = data[key].float() / 255.0  # Normalize pixels
            B, T, C, H, W = pixels.shape
            # Reshape to [B*T, C, H, W] for ResNet processing
            pixels = pixels.view(B * T, C, H, W)
            feat = self.vision_encoder[key](pixels, lang=lang)
            # Reshape back to [B, T, feat_dim]
            feat = feat.view(B, T, -1)
            vision_feats.append(feat)
            
        # 3. Process proprioceptive features if used
        if self.use_proprio:
            proprio = data["proprioceptive"]
            proprio_feat = self.proprio_projector(proprio)
            vision_feats.append(proprio_feat)
            
        # 4. Combine all features
        obs = torch.cat(vision_feats, dim=-1)

        # 5. Forward through GPT trunk
        pred_actions = self.obs_trunk(
            obs, 
            num_prompt_feats=0,
            stddev=1.0
        )

        return pred_actions

    def get_action(self, data):
        """Get action for evaluation/inference"""
        self.eval()
        with torch.no_grad():
            pred_actions = self.forward(data)
            
            # Handle different policy head outputs
            if self._policy_head == "deterministic":
                action = pred_actions.mean
            elif self._policy_head == "vqbet":
                action = pred_actions["predicted_action"]
            
            if self.temporal_agg: # action chunking
                step = data.get("step", 0)
                # Store current prediction
                self.all_time_actions[step, step:step+1] = action
                
                # Compute weighted average of past predictions
                actions_for_curr_step = self.all_time_actions[:, step]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                if len(actions_for_curr_step) > 0:
                    k = 0.01  # Decay factor
                    exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step))).to(self.device)
                    exp_weights = exp_weights / exp_weights.sum()
                    action = (actions_for_curr_step * exp_weights.unsqueeze(1)).sum(0)
            
            return action.cpu().numpy()[0]

    def reset(self):
        """Reset history buffers"""
        for key in self.pixel_keys:
            self.observation_buffer[key].clear()
        if self.use_proprio:
            self.proprio_buffer.clear()
        if self.temporal_agg:
            self.all_time_actions.zero_()