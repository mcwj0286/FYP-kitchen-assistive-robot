# FILE: baku.py
import torch
import torch.nn as nn
from collections import deque
from torch.nn import functional as F

from modules.rgb_modules import BaseEncoder, ResnetEncoder
from agent.networks.policy_head import (
    DeterministicHead,
    GMMHead,
    BeTHead,
    VQBeTHead,
    DiffusionHead,
)
from agent.networks.mlp import MLP
from agent.networks.gpt import GPT, GPTConfig
from agent.networks.kmeans_discretizer import KMeansDiscretizer

class ExtraModalityTokens(nn.Module):
    def __init__(
        self,
        use_proprio=False,
        proprio_dim=0,
        extra_num_layers=0,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):
        super().__init__()
        self.use_proprio = use_proprio
        self.extra_embedding_size = extra_embedding_size

        self.num_extra = int(use_proprio)
        extra_low_level_feature_dim = int(use_proprio) * proprio_dim

        assert extra_low_level_feature_dim > 0, "[error] no extra information"

        layers = []
        input_dim = extra_low_level_feature_dim
        for _ in range(extra_num_layers):
            layers.append(nn.Linear(input_dim, extra_hidden_size))
            layers.append(nn.ReLU())
            input_dim = extra_hidden_size
        layers.append(nn.Linear(input_dim, extra_embedding_size))
        self.extra_encoder = nn.Sequential(*layers)

    def forward(self, obs_dict):
        proprio = obs_dict.get('proprio', None)  # Adjust the key as per your data
        if proprio is not None:
            proprio = proprio.view(proprio.shape[0], -1)
            extra_tokens = self.extra_encoder(proprio)
            return extra_tokens.unsqueeze(1)  # (B, 1, E)
        else:
            return None

class BakuPolicy(nn.Module):
    def __init__(self, cfg, obs_shape, action_shape):
        super().__init__()
        self.cfg = cfg
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # Encoder
        if cfg['obs_type'] == 'pixels':
            self.image_encoder = ResnetEncoder(
                input_channel=obs_shape[0],
                output_size=cfg['embed_size'],
                backbone=cfg['backbone'],
                num_blocks=cfg['num_blocks'],
            )
        else:
            self.image_encoder = None

        # Language Encoder (if used)
        if cfg.get('use_language', False):
            self.language_encoder = MLP(
                input_dim=cfg['language_dim'],
                hidden_dims=[cfg['language_hidden_size']] * cfg['language_num_layers'],
                output_dim=cfg['embed_size'],
                activation=nn.ReLU,
            )
        else:
            self.language_encoder = None

        # Extra Modality Encoder
        self.extra_encoder = ExtraModalityTokens(
            use_proprio=cfg.get('use_proprio', False),
            proprio_dim=cfg.get('proprio_dim', 0),
            extra_num_layers=cfg.get('extra_num_layers', 0),
            extra_hidden_size=cfg.get('extra_hidden_size', 64),
            extra_embedding_size=cfg['embed_size'],
        )

        # Transformer
        self.transformer = GPT(
            GPTConfig(
                cfg['embed_size'],
                cfg['transformer_block_size'],
                n_layer=cfg['transformer_num_layers'],
                n_head=cfg['transformer_num_heads'],
                n_embd=cfg['transformer_embedding_size'],
            )
        )

        # Policy Head
        policy_head_type = cfg.get('policy_head', 'deterministic')
        if policy_head_type == 'deterministic':
            self.policy_head = DeterministicHead(
                input_size=cfg['embed_size'],
                output_size=action_shape[0],
                hidden_dims=[cfg['policy_hidden_size']] * cfg['policy_num_layers'],
                activation=nn.ReLU,
            )
        elif policy_head_type == 'gmm':
            self.policy_head = GMMHead(
                input_size=cfg['embed_size'],
                output_size=action_shape[0],
                num_modes=cfg['gmm_num_modes'],
                hidden_dims=[cfg['policy_hidden_size']] * cfg['policy_num_layers'],
                activation=nn.ReLU,
            )
        # Add other policy heads as needed

    def forward(self, data):
        # 1. Encode Image
        if self.image_encoder:
            images = data['obs']['images']  # Adjust the key as per your data
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
            image_tokens = self.image_encoder(images)
            image_tokens = image_tokens.view(B, T, -1, self.cfg['embed_size'])
        else:
            image_tokens = None

        # 2. Encode Language
        if self.language_encoder:
            language = data['obs']['language']  # Adjust the key as per your data
            language_tokens = self.language_encoder(language)
            language_tokens = language_tokens.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, E)
        else:
            language_tokens = None

        # 3. Encode Extra Modality
        extra_tokens = self.extra_encoder(data['obs'])

        # 4. Concatenate Tokens
        tokens = []
        if language_tokens is not None:
            tokens.append(language_tokens)
        if extra_tokens is not None:
            tokens.append(extra_tokens.unsqueeze(1))
        if image_tokens is not None:
            tokens.append(image_tokens)
        x = torch.cat(tokens, dim=2)  # (B, T, N, E)

        # 5. Flatten Tokens for Transformer
        B, T, N, E = x.shape
        x = x.view(B, T * N, E)

        # 6. Pass Through Transformer
        x = self.transformer(x)

        # 7. Policy Head
        action = self.policy_head(x[:, -1, :])  # Use the last token's output

        return action

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            action = self.forward(data)
        return action.cpu().numpy()

    def reset(self):
        pass  # Implement if your policy requires resetting between episodes