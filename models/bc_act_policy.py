#%%
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
import torch.nn.functional as F
from utils import get_route_embeddings
from dotenv import load_dotenv
import os

load_dotenv()
dataset_path_= os.getenv("DATASET_PATH")
n_task = 10
language_token_dim = 768
route_embeddings = None




class PrototypeTaskGate(nn.Module):
    """
    Prototype-based gating mechanism that routes similar language tokens 
    to the same expert by maintaining routing embeddings that get updated
    through direct replacement.
    """
    def __init__(self, language_token_dim=768, n_experts=10, similarity_threshold=0.5):
        super().__init__()
        self.language_token_dim = language_token_dim
        self.n_experts = n_experts
        self.similarity_threshold = similarity_threshold
        
        # Initialize random routing embeddings
        self.routing_embeddings = route_embeddings
        

    def forward(self, language_token):
        """
        Args:
            language_token (Tensor): shape (B, language_token_dim)
            
        Returns:
            weights (Tensor): shape (B, n_experts), one-hot routing vectors
            indices (Tensor): shape (B, 1), selected expert indices
        """
        batch_size = language_token.size(0)
        
        # Get device of language token
        device = language_token.device
        self.routing_embeddings = self.routing_embeddings.to(device)
        

        # Compute cosine similarity
        norm_tokens = F.normalize(language_token, p=2, dim=-1)
        norm_embeddings = F.normalize(self.routing_embeddings, p=2, dim=-1)
        similarities = torch.matmul(norm_tokens, norm_embeddings.t())  # (B, n_experts)
   
        max_sims, indices = similarities.max(dim=-1)  # (B,)
        indices = indices.unsqueeze(-1)  # (B, 1)
     
        
        # Create one-hot routing weights
        weights = torch.zeros(batch_size, self.n_experts, device=language_token.device)
        weights.scatter_(1, indices, 1.0)
        # expert_counts = torch.bincount(indices.view(-1), minlength=self.n_experts)
        # print("Expert selection counts:", expert_counts.tolist())
        return weights, indices


### ACT MOE Transformer Decoder
class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.
    """
    def __init__(self, dim, n_experts=8, n_expert_groups=1, n_limited_groups=1, 
                 n_activated_experts=2, score_func="softmax", route_scale=1.0):
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.gate_score = nn.Linear(language_token_dim, n_experts, bias=False)
        self.n_routed_experts = n_experts
        self.bias = nn.Parameter(torch.zeros(n_experts))
        self.token_counts = None

    def forward(self, x):
        """Forward pass for the gating mechanism."""
        scores = self.gate_score(x)
        
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        
        
        scores = scores + self.bias
            
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
            
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        # Store token counts for load balancing
        self.token_counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        
        return weights.type_as(x), indices


class Expert(nn.Module):
    """Expert layer for MoE models."""
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Mixture-of-Experts (MoE) module."""
    def __init__(self, input_size, mlp_hidden_size, n_experts=8, n_expert_groups=1, 
                 n_limited_groups=1, n_activated_experts=2, n_shared_experts=2, dropout=0.0,expert_scale=0.1):
        super().__init__()
        self.input_size = input_size
        self.n_routed_experts = n_experts
        self.n_local_experts = n_experts
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.n_local_experts
        self.expert_weights = nn.Parameter(torch.ones(n_experts) * expert_scale)
        # self.gate = Gate(
        #     input_size, 
        #     n_experts=n_experts,
        #     n_expert_groups=n_expert_groups,
        #     n_limited_groups=n_limited_groups,
        #     n_activated_experts=n_activated_experts
        # )
        # self.gate = TaskSpecificGate(language_token_dim=language_token_dim, n_experts=n_experts)
        self.gate = PrototypeTaskGate(language_token_dim=language_token_dim, n_experts=n_experts)
        # Initialize routed experts
        self.experts = nn.ModuleList([
            Expert(input_size, mlp_hidden_size) 
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(n_experts)
        ])
        
        # Add multiple shared experts using MLP
        shared_hidden_size = n_shared_experts * mlp_hidden_size
        self.shared_experts = nn.Sequential(
            nn.Linear(input_size, shared_hidden_size),
            nn.SiLU(),
            nn.Linear(shared_hidden_size, input_size)
        )

    def forward(self, x , language_token):
        original_shape = x.size()
        # x = x.view(-1, self.input_size)
        
        # Ensure batch sizes match
        if x.shape[0] != language_token.shape[0]:
            raise ValueError(f"Batch size mismatch: x has shape {x.shape[0]} but language_token has shape {language_token.shape[0]}")
        # Get routing weights and expert indices
        weights, indices = self.gate(language_token) # (B, 1), (B, 1)
        
        # Initialize output tensor
        y = torch.zeros_like(x)
        
        # Count tokens per expert
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        # Route inputs to experts
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, _ = torch.where(indices == i)
            # Multiply the expert output by its corresponding weight
            y[idx] += self.expert_weights[i] * expert(x[idx])
        print(f"expert_weights: {self.expert_weights}")
        # Add shared experts computation
        z = self.shared_experts(x)
       
        return (y + z).view(original_shape)


    
###
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
        learnable_tokens=True,
        n_dense_layers=0,  # NEW: number of dense (standard MLP) layers; layers beyond this will use MoE when use_moe is True
        use_moe=False      # NEW: flag to use MoE for layers beyond n_dense_layers
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_output_size = head_output_size
        self.dropout = dropout
        self.num_queries = num_queries
        self.n_dense_layers = n_dense_layers
        self.use_moe = use_moe
        
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
        
        # Layer components modified to have:
        # 1. Cross attention with one norm layer
        # 2. Self attention with one norm layer
        # 3. MLP (or MoE) with one norm layer
        self.layers = nn.ModuleList([])
        for layer_idx in range(num_layers):
            layer = {}
            # Step 1: Cross attention and its norm.
            layer['cross_attention'] = nn.MultiheadAttention(
                    embed_dim=input_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            layer['cross_norm'] = nn.LayerNorm(input_size)
            
            # Step 2: Self attention and its norm.
            layer['self_attention'] = nn.MultiheadAttention(
                    embed_dim=input_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            layer['self_norm'] = nn.LayerNorm(input_size)
            
            # Step 3: MLP (or MoE) and its norm.
            layer['mlp_norm'] = nn.LayerNorm(input_size)
            if self.use_moe and layer_idx >= self.n_dense_layers:
                # Use MoE module (using hard-coded parameters as in the transformer_modules example)
                layer['mlp'] = MoE(
                            input_size=input_size,
                            mlp_hidden_size=mlp_hidden_size,
                            n_experts=n_task,
                            n_expert_groups=1,
                            n_limited_groups=1,
                            n_activated_experts=1,
                            dropout=dropout,
                            n_shared_experts=1
                        )
            else:
                # Standard MLP
                layer['mlp'] = nn.Sequential(
                    nn.Linear(input_size, mlp_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_size, input_size),
                    nn.Dropout(dropout)
                )
            self.layers.append(nn.ModuleDict(layer))
            
    def forward(self, x,language_token):
        """
        Args:
            x: Input embeddings (B*T, num_features, input_size)
            language_token: Task embedding (B, 1, E)
        Returns:
            output: Processed action tokens (B, num_queries, input_size)
        """
        batch_size = x.shape[0]

        # language_token = 
        
        # Expand action tokens to batch size and add positional encoding
        queries = self.action_tokens.expand(batch_size, -1, -1)
        pos_enc = self.pos_encoding(queries)
        queries = queries + pos_enc
        
        # For storing attention weights for visualization if desired
        self.cross_attention_weights = []
        self.self_attention_weights = []
        
        # Process through layers in three clear steps per layer:
        # 1. Cross attention with residual connection
        # 2. Multi-head self attention (with causal mask) with residual connection
        # 3. MLP (or MoE) with residual connection
        for layer_idx, layer in enumerate(self.layers):
            # Step 1: Cross Attention + Residual connection
            q_norm = layer['cross_norm'](queries)
            cross_out, cross_weights = layer['cross_attention'](
                query=q_norm,
                key=x,
                value=x,
                need_weights=True
            )
            self.cross_attention_weights.append(cross_weights)
            queries = queries + cross_out
            
            # Step 2: Multi-head Self Attention (with causal mask) + Residual connection
            s_norm = layer['self_norm'](queries)
            self_out, self_weights = layer['self_attention'](
                query=s_norm,
                key=s_norm,
                value=s_norm,
                attn_mask=self.causal_mask,
                need_weights=True
            )
            self.self_attention_weights.append(self_weights)
            queries = queries + self_out
            
            # Step 3: MLP (or MoE) + Residual connection
            mlp_norm = layer['mlp_norm'](queries)  # (B, num_queries, E)
            if self.use_moe and layer_idx >= self.n_dense_layers:
                mlp_out = layer['mlp'](mlp_norm, language_token)
            else:
                mlp_out = layer['mlp'](mlp_norm)
            queries = queries + mlp_out
            
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
                 learnable_tokens=False,
                 n_layer=8,
                 use_moe=False,
                 benchmark_name="libero_spatial"):
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
        self.step = 0
        self.n_task = n_task
        self.use_moe = use_moe
        self.num_queries = num_queries

        global route_embeddings
        route_embeddings = get_route_embeddings(os.path.join(os.getenv("DATASET_PATH"), benchmark_name))
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
            num_layers=n_layer,
            num_heads=4,
            head_output_size=64,
            mlp_hidden_size=hidden_dim,
            dropout=0.1,
            num_queries=self.num_queries,
            learnable_tokens=learnable_tokens,
            n_dense_layers=1,
            use_moe=use_moe
        )
        
        # Initialize temporal position encoding
        self.temporal_position_encoding = SinusoidalPositionEncoding(
            input_size=repr_dim,
            inv_freq_factor=10000
        )
        
        # Action head for final prediction
        if policy_head == "deterministic":
            self.action_head = DeterministicHead(
                self.repr_dim, self.act_dim, num_layers=2 
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

    def temporal_encode(self, x,language_token):
        """
        Apply temporal encoding and transformer processing
        x: (B, T, num_modalities, E)
        """
        B,T,num_modalities,E = x.shape
        
        x = x.reshape(B*T, num_modalities, E)
        if self.use_moe:
            output = self.transformer(x,language_token) # (B*T, num_queries, E)
        else:
            output = self.transformer(x) # (B*T, num_queries, E)
        output = output.reshape(B, T, -1, E)
        

        
        return output

    def reset(self):
        """Reset episode state"""
        self.step = 0
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
        language_token = data["task_emb"] # (B, 1, E)
        
        # Encode spatial features
        x = self.spatial_encode(data)
        B,T,_,_ = x.shape
        # Expand language token to match temporal dimension
        # language_token is currently [B, 1, E]
        # Need to repeat T times to get [B, T, E]
        language_token = language_token.repeat(1, T, 1)  # [B, T, E]
        language_token = language_token.view(B*T, -1)
        # Apply temporal encoding
        x = self.temporal_encode(x,language_token)
        B,T,num_queries,E = x.shape
        # reshape x to [B,T*num_queries,E]
        x = x.reshape(B, -1, E)

        # Get action predictions from action head
        pred_actions = self.action_head(x) # (B, T*num_queries, act_dim)
 

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
            language_token = data["task_emb"]
            
            # Encode spatial features
            x = self.spatial_encode(data) # (B, 1, num_modalities, E)
            B,T,_,_ = x.shape
            language_token = language_token.repeat(1, T, 1)  # [B, T, E]
            language_token = language_token.view(B*T, -1)
            # Apply temporal encoding
            x = self.temporal_encode(x,language_token) # (B, 1, num_queries, E)
            
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
    def train_step(self, data, optimizer=None):
        """
        Performs a training step for BC-ACT.
        Expects data to contain ground truth actions under the key "actions".
        Returns:
            loss (torch.Tensor): the negative log likelihood loss.
        """
        self.train()
        gt_actions = data.get("actions", None)
        B,T,D = gt_actions.shape
        gt_actions = gt_actions.reshape(B, T, self.num_queries, self.act_dim)
        gt_actions = gt_actions.view(B, T*self.num_queries, self.act_dim)
        if gt_actions is None:
            raise ValueError("Ground truth actions missing in training batch")
        # Forward pass returns an output distribution.
        pred_dist = self.forward(data)
        # Compute the negative log likelihood loss from the output distribution.
        loss = -pred_dist.log_prob(gt_actions).mean()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if self.use_moe:
            # self.update_expert_bias()
            # task_emb = data["task_emb"] # (B,1, E)
            # Count occurrences of each unique task embedding
            # task_emb_flat = task_emb.view(task_emb.size(0), -1)  # Flatten to (B, E)
            
            # # Count frequency of each embedding
            # unique_counts = {}
            # for i in range(task_emb_flat.size(0)):
            #     # Convert tensor to tuple for hashing
            #     emb_tuple = tuple(task_emb_flat[i].cpu().detach().numpy())
            #     unique_counts[emb_tuple] = unique_counts.get(emb_tuple, 0) + 1
            
            # # Print frequency of each unique embedding
            # print("\nTask Embedding Frequencies:")
            # for i, (emb, count) in enumerate(unique_counts.items()):
            #     if count > 1:  # Only print embeddings that repeat
            #         print(f"Task Embedding {i}: appears {count} times")

            
        return loss
    
    def update_expert_bias(self, u=0.001, threshold=0.4):
        """
        Update the per-expert bias based on the token assignment counts from the gating modules.
        
        This function iterates over all submodules in the model, and for any module that contains a gate
        with a bias parameter and token_counts attribute, it computes:
        
            avg = mean(c_i)
            e_i = avg - c_i
            
        and updates the bias as: b_i = b_i + u * sign(e_i)
        
        Args:
            u (float): The bias update rate.
            threshold (float): Maximum absolute value for bias before stopping updates.
        """
        for module in self.modules():
            # Check if the module has a gate with bias and token counts
            if hasattr(module, 'gate') and hasattr(module.gate, "bias") and hasattr(module.gate, "token_counts"):
                counts = module.gate.token_counts  # assumed to be a tensor of shape [num_experts]
                print(counts)
                # Make sure counts are in float for the arithmetic
                counts = counts.float()
                avg = counts.mean()
                error = avg - counts # avg - counts
                with torch.no_grad():
                    # Only update bias if below threshold
                    mask = (module.gate.bias.abs() < threshold)
                    # Update: b = b + u * sign(error) where mask is True
                    module.gate.bias += u * torch.sign(error) * mask
                    print(f"bias updated: {module.gate.bias}")

