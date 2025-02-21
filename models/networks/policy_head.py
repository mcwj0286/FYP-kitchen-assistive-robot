import einops
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import utils


# from .utils.diffusion_policy import DiffusionPolicy
# from .utils.vqbet.pretrain_vqvae import init_vqvae, pretrain_vqvae
from .mlp import MLP





class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=True,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None, **kwargs):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


# NEW: MultiTokenDeterministicHead
class MultiTokenDeterministicHead(nn.Module):
    """
    Multi-token version of DeterministicHead that uses multiple sub-heads for
    parallel or sequential token predictions.
    Inspired by 'Better & Faster Large Language Models via Multi-token Prediction' 
    https://arxiv.org/abs/2210.06610
    and references from MTT https://github.com/ljatynu/MTT
    """
    def __init__(self, 
                 input_size, 
                 output_size, 
                 num_tokens=3, 
                 hidden_size=1024, 
                 num_layers=2, 
                 action_squash=True):
        super().__init__()
        # Instead of having an additional shared layer,
        # we assume that the transformer decoder already produces a shared representation.
        # Thus, each head will independently map the input representation
        # to its token prediction.
        self.heads = nn.ModuleList([
            DeterministicHead(
                input_size=input_size,   # use transformer's output dimension
                output_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                action_squash=action_squash
            )
            for _ in range(num_tokens)
        ])

    def forward(self, x, **kwargs):
        """
        Expects x to be the transformer decoder output.
        Applies each independent head to x and returns a list of distributions.
        """
        dists = [head(x, **kwargs) for head in self.heads]
        return dists

    def multi_token_loss(self, dists, targets, reduction="mean"):
        """
        Helper: compute negative log-likelihood across multiple sub-distributions.
        targets: list or tuple of ground-truth tokens: [y1, y2, ..., yN]
        """
        losses = []
        for dist, tgt in zip(dists, targets):
            log_probs = dist.log_prob(tgt)  # shape depends on dist
            loss_i = -log_probs
            if reduction == "mean":
                loss_i = loss_i.mean()
            losses.append(loss_i)
        total_loss = torch.stack(losses).sum()
        return total_loss

######################################### Task-Specific Head #########################################
# NEW: TaskSpecificHead

class TaskSpecificGate(nn.Module):
    """
    Prototype-based gating mechanism that routes similar language tokens 
    to the same expert by maintaining routing embeddings that get updated
    through direct replacement.
    """
    def __init__(self, language_token_dim=768, n_experts=10, similarity_threshold=0.5, route_embeddings=None):
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
class TaskSpecificHead(nn.Module):
    """
    Task-specific policy head that selects among n_task DeterministicHead policies
    via a gating mechanism. Given a task embedding (language token), it computes 
    cosine similarity with learnable gate embeddings to decide which policy to activate.
    """
    def __init__(self, 
                 input_size, 
                 output_size, 
                 n_tasks, 
                 language_token_dim,
                 hidden_size=256, 
                 num_layers=2, 
                 action_squash=True,
                 similarity_threshold=0.5,
                 route_embeddings=None):
        super().__init__()
        self.n_tasks = n_tasks
        self.output_size = output_size
        self.similarity_threshold = similarity_threshold
        
        self.gate = TaskSpecificGate(language_token_dim, n_tasks, similarity_threshold, route_embeddings)
        # Create n_tasks DeterministicHead policy heads
        self.policy_heads = nn.ModuleList([
            DeterministicHead(input_size, output_size, hidden_size, num_layers, action_squash)
            for _ in range(n_tasks)
        ])
        
    def forward(self, x, language_token, stddev=None, **kwargs):
        """
        Args:
            x (Tensor): Input features for the policy heads, shape (B, input_size)
            language_token (Tensor): Task embeddings, shape (B, language_token_dim)
            stddev (float, optional): Standard deviation override passed to DeterministicHead.
            
        
        """
        batch_size = x.size(0)
        weights, indices = self.gate(language_token) # (B, 1), (B, 1)
        # Count tokens per expert
        counts = torch.bincount(indices.flatten(), minlength=self.n_tasks).tolist()
        # print("Expert selection counts:", counts)
        # Initialize output tensor
        y = torch.zeros_like(x)
        for i in range(self.n_tasks):
            if counts[i] == 0:
                continue
            head = self.policy_heads[i]
            idx, _ = torch.where(indices == i)
            y[idx] = head(x[idx])
          
            
        
        return y 