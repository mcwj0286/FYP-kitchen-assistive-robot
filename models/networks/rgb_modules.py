"""
Code from: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/models/modules/rgb_modules.py 

This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import utils
import torchvision.transforms as T
###############################################################################
#
# Modules related to encoding visual information (can conditioned on language)
#
###############################################################################


class BaseEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 512

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        if obs_shape[1] == 84:
            dim = 39200
        elif obs_shape[1] == 128:
            dim = 103968
        elif obs_shape[1] == 224:
            dim = 352800
        self.trunk = nn.Sequential(nn.Linear(dim, 512), nn.LayerNorm(512), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        h = h.reshape(h.shape[0], -1)
        h = self.trunk(h)
        return h


class PatchEncoder(nn.Module):
    """
    A patch encoder that does a linear projection of patches in a RGB image.
    """

    def __init__(
        self, input_shape, patch_size=[16, 16], embed_size=64, no_patch_embed_bias=False
    ):
        super().__init__()
        C, H, W = input_shape
        num_patches = (H // patch_size[0] // 2) * (W // patch_size[1] // 2)
        self.img_size = (H, W)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.h, self.w = H // patch_size[0] // 2, W // patch_size[1] // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            64,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )
        self.bn = nn.BatchNorm2d(embed_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.proj(x)
        x = self.bn(x)
        return x


class SpatialSoftmax(nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class SpatialProjection(nn.Module):
    def __init__(self, input_shape, out_dim):
        super().__init__()

        assert (
            len(input_shape) == 3
        ), "[error] spatial projection: input shape is not a 3-tuple"
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class ResnetEncoder(nn.Module):
    """
    A Resnet-18-based encoder for mapping an image to a latent vector

    Encode (f) an image into a latent vector.

    y = f(x), where
        x: (B, C, H, W)
        y: (B, H_out)

    Args:
        input_shape:      (C, H, W), the shape of the image
        output_size:      H_out, the latent vector size
        pretrained:       whether use pretrained resnet
        freeze: whether   freeze the pretrained resnet
        remove_layer_num: remove the top # layers
        no_stride:        do not use striding
    """

    def __init__(
        self,
        input_shape,
        output_size,
        pretrained=False,
        freeze=False,
        remove_layer_num=2,
        no_stride=False,
        language_dim=768,
        language_fusion="film",
    ):
        super().__init__()

        ### 1. encode input (images) using convolutional layers
        assert remove_layer_num <= 5, "[error] please only remove <=5 layers"
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[
            :-remove_layer_num
        ]
        self.remove_layer_num = remove_layer_num

        assert (
            len(input_shape) == 3
        ), "[error] input shape of resnet should be (C, H, W)"

        in_channels = input_shape[0]
        if in_channels != 3:  # has eye_in_hand, increase channel size
            conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1

        self.resnet18_base = nn.Sequential(*layers[:4])
        self.block_1 = layers[4][0]
        self.block_2 = layers[4][1]
        self.block_3 = layers[5][0]
        self.block_4 = layers[5][1]

        self.language_fusion = language_fusion
        if language_fusion != "none":
            self.lang_proj1 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj2 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj3 = nn.Linear(language_dim, 128 * 2)
            self.lang_proj4 = nn.Linear(language_dim, 128 * 2)

        if freeze:
            if in_channels != 3:
                raise Exception(
                    "[error] cannot freeze pretrained "
                    + "resnet with the extra eye_in_hand input"
                )
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        ### 2. project the encoded input to a latent space
        x = torch.zeros(1, *input_shape)
        y = self.block_4(
            self.block_3(self.block_2(self.block_1(self.resnet18_base(x))))
        )
        output_shape = y.shape  # compute the out dim
        self.projection_layer = SpatialProjection(output_shape[1:], output_size)
        self.output_shape = self.projection_layer(y).shape

        # Replace BatchNorm layers with GroupNorm
        self.resnet18_base = utils.batch_norm_to_group_norm(self.resnet18_base)
        self.block_1 = utils.batch_norm_to_group_norm(self.block_1)
        self.block_2 = utils.batch_norm_to_group_norm(self.block_2)
        self.block_3 = utils.batch_norm_to_group_norm(self.block_3)
        self.block_4 = utils.batch_norm_to_group_norm(self.block_4)

        # self.normlayer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, lang=None, return_intermediate=False):
        # # preprocess
        # preprocess = nn.Sequential(self.normlayer)
        # x = preprocess(x)

        h = self.resnet18_base(x)

        h = self.block_1(h)
        if lang is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj1(lang).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_2(h)
        if lang is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj2(lang).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_3(h)
        if lang is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj3(lang).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_4(h)
        if lang is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj4(lang).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        if not return_intermediate:
            h = self.projection_layer(h)
        return h

    def output_shape(self):
        return self.output_shape

class MPIVisionEncoder(nn.Module):
    def __init__(self, use_proj=False,mpi_root_dir="/home/john/project/FYP-kitchen-assistive-robot/models/networks/utils/MPI/mpi/checkpoints/mpi-small", device="cuda:0", output_dim=512):
        super().__init__()
        import sys
        import os
        # Add MPI directory to path
        mpi_dir = os.path.dirname(os.path.dirname(os.path.dirname(mpi_root_dir)))
        if mpi_dir not in sys.path:
            sys.path.append(mpi_dir)
        from mpi import load_mpi
        # Load the MPI model; freeze=True ensures weights are fixed
        self.mpi_model = load_mpi(mpi_root_dir, device, freeze=True)
        # Projection layer: assume MPI output dimension is 384 (from example), project to output_dim
        if use_proj:
            self.proj = nn.Linear(384, output_dim)
        else:
            self.proj = nn.Identity()
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
        # x = x[:,-1,:].unsqueeze(1) # (N, 1, 384)
        out = self.proj(x)  # (N, 1, output_dim)
        return out
