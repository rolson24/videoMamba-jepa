# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks

import os

from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    


class VisionMambaPredictor(nn.Module):
    """ Vision Mamba """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        depth=12, 
        embed_dim=768,
        channels=3, 
        predictor_embed_dim = 384,
        # num_classes=1000,
        drop_rate=0.,
        drop_path_rate=0.1,
        ssm_cfg=None, 
        norm_epsilon=1e-5, 
        initializer_cfg=None,
        fused_add_norm=True,
        rms_norm=True, 
        residual_in_fp32=True,
        bimamba=True,
        # video
        kernel_size=2, # token depth in temporal dimension
        num_frames=8, # must be greater than kernel_size
        fc_drop_rate=0., 
        device=None,
        dtype=None,
        # checkpoint
        use_checkpoint=False,
        checkpoint_num=0,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.bimamba = bimamba
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for i in range(num_mask_tokens)
            ])

        # Determine positional embedding
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        self.num_patches = (num_frames // kernel_size) * self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))



        self.input_size = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
       
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights

        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        ) 


        # if self.predictor_pos_embed is not None:
        #     self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std = 0.02
        
        # init mask tokens
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        # self.apply(self._init_weights)
        # self._rescale_blocks()

    # def _init_pos_embed(self, pos_embed):
    #     embed_dim = pos_embed.size(-1)
    #     grid_size = self.input_size // self.patch_size
    #     if self.is_video:
    #         grid_depth = self.num_frames // self.tubelet_size
    #         sincos = get_3d_sincos_pos_embed(
    #             embed_dim,
    #             grid_size,
    #             grid_depth,
    #             cls_token=False,
    #             uniform_power=self.uniform_power
    #         )
    #     else:
    #         sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
    #     pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=self.init_std)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.mixer.mamba.in_proj.weight.data, layer_id + 1)
            rescale(layer.mixer.mamba.conv1d.weight.data, layer_id + 1)
            rescale(layer.mixer.mamba.x_proj.weight.data, layer_id + 1)
            rescale(layer.mixer.mamba.dt_proj.weight.data, layer_id + 1)
            rescale(layer.mixer.mamba.out_proj.weight.data, layer_id + 1)
            if self.bimamba:
                rescale(layer.mixer.mamba.conv1d_b.weight.data, layer_id + 1)
                rescale(layer.mixer.mamba.x_proj_b.weight.data, layer_id + 1)
                rescale(layer.mixer.mamba.dt_proj_b.weight.data, layer_id + 1)


    def diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):

        # Prepare diffusion noise schedule
        b1, b2 = noise_beta
        beta_scheduler = (b1 + i*(b2-b1)/steps for i in range(steps))
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1.-_beta
            alpha_scheduler += [_alpha]

        # Sample diffusion time step
        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)

        # Normalize features and apply noise
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha**0.5 * x + (1.-alpha)**0.5 * torch.randn(x.shape, device=x.device)
        return x

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=1):
        """
        :param ctxt: context tokens
        :param tgt: target tokens
        :param masks_ctxt: indices of context tokens in input
        :params masks_tgt: indices of target tokens in input
        """

        assert (masks_ctxt is not None) and (masks_tgt is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]

        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        # Batch Size
        B = len(ctxt) // len(masks_ctxt)

        # Map context tokens to pedictor dimensions
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape

        # Add positional embedding to ctxt tokens
        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(ctxt_pos_embed, masks_ctxt)

        # Map target tokens to predictor dimensions & add noise (fwd diffusion)
        if self.mask_tokens is None:
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = self.diffusion(pred_tokens)
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)

        # Add positional embedding to target tokens
        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_tgt)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs

        # Concatenate context & target tokens
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # FIXME: this implementation currently assumes masks_ctxt and masks_tgt
        # are alligned 1:1 (ok with MultiMask wrapper on predictor but
        # otherwise will break)
        masks_ctxt = torch.cat(masks_ctxt, dim=0)
        masks_tgt = torch.cat(masks_tgt, dim=0)
        masks = torch.cat([masks_ctxt, masks_tgt], dim=1)

        # Fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, mask=masks)
        x = self.predictor_norm(x)

        # Return output corresponding to target tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


def vmamba_predictor(**kwargs):
    model = VisionMambaPredictor(
        rms_norm=True, **kwargs)
    return model
