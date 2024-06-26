# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        # print(f"mask shape: {m.shape}")
        # print(f"x shape: {x.shape}")
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        # print(f"mask shape unsqueezed: {mask_keep.shape}")
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
        # print(f"all_x shape: {all_x[0].shape}")
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)
