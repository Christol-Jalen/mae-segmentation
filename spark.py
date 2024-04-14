# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import encoder
from decoder import LightDecoder


class SparK(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, densify_norm='bn', sbn=False,
    ):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        self.fmap_h, self.fmap_w = input_size // downsample_raito, input_size // downsample_raito
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
        
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        
        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        
        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(self.hierarchy): # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            
            # create densify norm
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)
            
            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()    # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)
            
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2
        
        print(f'[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')
        
        # these are deprecated and would never be used; can be removed.
        self.register_buffer('imn_m', torch.empty(1, 3, 1, 1))
        self.register_buffer('imn_s', torch.empty(1, 3, 1, 1))
        self.register_buffer('norm_black', torch.zeros(1, 3, input_size, input_size))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...
    
    def mask(self, B: int, device, generator=None):
        '''
        While this function originally generate a random mask map, it is modified to return a all-ones mask map for fine-tuning.
        '''
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.ones(B, h * w, dtype=torch.bool, device=device).view(B, 1, h, w)
    
    def forward(self, inp_bchw: torch.Tensor, active_b1ff: torch.BoolTensor):
        # step1. Mask
        if active_b1ff is None:    
            active_b1ff: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)
        encoder._cur_active = active_b1ff    # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw
        
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        
        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff     # (B, 1, f, f)
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
  
        return rec_bchw
    

    
    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, "")}'
        )
    
    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            
            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(SparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
