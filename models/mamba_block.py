# Dominic + Zhenning
# What this file does:
# This is where the actual denoising happens. Mamba scans along time and frequency
# to clean up the audio spectrogram. Think of it like an RNN that goes through
# the spectrogram row by row for time and column by column for frequency.
#
# What you need to do:
# 1.Fix the imports cause they break on mamba-ssm 2.3.0 and the whole thing won't run
# 2.Let MambaBlock run forward-only, right now it always does forward+backward
# 3.Add a helper so we can switch between Mamba1 and Mamba2 from config
# 4.Add CausalTFMambaBlock — this is the main thing, it's what LiteAVSEMamba actually uses
#
# Why CausalTFMambaBlock matters:
# Real-time speech enhancement cannot look into the future. You cannot wait for
# someone to finish talking before you start denoising. So the time dimension
# has to be causal, meaning forward-only. But frequency is different — 100Hz and 4kHz
# exist at the same time, there is no "future" in frequency — so frequency stays
# bidirectional. This is one of the key differences from the original SEMamba.
#
# Recommended order: do 1 first, otherwise nothing runs, then 2 and 4 together, and 3 last.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange

# TODO fix these three imports, they are mamba-ssm 1.x paths and will crash on 2.3.0
# Block moved to mamba_ssm.modules.block
# RMSNorm moved to mamba_ssm.ops.triton.layer_norm, layernorm -> layer_norm
# also add this: from mamba_ssm.modules.mamba2 import Mamba2, you need it for the mixer switch
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm


# TODO add _get_mixer_cls(cfg, layer_idx) function here before create_block
# this reads cfg['model_cfg']['mamba_version'] which is either 'mamba' or 'mamba2'
# and returns the right class wrapped in functools.partial
# for 'mamba' as default use Mamba with d_state=16
# for 'mamba2' use Mamba2 with d_state=64 has to be power of 2 and headdim=32
# the point is so we can switch versions from the YAML config without touching code
# not urgent, do this last, the model runs fine on Mamba1 without it
# you can hardcode Mamba1 for now, just make sure to import Mamba2 at the top and add it to the mixer switch later when you do this


def create_block(
    d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
    ):
    d_state = cfg['model_cfg']['d_state'] # 16
    d_conv = cfg['model_cfg']['d_conv'] # 4
    expand = cfg['model_cfg']['expand'] # 4
    norm_epsilon = cfg['model_cfg']['norm_epsilon'] # 0.00001

    # TODO once you have _get_mixer_cls, replace this line with mixer_cls = _get_mixer_cls(cfg, layer_idx)
    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    # TODO add mlp_cls=nn.Identity here, mamba-ssm 2.3.0 requires it or it crashes
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block


# TODO add a bidirectional parameter here, default True so nothing breaks
# the problem right now is MambaBlock always does forward + backward and outputs [B, T, 2C]
# but CausalTFMambaBlock needs forward-only which outputs [B, T, C]
# so when bidirectional=False, skip the backward part entirely:
#   in __init__ don't create backward_blocks at all (saves memory)
#   in forward() skip the flip, skip backward_blocks, just return y_forward on its own
class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )
        self.backward_blocks = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward pass: processes the sequence in its original order
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward pass: flips the sequence, runs it through backward_blocks, then flips output back
        # this gives the model extra context by looking at the sequence in reverse
        # the two outputs get concatenated so the final output is [B, T, 2C]
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)


class TFMambaBlock(nn.Module):
    """
    Both time and freq are bidirectional here. This is the original non-causal version,
    it needs the full audio before it can process anything so it only works offline.
    We keep it for ablation comparison. LiteAVSEMamba uses CausalTFMambaBlock below.
    """
    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)

        # both are 2C -> C cause both mamba blocks are bidirectional so they output 2C
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


# TODO add CausalTFMambaBlock class here
# this is the block that LiteAVSEMamba actually uses, generator.py imports it
# it is almost identical to TFMambaBlock above, you can just copy it and change two things
# 1.time_mamba uses bidirectional=False (causal cannot see future frames)
#    freq_mamba uses bidirectional=True (frequency has no causality, same as before)
# 2.because causal mamba outputs C instead of 2C, tlinear becomes C->C not 2C->C
#    flinear stays 2C->C cause freq is still bidirectional
#
# the forward() logic is exactly the same as TFMambaBlock, nothing changes there input and output shape: [B, 64, T, F] -> [B, 64, T, F]
