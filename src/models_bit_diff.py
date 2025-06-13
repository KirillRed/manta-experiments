from turtle import st
import torch
import torch.nn as nn
import copy
import math
from einops import rearrange
from utils import *

from mamba_ssm.modules.mamba_simple import Mamba as ViM


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BitDiffPredictorTCN(nn.Module):
    def __init__(self, args, causal=False):
        super(BitDiffPredictorTCN, self).__init__()

        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,
            args.input_dim + 2 * args.num_classes,  # feat dim + noise dim + self cond dim
            args.num_classes,
            args.channel_dropout_prob,
            args.use_features,
        )

        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)


    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None):
        # arange
        x = rearrange(x, "b t c -> b c t")
        obs_cond = rearrange(obs_cond, "b t c -> b c t")
        self_cond = rearrange(self_cond, "b t c -> b c t")
        stage_masks = [rearrange(mask, "b t c -> b c t") for mask in stage_masks]

        if self.use_inp_ch_dropout:
            x = self.channel_dropout(x)

        # condition on input
        x = torch.cat((x, obs_cond), dim=1)
        x = torch.cat((x, self_cond), dim=1)

        frame_wise_pred, _ = self.ms_tcn(x, t, stage_masks)
        frame_wise_pred = rearrange(frame_wise_pred, "s b c t -> s b t c")
        return frame_wise_pred


class DiffMultiStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_stages,
        num_layers,
        num_f_maps,
        dim,
        num_classes,
        dropout,
        use_features=False,
    ):
        super(DiffMultiStageModel, self).__init__()
        self.stage1 = DiffSingleStageModel(
            layer_type,
            kernel_size,
            num_layers,
            num_f_maps,
            dim,
            num_classes,
            dropout,
        )

    def forward(self, x, t, stage_masks):
        out, out_features = self.stage1(x, t, stage_masks[0])
        outputs = out.unsqueeze(0)
        return outputs, out_features


class DiffSingleStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_layers,
        num_f_maps,
        dim,
        num_classes,
        dropout,
    ):
        super(DiffSingleStageModel, self).__init__()

        self.layer_types = {
            "mamba": DiffMambaResidualLayer,
        }

        #
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        # time cond
        time_dim = num_f_maps * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_f_maps),
            nn.Linear(num_f_maps, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )


        # MAMBA
        if layer_type in ["mamba"]:
            self.layers = []
            for i in range(num_layers):
                self.layers.append(
                    copy.deepcopy(
                        self.layer_types[layer_type](
                            kernel_size,
                            num_f_maps,
                            time_dim,
                            dropout,
                            'sum',
                            bimamba=True,
                        )
                    )
                )

        # ACCUM
        print(f"Total layers: {len(self.layers)}")
        self.layers = nn.ModuleList(self.layers)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x, t, mask):
        # embed
        out = self.conv_1x1(x) * mask
        time = self.time_mlp(t)

        # pass through layers
        for layer in self.layers:
            out = layer(out, time, mask)

        # output
        out_features = out * mask
        out_logits = self.conv_out(out) * mask
        return out_logits, out_features



# MAMBA
class DiffMambaResidualLayer(nn.Module):
    def __init__(
        self,
        kernel_size,
        out_channels,
        time_channels=-1,
        dropout=0.2,
        accum='sum',
        bimamba=True,
    ):
        super(DiffMambaResidualLayer, self).__init__()

        # mamba block
        self.mamba = ViM(
            out_channels,
            d_conv=kernel_size,
            use_fast_path=True,
            bimamba=bimamba,
            dropout=0.2,
            accum=accum,
        )
        self.drop_path = AffineDropPath(out_channels, drop_prob=dropout)
        self.norm = nn.LayerNorm(out_channels)

        # out block
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        # Time Net
        self.time_channels = time_channels
        if time_channels > 0:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_channels, out_channels * 2)
            )


    def forward(self, x, t, mask):
        # 1
        mamba_out = self.norm(x.permute(0, 2, 1)) * mask.permute(0, 2, 1)  # B x T x C
        mamba_out = self.mamba(mamba_out, mask)  # B x T x C
        mamba_out = mamba_out.permute(0, 2, 1)  # B x C x T
        mamba_out = self.drop_path(mamba_out) * mask

        # 2
        mamba_out = self.conv_1x1(mamba_out) * mask
        mamba_out = self.dropout(mamba_out)

        # 3
        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, "b d -> b d 1")
            time_shift = rearrange(time_shift, "b d -> b d 1")
            mamba_out = mamba_out * (time_scale + 1) + time_shift

        return (x + mamba_out) * mask

