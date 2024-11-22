import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
# from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing
from torch.utils.data import DataLoader, random_split, Subset
import torchmetrics
import tqdm
import time
import datetime
from earthnet.parallel_score import CubeCalculator
import yaml
import rerun as rr
from earthnet.plot_cube import gallery
from WeatherDataAutoEncoder import WeatherDataAutoEncoder
import torch.multiprocessing as mp
import os
import re

def closest_divisors(n):
    # Find the square root of the number
    sqrt_n = int(np.sqrt(n))
    
    # Iterate from sqrt(n) down to 1 to find the divisors
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            # Return the pair (i, n // i) when the divisor is found
            return i, n // i

def timeUpsample(x, factor):
    """ 
    Args:
        x: (B, C, T, H, W)
    """
    _, _, T, H, W = x.shape
    return F.interpolate(input=x, size=(T*factor, H, W), mode='nearest')

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint
        self.norm_layer = norm_layer

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        if self.norm_layer == 'LayerNorm':
            self.norm1 = nn.LayerNorm(dim)
        elif self.norm_layer == 'BatchNorm':
            self.norm1 = nn.BatchNorm3d(dim)
        else:
            self.norm1 = None
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_layer == 'LayerNorm':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_layer == 'BatchNorm':
            self.norm2 = nn.BatchNorm3d(dim)
        else:
            self.norm2 = None
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        if isinstance(self.norm1, nn.LayerNorm):
            x = self.norm1(x)       # B, D, H, W, C
        elif isinstance(self.norm1, nn.BatchNorm3d):
            x = x.permute(0, 4, 1, 2, 3) # B C D H W 
            x = self.norm1(x)
            x = x.permute(0, 2, 3, 4, 1) # B, D, H, W, C
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        if isinstance(self.norm2, nn.LayerNorm):
            x = self.norm2(x)      # B, D, H, W, C
        elif isinstance(self.norm2, nn.BatchNorm3d):
            x = x.permute(0, 4, 1, 2, 3) # B C D H W 
            x = self.norm2(x)
            x = x.permute(0, 2, 3, 4, 1) # B, D, H, W, C
        
        return self.drop_path(self.mlp(x))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of output channels.
        dim_down: Number of input channels
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, dim_down, norm_layer=None):
        super().__init__()
        self.dim = dim
        if norm_layer == 'LayerNorm':
            self.norm = nn.LayerNorm(4 * dim_down)
        elif norm_layer == 'BatchNorm':
            self.norm = nn.BatchNorm3d(4 * dim_down)
        else:
            self.norm = None
        self.reduction = nn.Linear(4 * dim_down, dim, bias=False)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        # x = x.permute(0,2,3,4,1) # B,T,H,W,Ch
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        if isinstance(self.norm, nn.LayerNorm):
            x = self.norm(x)
        elif isinstance(self.norm, nn.BatchNorm3d):
            x = x.permute(0, 4, 1, 2, 3) # B 4*C D H/2 W/2 
            x = self.norm(x)
            x = x.permute(0, 2, 3, 4, 1) # B D H/2 W/2 4*C 
        x = self.reduction(x)

        # x = x.permute(0,4,1,2,3) # B,Ch,T,H,W
        return x

class PatchExpansionV2(nn.Module):

    def __init__(self, dim, outputChannels, spatialScale=2, norm=None):
        super().__init__()
        self.spatialScale = spatialScale
        self.upsampe = nn.Upsample(scale_factor=(1, spatialScale, spatialScale), mode='nearest')
        self.conv = nn.Conv3d(in_channels=dim, out_channels=outputChannels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True)
        if norm == 'LayerNorm':
            self.norm = nn.LayerNorm(outputChannels)
        elif norm == 'BatchNorm':
            self.norm = nn.BatchNorm3d(outputChannels)
        else:
            self.norm = None
        

    def forward(self, x):
        # x -> B C T H W
        x = self.upsampe(x)

        x = self.conv(x)

        if isinstance(self.norm, nn.LayerNorm):
            x = x.permute(0, 2, 3, 4, 1) # x -> B T H W C
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3) # x -> B C T H W
        elif isinstance(self.norm, nn.BatchNorm3d):
            x = self.norm(x)

        return x
    
class RegressionHead(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(RegressionHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.in_channels // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.act = nn.Hardtanh(min_val=0, max_val=1)
        self.conv2 = nn.Conv3d(self.in_channels // 2, self.in_channels // 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.act2 = nn.Hardtanh(min_val=0, max_val=1)
        self.conv3 = nn.Conv3d(self.in_channels // 4, self.out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True)
        # self.sigm = nn.Sigmoid()
        # self.relu = nn.ReLU()

    def forward(self, x):
        """
        x : torch.tensor [N, C, H, W]
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        # x = self.sigm(x)
        # x = torch.clamp(x, min=0.0, max=1.0)
        # x = self.relu(x)

        return x

# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayerEncoder(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=None,
                 downsample=None,
                 use_checkpoint=False,
                 downsampleNorm=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, dim_down=dim//2, norm_layer=downsampleNorm)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

    def forward(self, x, contextWeather):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c') # x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')

        x = x + F.interpolate(contextWeather, size=(contextWeather.shape[2], x.shape[3], x.shape[4])) # Conditioning encoder on context weather data

        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x
    
class BasicLayerDecoder(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=None,
                 upsample=None,
                 use_checkpoint=False,
                 upsampleNorm=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim, dim//2, spatialScale=2, norm=upsampleNorm)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # B, C, D, H, W = x.shape
        # x = rearrange(x, 'b c d h w -> b d h w c') # x.view(B, D, H, W, C)

        
        # x = rearrange(x, 'b d h w c -> b c d h w')

        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        
        x = rearrange(x, 'b d h w c -> b c d h w')

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer == "LayerNorm":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == "BatchNorm":
            self.norm = nn.BatchNorm3d(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size() # B C D H W
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if isinstance(self.norm, nn.LayerNorm):
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        elif isinstance(self.norm, nn.BatchNorm3d):
            x = self.norm(x)

        return x # B C D Wh Ww

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(config['encoder']['layerDepths'])):
            layer = BasicLayerEncoder(dim=config['encoder']['numChannels'][i], 
                                      depth=config['encoder']['layerDepths'][i], 
                                      num_heads=config['encoder']['layerNumHeads'][i], 
                                      window_size=tuple(config['windowSize']), 
                                      mlp_ratio=config['encoder']['mlpRatio'], 
                                      qkv_bias=config['encoder']['qkvBias'], 
                                      qk_scale=config['encoder']['qkScale'], 
                                      drop=config['encoder']['drop'], 
                                      attn_drop=config['encoder']['attnDrop'], 
                                      drop_path=config['encoder']['dropPath'], 
                                      norm_layer=config['encoder']['norm'], 
                                      downsample=PatchMerging if config['encoder']['downsample'][i] else None, #None if i == 0 else PatchMerging, 
                                      use_checkpoint=config['encoder']['useCheckpoint'],
                                      downsampleNorm=config['encoder']['downsampleNorm'])
            self.layers.append(layer)

        # self.block0 = BasicLayerEncoder(dim=96, depth=2, num_heads=3, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, downsample=None, use_checkpoint=False)
        # self.block1 = BasicLayerEncoder(dim=192, depth=2, num_heads=6, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, downsample=PatchMerging, use_checkpoint=False)
        # self.block2 = BasicLayerEncoder(dim=384, depth=2, num_heads=12, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, downsample=PatchMerging, use_checkpoint=False)

    def forward(self, x, contextWeather):
        x_out = []

        N, C, T, H, W = x.shape

        # if self.ape:
        #     x = x + self.absolute_pos_embed

        for i, layer in enumerate(self.layers):

            # x = x + F.interpolate(contextWeather[0], size=(T, H, W)) # Conditioning encoder on context weather data
            # x = x + F.interpolate(staticData[0], size=(T, H, W)) # Conditioning encoder on context weather data

            x = layer(x, contextWeather[i])

            x_out.append(x)  

        return x_out
    
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(config['decoder']['layerDepths'])):
            layer = BasicLayerDecoder(dim=config['decoder']['numChannels'][i], 
                                      depth=config['decoder']['layerDepths'][i], 
                                      num_heads=config['decoder']['layerNumHeads'][i], 
                                      window_size=tuple(config['windowSize']), 
                                      mlp_ratio=config['decoder']['mlpRatio'], 
                                      qkv_bias=config['decoder']['qkvBias'], 
                                      qk_scale=config['decoder']['qkScale'], 
                                      drop=config['decoder']['drop'], 
                                      attn_drop=config['decoder']['attnDrop'], 
                                      drop_path=config['decoder']['dropPath'], 
                                      norm_layer=config['decoder']['norm'], 
                                      upsample=PatchExpansionV2 if config['decoder']['upsample'][i] else None, #None if i == len(config['decoder']['layerDepths'])-1 else PatchExpansionV2, 
                                      use_checkpoint=config['decoder']['useCheckpoint'],
                                      upsampleNorm=config['decoder']['upsampleNorm'])
            self.layers.append(layer)
        
        # self.block3 = BasicLayerDecoder(dim=384, depth=2, num_heads=3, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, upsample=PatchExpansionV2, use_checkpoint=False)
        # self.block4 = BasicLayerDecoder(dim=384, depth=2, num_heads=6, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, upsample=PatchExpansionV2, use_checkpoint=False)
        # self.block5 = BasicLayerDecoder(dim=288, depth=2, num_heads=12, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, upsample=None, use_checkpoint=False)


    def forward(self, x, targetWeather):

        for i, x_i in enumerate(reversed(x)):
            
            if i != 0:
                if x_out.size()[-3:] != x_i.size()[-3:]:
                    x_out = F.interpolate(x_out, size=x_i.size()[-3:], mode='nearest')

                x_out = x_out + F.interpolate(list(reversed(targetWeather))[i], size=(list(reversed(targetWeather))[i].shape[2], x_out.shape[3], x_out.shape[4])) # Conditioning decoder on target weather data

                x_i = torch.cat((x_out, x_i), dim=1)

            else:
                x_i = x_i + F.interpolate(list(reversed(targetWeather))[i], size=(list(reversed(targetWeather))[i].shape[2], x_i.shape[3], x_i.shape[4])) # Conditioning decpder on target weather data

            x_out = self.layers[i](x_i)

        return x_out

        # x = self.block3(x_out[-1])

        # if x.size()[-3:] != x_out[-2].size()[-3:]:
        #     x = F.interpolate(x, size=x_out[-2].size()[-3:], mode='trilinear', align_corners=False)
        # x = torch.cat((x, x_out[-2]), dim=1)

        # x = self.block4(x)

        # if x.size()[-3:] != x_out[-3].size()[-3:]:
        #     x = F.interpolate(x, size=x_out[-3].size()[-3:], mode='trilinear', align_corners=False)
        # x = torch.cat((x, x_out[-3]), dim=1)

        # x = self.block5(x)


class VideoSwinUNet(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.config = config
        self.timeUpsamplingFactor = config['outputTime'] // config['mainInputTime'] * config['patchSize'][0]

        self.patch_embed = PatchEmbed3D(patch_size=tuple(config['patchSize']), 
                                        in_chans=config['modelInputCh'], 
                                        embed_dim=config['C'], 
                                        norm_layer=config['patchEmbedding']['norm'])
        
        if self.config['ape']:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.config['C'], self.config['mainInputTime'] // config['patchSize'][0], config['inputHeightWidth'] // config['patchSize'][1], config['inputHeightWidth'] // config['patchSize'][2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.encoder = Encoder(config=config)
        # self.block0 = BasicLayerEncoder(dim=96, depth=2, num_heads=3, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, downsample=None, use_checkpoint=False)
        # self.block1 = BasicLayerEncoder(dim=192, depth=2, num_heads=6, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, downsample=PatchMerging, use_checkpoint=False)
        # self.block2 = BasicLayerEncoder(dim=384, depth=2, num_heads=12, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, downsample=PatchMerging, use_checkpoint=False)
        self.contextWeatherEncoder = WeatherDataAutoEncoder(config=config)
        if config['contextWeatherAutoencoderWeights']:
            weatherDataAutoEncoderCheckpoint = torch.load(config['contextWeatherAutoencoderWeights'])
            self.contextWeatherEncoder.load_state_dict(weatherDataAutoEncoderCheckpoint['state_dict'])
        self.contextWeatherEncoder = self.contextWeatherEncoder.encoder
        self.contextWeatherEncoder.eval()
        for param in self.contextWeatherEncoder.parameters():
            param.requires_grad = False

        self.targetWeatherEncoder = WeatherDataAutoEncoder(config=config)
        if config['targetWeatherAutoencoderWeights']:
            weatherDataAutoEncoderCheckpoint = torch.load(config['targetWeatherAutoencoderWeights'])
            self.targetWeatherEncoder.load_state_dict(weatherDataAutoEncoderCheckpoint['state_dict'])
        self.targetWeatherEncoder = self.targetWeatherEncoder.encoder
        self.targetWeatherEncoder.eval()
        for param in self.targetWeatherEncoder.parameters():
            param.requires_grad = False

        self.decoder = Decoder(config=config)
        # self.block3 = BasicLayerDecoder(dim=384, depth=2, num_heads=3, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, upsample=PatchExpansionV2, use_checkpoint=False)
        # self.block4 = BasicLayerDecoder(dim=384, depth=2, num_heads=6, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, upsample=PatchExpansionV2, use_checkpoint=False)
        # self.block5 = BasicLayerDecoder(dim=288, depth=2, num_heads=12, window_size=(3, 7, 7), mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=None, upsample=None, use_checkpoint=False)
          
        self.head = RegressionHead(in_channels=config['decoder']['numChannels'][-1], out_channels=config['modelOutputCh'])

        self.init_weights(logger=logger, config=config)

        
    def inflate_weights(self, logger, config):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # delete head and last norm layer
        del state_dict['head.weight'], state_dict['head.bias']
        if not config['patchEmbedding']['norm']:
            del state_dict['norm.weight'], state_dict['norm.bias']

        # state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,config['patchSizeT'],1,1) / config['patchSizeT']
        # prepare patch embed weights
        m1 = state_dict['patch_embed.proj.weight'].mean()
        s1 = state_dict['patch_embed.proj.weight'].std()
        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].mean(axis=(1, 2, 3)).\
        unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, config['modelInputCh'], config['patchSize'][0], config['patchSize'][1], config['patchSize'][2])
        m2 = state_dict['patch_embed.proj.weight'].mean()
        s2 = state_dict['patch_embed.proj.weight'].std()
        state_dict['patch_embed.proj.weight'] = m1 + (state_dict['patch_embed.proj.weight'] - m2) * s1/s2

        # Change names of relative position bias tables
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        # Set pretrained weights only on encoder
        relative_position_bias_table_keys_ori = sorted([k for k in self.state_dict().keys() if ("relative_position_bias_table" in k and ('encoder' in k))])
        
        rel_pos_bias_table_tmp = {}
        for k, j in zip(relative_position_bias_table_keys, relative_position_bias_table_keys_ori):
            rel_pos_bias_table_tmp[j] = state_dict[k]

        for k in rel_pos_bias_table_tmp.keys():
                state_dict[k] = rel_pos_bias_table_tmp[k]


        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            if k in list(self.state_dict().keys()):
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = self.state_dict()[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                L2 = (2*config['windowSize'][1]-1) * (2*config['windowSize'][2]-1)
                wd = config['windowSize'][0]
                if nH1 != nH2:
                    if logger: logger.warning(f"Error in loading {k}, passing")
                else:
                    if L1 != L2:
                        S1 = int(L1 ** 0.5)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*config['windowSize'][1]-1, 2*config['windowSize'][2]-1),
                            mode='bicubic')
                        relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        if logger:
            logger.info(msg)
            logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def transfer_weights(self, logger, config):
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']

        new_state_dict = {}

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # delete last norm layer
        del state_dict['backbone.norm.weight'], state_dict['backbone.norm.bias']

        # If we have different number of embedding dimension, reshape the weights
        if config['C'] != 96:
            factor = 96 // config['C']
            for k in state_dict.keys():
                if 'patch_embed.proj.weight' in k:
                    # reduce only 1st dimension
                    A, B, C, D, E = state_dict[k].shape
                    state_dict[k] = state_dict[k].reshape(config['C'], factor, B, C, D, E).mean(1).reshape(config['C'], B, C, D, E)
                elif 'relative_position_bias_table' in k:
                    continue
                else:
                    # reduce all dimensions
                    if len(state_dict[k].shape) == 1:
                        A = state_dict[k].shape[0]
                        state_dict[k] = state_dict[k].reshape(A // factor, factor).mean(1).reshape(A // factor)
                    elif len(state_dict[k].shape) == 2:
                        A, B = state_dict[k].shape
                        state_dict[k] = state_dict[k].reshape(A // factor, factor, B // factor, factor).mean(dim=(1, 3)).reshape(A // factor, B // factor)

        # prepare patch embed weights
        m1 = state_dict['backbone.patch_embed.proj.weight'].mean()
        s1 = state_dict['backbone.patch_embed.proj.weight'].std()
        state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'].mean(axis=(1, 2, 3, 4)).\
        unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, config['modelInputCh'], config['patchSize'][0], config['patchSize'][1], config['patchSize'][2])
        m2 = state_dict['backbone.patch_embed.proj.weight'].mean()
        s2 = state_dict['backbone.patch_embed.proj.weight'].std()
        state_dict['backbone.patch_embed.proj.weight'] = m1 + (state_dict['backbone.patch_embed.proj.weight'] - m2) * s1/s2

        patch_embed_keys = [k for k in state_dict.keys() if "patch_embed" in k]
        for k in patch_embed_keys:
            new_state_dict[k.removeprefix("backbone.")] = state_dict[k]

        # change 'backbone' prefix to 'encoder' prefix and add 1 to the downsample layers numbers for the encoder
        for k in list(state_dict.keys()):
            match = re.search(r"backbone.layers\.(\d+)", k)
            if not match:
                continue
            numLayer = int(match.group(1))
            if (numLayer == 0 or numLayer == 1 or numLayer == 2):
                if "downsample" in k:
                    new_state_dict["encoder.layers." + str(numLayer + 1) + k.removeprefix("backbone.layers." + str(numLayer))] = state_dict[k]
                    # del state_dict[k]
                else:
                    if "relative_position_bias_table" in k:
                        state_dict["encoder" + k.removeprefix("backbone")] = state_dict[k]
                        del state_dict[k]
                    else:
                        new_state_dict["encoder" + k.removeprefix("backbone")] = state_dict[k]
                        # del state_dict[k]
        
        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            if k in list(self.state_dict().keys()):
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = self.state_dict()[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                L2 = (2*config['windowSize'][1]-1) * (2*config['windowSize'][2]-1)
                wd = config['windowSize'][0]
                if nH1 != nH2:
                    if logger: logger.warning(f"Error in loading {k}, passing")
                else:
                    if L1 != L2:
                        # S1 = int(L1 ** 0.5)
                        S1, D1 = closest_divisors(L1)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, D1), size=(2*config['windowSize'][1]-1, 2*config['windowSize'][2]-1),
                            mode='bicubic')
                        relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)
                new_state_dict[k] = state_dict[k]
        
        # for k in list(new_state_dict.keys()):
        #     print(k + f" {new_state_dict[k].shape}")
        msg = self.load_state_dict(new_state_dict, strict=False)
        if logger:
            logger.info(msg)
            logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, logger, config, init_type='xavier', gain=1):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def init_weights(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1 or classname.find('LayerNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    #torch.nn.init.normal_(m.weight.data, 1.0, gain)
                    torch.nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_weights)

        def init_func(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_func)

        for i, m in enumerate(self.head.children()):
            if hasattr(m, '_init_weights'):
                m._init_weights()
            else:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                    trunc_normal_(m.weight, std=.02)
                    if i == 4:
                        nn.init.constant_(m.bias, 0.263)

        if config['pretrained']:
            self.pretrained = config['pretrained']
        if isinstance(config['pretrained'], str):
            if logger: logger.info(f'load model from: {self.pretrained}')

            if config['pretrained2D']:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger, config)
            elif config['pretrained3D']:
                # Directly load 3D model.
                self.transfer_weights(logger, config)
                # torch.load(self, self.pretrained, strict=False, logger=logger)
        else:
            if logger: logger.info("No pretrained loading")

    def forward(self, contextImgDEM, contextWeather, targetWeather):

        # Input shape B,C,T,H,W
        x = self.patch_embed(contextImgDEM)
        # print("Patch embedding output shape {}".format(x.shape))

        if self.config['ape']:
            x = x + self.absolute_pos_embed

        if self.config['timeUpsampling'] == 'patchEmbedding':
            x = timeUpsample(x, self.timeUpsamplingFactor)
            # print("Time upsampling output shape {}".format([x_i.shape for x_i in x]))

        with torch.no_grad():
            contextWeather = self.contextWeatherEncoder(contextWeather)
            # print("Context weather data shape {}".format([x_i.shape for x_i in contextWeather]))
            targetWeather = self.targetWeatherEncoder(targetWeather)

        x = self.encoder(x, contextWeather)
        # print("Encoder output shape {}".format(encoder_outputs.shape))

        if self.config['timeUpsampling'] == 'encoder':
            x = [timeUpsample(x_out, self.timeUpsamplingFactor) for x_out in x]
            # print("Time upsampling output shape {}".format([x_i.shape for x_i in x]))

        # targetWeatherEnc = self.targetWeatherEncoder(targetWeather)
        # print(f"Target weather data encoder output shape {targetWeatherEnc.shape}")

        # x = [torch.cat((x_i, F.interpolate(targetWeather, size=(targetWeather.shape[2], x_i.shape[3], x_i.shape[4]))), dim=1) for x_i in x]
        # print("Decoder input shape {}".format([x_i.shape for x_i in x]))

        x = self.decoder(x, targetWeather)
        # print("Decoder output shape {}".format(decoder_output.shape))

        if self.config['patchSize'] != (1, 1, 1):
            x = F.interpolate(x, size=(self.config['outputTime'], self.config['inputHeightWidth'], self.config['inputHeightWidth']), mode='nearest')
            # print("Final interpolation output shape {}".format(decoder_output.shape))

        x = self.head(x)
        # print("Head output shape {}".format(decoder_output.shape))

        return x


if __name__ == "__main__":

    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    config = load_config('/home/nikoskot/earthnetThesis/experiments/videoSwinUnetV7/videoSwinUnetV7_20-10-2024_14-06-14/configV7.yml')

    startTime = time.time()
    model = VideoSwinUNet(config, logger=None).to(torch.device('cuda'))
    endTime = time.time()
    print("Model creation time {}".format(endTime - startTime))
    print(model)

    for k in list(model.state_dict().keys()):
        print(k + f" {torch.mean(model.state_dict()[k].float())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    model.train()

    totalParams = sum(p.numel() for p in model.parameters())

    startTime = time.time()
    contextImgAndDEM = torch.rand(1, 5, 10, 128, 128).to(torch.device('cuda')) # B, C, T, H, W
    contextWeather = torch.rand(1, 5, 10, 80, 80).to(torch.device('cuda')) # B, C, T, H, W
    targetWeather = torch.rand(1, 5, 20, 80, 80).to(torch.device('cuda')) # B, C, T, H, W
    endTime = time.time()
    print("Dummy data creation time {}".format(endTime - startTime))

    # torch.cuda.memory._record_memory_history(max_entries=100000)
    
    for i in range(20):
        _ = torch.randn(1).cuda()
        startTime = time.time()
        y = model(contextImgAndDEM, contextWeather, targetWeather)
        endTime = time.time()
        print("Pass time {}".format(endTime - startTime))
        optimizer.zero_grad(set_to_none=True)

    # torch.cuda.memory._dump_snapshot("videoSwinUnetV1.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)

    y = model(contextImgAndDEM, contextWeather, targetWeather)

    print("Output shape {}".format(y.shape))

    print("Number of model parameters {}".format(totalParams))


def train_loop(dataloader, 
               model, 
               l1Loss, 
               ssimLoss, 
               mseLoss,
               vggLoss,
               optimizer, 
               config, 
               logger, 
               trainVisualizationCubenames,
               epoch):

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    l1LossSum = 0
    mseLossSum = 0
    SSIMLossSum = 0
    vggLossSum = 0

    batchNumber = 0

    maxGradPre = torch.tensor(0)
    minGradPre = torch.tensor(0)

    for data in tqdm.tqdm(dataloader):
        batchNumber += 1

        optimizer.zero_grad()

        # Move data to GPU
        contextImgDEM = data['contextImgDEM'].to(torch.device('cuda'))
        contextWeather = data['contextWeather'].to(torch.device('cuda'))
        targetWeather = data['targetWeather'].to(torch.device('cuda'))
        y = data['y'].to(torch.device('cuda'))
        masks = data['targetMask'].to(torch.device('cuda'))

        # Compute prediction and loss
        pred = model(contextImgDEM, contextWeather, targetWeather)
        # pred = torch.clamp(pred, min=0.0, max=1.0)

        l1LossValue = l1Loss(pred, y, masks)
        if config['useMSE']:
            mseLossValue = mseLoss(pred, y, masks)
        if config['useSSIM']:
            ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, masks)
        if config['useVGG']:
            vggLossValue = vggLoss(torch.clamp(pred.clone(), min=0, max=1), y.clone(), masks)

        # Backpropagation
        totalLoss = config['l1Weight'] * l1LossValue
        if config['useMSE']:
            totalLoss = totalLoss + (config['mseWeight'] * mseLossValue)
        if config['useSSIM']:
            totalLoss = totalLoss + (config['ssimWeight'] * ssimLossValue)
        if config['useVGG']:
            totalLoss = totalLoss + (config['vggWeight'] * vggLossValue)

        totalLoss.backward()

        for param in model.parameters():
            if param.grad is not None:
                maxGradPre = torch.maximum(maxGradPre, torch.max(param.grad.view(-1)))
                minGradPre = torch.minimum(minGradPre, torch.min(param.grad.view(-1)))

        if config['gradientClipping']:
            nn.utils.clip_grad_value_(model.parameters(), config['gradientClipValue'])

        optimizer.step()

        # Add the loss to the total loss 
        l1LossSum += l1LossValue.item()
        if config['useSSIM']:
            SSIMLossSum += ssimLossValue.item()
        if config['useMSE']:
            mseLossSum += mseLossValue.item()
        if config['useVGG']:
            vggLossSum += vggLossValue.item()

        # Log visualizations if available
        if epoch % config['visualizationFreq'] == 0 or epoch == 1 or epoch == config['epochs']:
            rr.set_time_sequence('epoch', epoch)
            for i in range(len(data['cubename'])):
                if data['cubename'][i] in trainVisualizationCubenames:

                    # # Log ground truth
                    # targ = data['y'][i, ...].numpy()
                    # targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
                    # targ[targ < 0] = 0
                    # targ[targ > 0.5] = 0.5
                    # targ = 2 * targ
                    # targ[np.isnan(targ)] = 0
                    # grid = gallery(targ)
                    # rr.log('/train/groundTruth/{}'.format(data['cubename'][i]), rr.Image(grid))

                    # Log prediction
                    targ = pred[i, ...].detach().cpu().numpy()
                    targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
                    targ[targ < 0] = 0
                    targ[targ > 0.5] = 0.5
                    targ = 2 * targ
                    targ[np.isnan(targ)] = 0
                    grid = gallery(targ)
                    rr.log('/train/prediction/{}'.format(data['cubename'][i]), rr.Image(grid))
    
    logger.info("Maximum gradient before clipping: {}".format(maxGradPre))
    logger.info("Minimum gradient before clipping: {}".format(minGradPre))

    return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber, vggLossSum / batchNumber

def validation_loop(dataloader, 
                    model, 
                    l1Loss, 
                    ssimLoss, 
                    mseLoss,
                    vggLoss,
                    config, 
                    validationVisualizationCubenames,
                    epoch,
                    ensCalculator,
                    logger):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    l1LossSum = 0
    SSIMLossSum = 0
    mseLossSum = 0
    vggLossSum = 0
    earthnetScore = 0

    batchNumber = 0

    pool = mp.Pool(processes=config['batchSize'])
    allENSMetrics = []
    def appendENSResults(result):
        allENSMetrics.append([result['MAD'], result['OLS'], result['EMD'], result['SSIM']])
    def handler(error):
        print(f'Error: {error}', flush=True)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            batchNumber += 1
            
            # Move data to GPU
            contextImgDEM = data['contextImgDEM'].to(torch.device('cuda'))
            contextWeather = data['contextWeather'].to(torch.device('cuda'))
            targetWeather = data['targetWeather'].to(torch.device('cuda'))
            y = data['y'].to(torch.device('cuda'))
            masks = data['targetMask'].to(torch.device('cuda'))

            # Compute prediction and loss
            pred = model(contextImgDEM, contextWeather, targetWeather)
            # pred = torch.clamp(pred, min=0.0, max=1.0)

            l1LossValue = l1Loss(pred, y, masks)
            if config['useMSE']:
                mseLossValue = mseLoss(pred, y, masks)
            if config['useSSIM']:
                ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, masks)
            if config['useVGG']:
                vggLossValue = vggLoss(torch.clamp(pred.clone(), min=0, max=1), y.clone(), masks)

            # Add the loss to the total loss
            l1LossSum += l1LossValue.item()
            if config['useSSIM']:
                SSIMLossSum += ssimLossValue.item()
            if config['useMSE']:
                mseLossSum += mseLossValue.item()
            if config['useVGG']:
                vggLossSum += vggLossValue.item()

            # Log visualizations if available
            if epoch % config['visualizationFreq'] == 0 or epoch == 1 or epoch == config['epochs']:
                rr.set_time_sequence('epoch', epoch)
                for i in range(len(data['cubename'])):
                    if data['cubename'][i] in validationVisualizationCubenames:

                        # # Log ground truth
                        # targ = data['y'][i, ...].numpy()
                        # targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
                        # targ[targ < 0] = 0
                        # targ[targ > 0.5] = 0.5
                        # targ = 2 * targ
                        # targ[np.isnan(targ)] = 0
                        # grid = gallery(targ)
                        # rr.log('/train/groundTruth/{}'.format(data['cubename'][i]), rr.Image(grid))

                        # Log prediction
                        targ = pred[i, ...].detach().cpu().numpy()
                        targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
                        targ[targ < 0] = 0
                        targ[targ > 0.5] = 0.5
                        targ = 2 * targ
                        targ[np.isnan(targ)] = 0
                        grid = gallery(targ)
                        rr.log('/validation/prediction/{}'.format(data['cubename'][i]), rr.Image(grid))
            
            if epoch % config['calculateENSonValidationFreq'] == 0 or epoch == config['epochs']:
                # pred = torch.clamp(pred.clone(), min=0, max=1)
                # ensCalculator(pred.permute(0, 2, 3, 4, 1), y.permute(0, 2, 3, 4, 1), 1-masks.permute(0, 2, 3, 4, 1))
                # preprocessing
                pred = torch.clamp(pred, min=0, max=1).permute(0, 3, 4, 1, 2).detach().cpu().numpy() # BHWCT
                target = y.permute(0, 3, 4, 1, 2).detach().cpu().numpy() # BHWCT
                mask = masks.permute(0, 3, 4, 1, 2).detach().cpu().numpy().repeat(4,3)
                ndvi_pred = ((pred[:,:,:,3,:] - pred[:,:,:,2,:])/(pred[:,:,:,3,:] + pred[:,:,:,2,:] + 1e-6))[:,:,:,np.newaxis,:]
                ndvi_targ = ((target[:,:,:,3,:] - target[:,:,:,2,:])/(target[:,:,:,3,:] + target[:,:,:,2,:] + 1e-6))[:,:,:,np.newaxis,:]
                ndvi_mask = mask[:,:,:,0,:][:,:,:,np.newaxis,:]
                for t in range(pred.shape[0]):
                    pool.apply_async(calculateENSMetrics, args=(pred[t], target[t], mask[t], ndvi_pred[t], ndvi_targ[t], ndvi_mask[t]), callback=appendENSResults, error_callback=handler)

        if epoch % config['calculateENSonValidationFreq'] == 0 or epoch == config['epochs']:
            logger.info("Validation split Earthnet Score on epoch {}".format(epoch))
            pool.close()
            pool.join()
            allENSMetrics = np.array(allENSMetrics, dtype = np.float64)
            meanScores = np.nanmean(allENSMetrics, axis = 0).tolist()
            earthnetScore = harmonicMean(meanScores)
            logger.info(f"MAD: {meanScores[0]}, OLS: {meanScores[1]}, EMD: {meanScores[2]}, SSIM: {meanScores[3]}, ENS: {earthnetScore}")
            # ens = ensCalculator.compute()
            # logger.info(ens)
            # earthnetScore = ens['EarthNetScore']
            # ensCalculator.reset()

    return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber, vggLossSum / batchNumber, earthnetScore

def testing_loop(dataloader, 
                model, 
                l1Loss, 
                ssimLoss, 
                mseLoss,
                vggLoss,
                predsFolder,
                config):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    l1LossSum = 0
    SSIMLossSum = 0
    mseLossSum = 0
    vggLossSum = 0

    batchNumber = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            
            batchNumber += 1
            
            # Move data to GPU
            contextImgDEM = data['contextImgDEM'].to(torch.device('cuda'))
            contextWeather = data['contextWeather'].to(torch.device('cuda'))
            targetWeather = data['targetWeather'].to(torch.device('cuda'))
            y = data['y'].to(torch.device('cuda'))
            masks = data['targetMask'].to(torch.device('cuda'))

            # Compute prediction and loss
            pred = model(contextImgDEM, contextWeather, targetWeather)
            # pred = torch.clamp(pred, min=0.0, max=1.0)

            l1LossValue = l1Loss(pred, y, masks)
            if config['useMSE']:
                mseLossValue = mseLoss(pred, y, masks)
            if config['useSSIM']:
                ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, masks)
            if config['useVGG']:
                vggLossValue = vggLoss(torch.clamp(pred.clone(), min=0, max=1), y.clone(), masks)

            # Add the loss to the total loss
            l1LossSum += l1LossValue.item()
            if config['useSSIM']:
                SSIMLossSum += ssimLossValue.item()
            if config['useMSE']:
                mseLossSum += mseLossValue.item()
            if config['useVGG']:
                vggLossSum += vggLossValue.item()

            tiles = data['tile']
            cubenames = data['cubename']

            # Save predictions
            for i in range(len(tiles)):
                path = os.path.join(predsFolder, tiles[i])
                os.makedirs(path, exist_ok=True)
                np.savez_compressed(os.path.join(path, cubenames[i]), highresdynamic=pred[i].permute(2, 3, 0, 1).detach().cpu().numpy().astype(np.float16))

    return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber, vggLossSum / batchNumber

def calculateENSMetrics(pred, targ, mask, ndvi_pred, ndvi_targ, ndvi_mask):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mad, _ = CubeCalculator.MAD(pred, targ, mask)
    ols, _ = CubeCalculator.OLS(ndvi_pred, ndvi_targ, ndvi_mask)
    emd, _ = CubeCalculator.EMD(ndvi_pred, ndvi_targ, ndvi_mask)
    ssim, _ = CubeCalculator.SSIM(pred, targ, mask)
    
    return {
        "MAD": mad,
        "OLS": ols,
        "EMD": emd,
        "SSIM": ssim,
    }

def harmonicMean(vals):
    vals = list(filter(None, vals))
    if len(vals) == 0:
        return None
    else:
        return min(1,len(vals)/sum([1/(v+1e-8) for v in vals]))