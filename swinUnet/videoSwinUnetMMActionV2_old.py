# Version 2 of Video Swin Transformer. Here themesodynamic data of the target part are not given at the begining of the model. They are used at the bottleneck as extra context.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing
from torch.utils.data import DataLoader, random_split, Subset
import torchmetrics
import tqdm
import time
import datetime
from earthnet.parallel_score import CubeCalculator


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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
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

        if self.norm_layer is not None:
            self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_layer is not None:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        if self.norm_layer is not None:
            x = self.norm1(x)
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
        return self.drop_path(self.mlp(self.norm2(x))) if self.norm_layer is not None else self.drop_path(self.mlp(x))

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
    def __init__(self, dim, dim_down, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim_down) if norm_layer is not None else None
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

        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)

        # x = x.permute(0,4,1,2,3) # B,Ch,T,H,W
        return x

class PatchExpansionV2(nn.Module):

    def __init__(self, dim, outputChannels, spatialScale=2, norm=None):
        super().__init__()
        self.spatialScale = spatialScale
        self.upsampe = nn.Upsample(scale_factor=(1, spatialScale, spatialScale), mode='trilinear')
        self.conv = nn.Conv3d(in_channels=dim, out_channels=outputChannels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True)
        self.norm = nn.LayerNorm(outputChannels) if norm else None

    def forward(self, x):
        # x -> B C T H W
        x = self.upsampe(x)

        x = self.conv(x)

        if self.norm is not None:
            x = x.permute(0, 2, 3, 4, 1) # x -> B T H W C
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3) # x -> B C T H W

        #x = x.permute(0,4,1,2,3) # B,Ch,T,H,W
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
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, dim_down=dim//2, norm_layer=norm_layer)

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

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c') # x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')

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
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False):
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
            self.upsample = upsample(dim, dim//2, spatialScale=2, norm=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
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
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
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
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

class Encoder(nn.Module):
    def __init__(self, C, window_size):
        super().__init__()
        self.enc_swin_blocks = nn.ModuleList([
            BasicLayer(dim=C,   depth=2, num_heads=3, window_size=window_size, downsample=None),
            BasicLayer(dim=2*C, depth=2, num_heads=6, window_size=window_size, downsample=None),
            BasicLayer(dim=4*C, depth=2, num_heads=12, window_size=window_size, downsample=None)
        ])
        self.enc_patch_merge_blocks = nn.ModuleList([
            PatchMerging(C),
            PatchMerging(2*C),
            PatchMerging(4*C)
        ])

    def forward(self, x):
        skip_conn_ftrs = []
        for swin_block, patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs
    
class Decoder(nn.Module):
    def __init__(self, C, window_size):
        super().__init__()
        self.dec_swin_blocks = nn.ModuleList([
            BasicLayer(dim=4*C, depth=2, num_heads=12, window_size=window_size, downsample=None),
            BasicLayer(dim=2*C, depth=2, num_heads=6, window_size=window_size, downsample=None),
            BasicLayer(dim=C,   depth=2, num_heads=3, window_size=window_size, downsample=None)
        ])
        self.dec_patch_expand_blocks = nn.ModuleList([
            PatchExpansion(8*C),
            PatchExpansion(4*C),
            PatchExpansion(2*C)
        ])
        self.skip_conn_concat = nn.ModuleList([
            nn.Linear(8*C, 4*C),
            nn.Linear(4*C, 2*C),
            nn.Linear(2*C, 1*C)
        ])

    def forward(self, x, encoder_features):
        for patch_expand,swin_block, enc_ftr, linear_concatter in zip(self.dec_patch_expand_blocks, self.dec_swin_blocks, encoder_features, self.skip_conn_concat):
            x = patch_expand(x)
            x = torch.cat([x, enc_ftr.permute(0,2,3,4,1)], dim=-1)
            x = linear_concatter(x)
            x = x.permute(0,4,1,2,3) # B,Ch,T,H,W
            x = swin_block(x)
        return x
    
class ExtraDataEncoder(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, bottleneck_shape):
        super(ExtraDataEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1)  # Reduce size
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, bottleneck_channels, kernel_size=3, stride=2, padding=1)
        self.avgPool3d = nn.AdaptiveAvgPool3d((bottleneck_shape[0], bottleneck_shape[1], bottleneck_shape[2]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgPool3d(x)
        return x
    
class CombineWithConv(nn.Module):
    def __init__(self, bottleneck_channels):
        super(CombineWithConv, self).__init__()
        # self.extra_data_encoder = ExtraDataEncoder(extra_data_channels, bottleneck_channels)
        self.convCombination = nn.Conv3d(bottleneck_channels * 2, bottleneck_channels, kernel_size=1)

    def forward(self, bottleneck, extra_data_encoded):
        # Permute extra_data to match Conv3d input shape: (Batch, Channels, Depth, Height, Width)
        # extra_data = extra_data.permute(3, 2, 0, 1).unsqueeze(0)  # (1, Channels, Depth, Height, Width)
        
        # Encode extra data
        # extra_data_encoded = self.extra_data_encoder(extra_data)
        
        # Reshape back to (Height, Width, Channels, Depth)
        # extra_data_encoded = extra_data_encoded.squeeze(0).permute(2, 3, 1, 0)  # (Height, Width, Channels, Depth)
        
        # Ensure dimensions match
        if extra_data_encoded.shape != bottleneck.shape:
            raise ValueError(f"Shapes of bottleneck and extra_data_encoded do not match: {bottleneck.shape} vs {extra_data_encoded.shape}")
        
        # Combine bottleneck and extra data (concatenate along channel dimension)
        combined = torch.cat((bottleneck, extra_data_encoded), dim=1)  # (Height, Width, Channels*2, Depth)
        
        # Apply final convolution to adjust the number of channels
        # combined = combined.permute(3, 2, 0, 1).unsqueeze(0)  # (1, Channels*2, Depth, Height, Width)
        combined = self.convCombination(combined)
        # combined = combined.squeeze(0).permute(2, 3, 1, 0)  # (Height, Width, Channels, Depth)
        
        return combined

class VideoSwinUNetV2(nn.Module):
    def __init__(self, mainInputChannels, extraInputChannels, mainInputTimeDimension, inputHW, C=96, num_blocks=3, patch_size=(2, 4, 4), window_size=(8, 7, 7)):
        super().__init__()
        # T is main input time dimension
        self.patch_embed     = PatchEmbed3D(patch_size=patch_size, in_chans=mainInputChannels, embed_dim=C) # Output: B, C, T/patch_size[0], H/patch_size[1], W/patch_size[2]
        self.encoder         = Encoder(C, window_size=window_size)                                          # Output shape B, C*(2**num_blocks), T/patch_size[0], H/patch_size[1]/(2**num_blocks), H/patch_size[1]/(2**num_blocks)
        self.bottleneck      = BasicLayer(dim = C*(2**num_blocks), depth=2, num_heads=12, window_size=window_size, downsample=None) # No change in shape
        self.extraDataEnc    = ExtraDataEncoder(in_channels=extraInputChannels, bottleneck_channels=C*(2**num_blocks), bottleneck_shape=(mainInputTimeDimension//patch_size[0], inputHW//patch_size[1]//(2**num_blocks), inputHW//patch_size[1]//(2**num_blocks))) # Output shape same as encoder/bottleneck B, C*(2**num_blocks), T/patch_size[0], H/patch_size[1]/(2**num_blocks), H/patch_size[1]/(2**num_blocks)     
        self.bottleneckComb  = CombineWithConv(bottleneck_channels=C*(2**num_blocks))                       # Same as bottleneck
        self.decoder         = Decoder(C=C, window_size=window_size)                                        # Output shape B, C, T/patch_size[0], H/patch_size[1], W/patch_size[2]
        self.timeUpsampling1 = nn.ConvTranspose3d(in_channels=C, out_channels=C, kernel_size=(2, 1, 1), stride=(2,1,1)) # Output shape B, C, T/patch_size[0]*2, H/patch_size[1], W/patch_size[2]
        self.final_expansion = FinalPatchExpansion(dim=C)                                                   # Output shape B, C, T/patch_size[0]*2, H, W
        self.timeUpsampling2 = nn.ConvTranspose3d(in_channels=C, out_channels=C, kernel_size=(2, 1, 1), stride=(2,1,1)) # Output shape B, C, T/patch_size[0]*2*2, H, W
        self.head            = nn.Conv3d(in_channels=C, out_channels=5, kernel_size=1, padding='same')      # Output shape B, 5, T/patch_size[0]*2*2, H, W
        

    def forward(self, x, xExtra):

        # Input shape B,C,T,H,W
        # startTime = time.time()
        x = self.patch_embed(x)
        # print("Patch embedding output shape {}".format(x.shape))
        # endTime = time.time()
        # print("Patch embedding time {}".format(endTime - startTime))

        # Extra data processing
        xExtra = self.extraDataEnc(xExtra)
        # print("Extra data encoder output shape {}".format(xExtra.shape))

        # startTime = time.time()
        x,skip_ftrs = self.encoder(x)
        # print("Encoder output shape {}".format(x.shape))
        # for s in skip_ftrs:
        #     print("Skip feature output shape {}".format(s.shape))
        # endTime = time.time()
        # print("Encoder time {}".format(endTime - startTime))

        # startTime = time.time()
        x = self.bottleneck(x)
        # print("Bottleneck output shape {}".format(x.shape))
        # endTime = time.time()
        # print("Bottleneck time {}".format(endTime - startTime))

        # Combine bottleneck output and processed extra data
        x = self.bottleneckComb(x, xExtra)
        # print("After combination output shape {}".format(x.shape))

        # startTime = time.time()
        x = self.decoder(x, skip_ftrs[::-1])
        # print("Decoder output shape {}".format(x.shape))
        # endTime = time.time()
        # print("Decoder time {}".format(endTime - startTime))

        # startTime = time.time()
        # for time upsampling from 5 to 10 frames
        x = self.timeUpsampling1(x)
        # print("Time upsampling 1 output shape {}".format(x.shape))
        # endTime = time.time()
        # print("Time upsampling 1 time {}".format(endTime - startTime))

        # startTime = time.time()
        x = self.final_expansion(x)
        # print("Final expansion output shape {}".format(x.shape))
        # endTime = time.time()
        # print("Final exxpansion time {}".format(endTime - startTime))

        # startTime = time.time()
        # for time upsampling from 10 to 20 frames
        x = self.timeUpsampling2(x)
        # print("Time upsampling 2 output shape {}".format(x.shape))
        # endTime = time.time()
        # print("Time upsampling 2 time {}".format(endTime - startTime))

        # startTime = time.time()
        x = self.head(x)
        # print("Head output shape {}".format(x.shape))
        # endTime = time.time()
        # print("Head time {}".format(endTime - startTime))

        return x
    
# def preprocess(X, Y):

#     X["highresdynamic"] = torch.permute(X["highresdynamic"], (0, 3, 4, 1, 2))

#     X["highresstatic"] = torch.unsqueeze(X["highresstatic"], -1)
#     X["highresstatic"] = torch.permute(X["highresstatic"], (0, 3, 4, 1, 2))
#     X["highresstatic"] = F.interpolate(X["highresstatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4]))

#     X["mesodynamic"] = torch.permute(X["mesodynamic"], (0, 3, 4, 1, 2))
#     X["mesodynamic"] = F.interpolate(X["mesodynamic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4]))

#     X["mesostatic"] = torch.unsqueeze(X["mesostatic"], -1)
#     X["mesostatic"] = torch.permute(X["mesostatic"], (0, 3, 4, 1, 2))
#     X["mesostatic"] = F.interpolate(X["mesostatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4]))

#     x = torch.cat((X["highresdynamic"], X["highresstatic"], X["mesodynamic"], X["mesostatic"]), 1)

#     y = torch.permute(Y["highresdynamic"], ((0, 3, 4, 1, 2)))

#     return x, y


if __name__ == "__main__":

    startTime = time.time()
    model = VideoSwinUNetV2(mainInputChannels=12, extraInputChannels=5, mainInputTimeDimension=10, inputHW=128, C=96, window_size=(5,7,7)).to(torch.device('cuda'))
    endTime = time.time()
    print("Model creation time {}".format(endTime - startTime))

    model.train()

    totalParams = sum(p.numel() for p in model.parameters())

    startTime = time.time()
    x = torch.rand(16, 12, 10, 128, 128).to(torch.device('cuda')) # B, C, T, H, W
    xExtra = torch.rand(16, 5, 20, 128, 128).to(torch.device('cuda')) # B, C, T, H, W
    endTime = time.time()
    print("Dummy data creation time {}".format(endTime - startTime))

    # for i in range(20):
    #     _ = torch.randn(1).cuda()        
    #     startTime = time.time()
    #     y = model(x, xExtra)
    #     endTime = time.time()
    #     print("Pass time {}".format(endTime - startTime))

    y = model(x, xExtra)

    print("Output shape {}".format(y.shape))

    print("Number of model parameters {}".format(totalParams))

    # embed = PatchEmbed3D(patch_size=(2, 4, 4), in_chans=3, embed_dim=C)
    # y1 = embed(x)
    # print("y1: {}".format(y1.shape))

    # encoder = Encoder(C, window_size=(8, 7, 7))
    # y2, skip_features = encoder(y1)
    # print("y2: {}".format(y2.shape))

    # bottleneck = BasicLayer(dim = C*(2**3), depth=2, num_heads=12, window_size=(8, 7, 7), downsample=None)
    # y3 = bottleneck(y2)
    # print("y3: {}".format(y3.shape))

    # decoder = Decoder(C, window_size=(8, 7, 7))
    # y4 = decoder(y3, skip_features[::-1])
    # print("y4: {}".format(y4.shape))

    # # to upsample time
    # ConvT1 = nn.ConvTranspose3d(96, 96, kernel_size=(2, 1, 1), stride=(2,1,1))
    # y5 = ConvT1(y4)
    # print("y5: {}".format(y5.shape))

    # final_exp = FinalPatchExpansion(C)
    # y6 = final_exp(y5)
    # print("y6: {}".format(y6.shape))

    # # to upsample time
    # ConvT2 = nn.ConvTranspose3d(96, 96, kernel_size=(2, 1, 1), stride=(2,1,1))
    # y7 = ConvT2(y6)
    # print("y7: {}".format(y7.shape))

    # head = nn.Conv3d(C, 3, 1,padding='same')
    # y8 = head(y7)
    # print("y8: {}".format(y8.shape))
    # None

    print("Done!")

