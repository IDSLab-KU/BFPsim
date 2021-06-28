'''DenseNet model in PyTorch.
Implemented by isaaccorley, https://github.com/isaaccorley/mlp-mixer-pytorch/mlp_mixer/
Edited some layer configurations to match blocking
'''

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


from functions import SetLinearLayer

class PatchEmbeddings(nn.Module):

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        channels: int
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c")
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GlobalAveragePooling(nn.Module):

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class Classifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.model.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPBlock(nn.Module):

    def __init__(self, bf_conf, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            SetLinearLayer("0", bf_conf, input_dim, hidden_dim),
            # nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            SetLinearLayer("1", bf_conf, hidden_dim, input_dim),
            # nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MixerBlock(nn.Module):

    def __init__(
        self,
        bf_conf,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        
        tm = bf_conf["token_mixing"] if "token_mixing" in bf_conf else dict()
        cm = bf_conf["channel_mixing"] if "channel_mixing" in bf_conf else dict()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(tm , num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(cm,num_channels, channels_hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class MLPMixer(nn.Module):

    def __init__(
        self,
        bf_conf,
        num_classes: int,
        image_size: int = 256,
        channels: int = 3,
        patch_size: int = 32,
        num_layers: int = 8,
        hidden_dim: int = 512,
        tokens_hidden_dim: int = 256,
        channels_hidden_dim: int = 2048
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.embed = PatchEmbeddings(patch_size, hidden_dim, channels)
        layers = []
        for i in range(num_layers):
            b = bf_conf[str(i)] if str(i) in bf_conf else dict()
            layers.append(MixerBlock(
                b,
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            ))
                
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = GlobalAveragePooling(dim=1)
        self.classifier = Classifier(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.embed(x)           # [b, p, c]
        x = self.layers(x)          # [b, p, c]
        x = self.norm(x)            # [b, p, c]
        x = self.pool(x)            # [b, c]
        x = self.classifier(x)      # [b, num_classes]
        return x

def mlp_mixer_s16(bf_conf, num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=8, hidden_dim=512,
                  tokens_hidden_dim=256, channels_hidden_dim=2048)
    return MLPMixer(bf_conf, num_classes, image_size, channels, **params)

def mlp_mixer_s32(bf_conf, num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=8, hidden_dim=512,
                  tokens_hidden_dim=256, channels_hidden_dim=2048)
    return MLPMixer(bf_conf, num_classes, image_size, channels, **params)

def mlp_mixer_b16(bf_conf, num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=12, hidden_dim=768,
                  tokens_hidden_dim=384, channels_hidden_dim=3072)
    return MLPMixer(bf_conf, num_classes, image_size, channels, **params)

def mlp_mixer_b32(bf_conf, num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=12, hidden_dim=768,
                  tokens_hidden_dim=384, channels_hidden_dim=3072)
    return MLPMixer(bf_conf, num_classes, image_size, channels, **params)

def mlp_mixer_l16(bf_conf, num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=24, hidden_dim=1024,
                  tokens_hidden_dim=512, channels_hidden_dim=4096)
    return MLPMixer(bf_conf, num_classes, image_size, channels, **params)

def mlp_mixer_l32(bf_conf, num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=24, hidden_dim=1024,
                  tokens_hidden_dim=512, channels_hidden_dim=4096)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_h14(bf_conf, num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=14, num_layers=32, hidden_dim=1280,
                  tokens_hidden_dim=640, channels_hidden_dim=5120)
    return MLPMixer(bf_conf, num_classes, image_size, channels, **params)