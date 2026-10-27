"""SFTC-V2.1 local source; task adaptation, not an MRFP/CTI/HTM replica."""
import torch
from torch import nn
from torch.nn import functional as F


def sensor_order(modalities):
    """Map folder order to stable RGB/NIR/TIR parameter rows."""
    aliases = {'rgb': 0, 'nir': 1, 'ir': 1, 'tir': 2, 'thermal': 2}
    try:
        order = tuple(aliases[name.strip().lower()] for name in modalities)
    except KeyError as exc:
        raise ValueError('SFTC requires RGB, IR/NIR, Thermal/TIR folders') from exc
    if len(order) != 3 or set(order) != {0, 1, 2}:
        raise ValueError('SFTC requires exactly one folder for each RGB/NIR/TIR sensor')
    return order


def validate_sftc_config(cfg):
    expected = {'BLOCK_INDEX': 6, 'LOCAL_GRID': (16, 8),
                'CNN_CHANNELS': (32, 64, 128), 'EMBED_DIM': 768}
    for key, value in expected.items():
        actual = getattr(cfg.SFTC, key)
        if isinstance(value, tuple):
            actual = tuple(actual)
        if actual != value:
            raise ValueError('SFTC-V2.1 requires {}={}'.format(key, value))
    if (cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_in'
            or tuple(cfg.MODEL.STRIDE_SIZE) != (16, 16)
            or tuple(cfg.INPUT.SIZE_TRAIN) != (256, 128)
            or tuple(cfg.INPUT.SIZE_TEST) != (256, 128)):
        raise ValueError('SFTC-V2.1 requires ViT-B/16, stride 16 and 256x128 inputs')
    if cfg.TEST.NECK_FEAT != 'before':
        raise ValueError('SFTC-V2.1 retrieval requires TEST.NECK_FEAT=before')
    if cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW:
        raise ValueError('SFTC-V2.1 follows UAD with camera/view SIE disabled')
    if cfg.SFTC.SOURCE_BIAS_FIXED_EPOCHS < 0 or cfg.SFTC.NEW_LR_MULT <= 0:
        raise ValueError('Invalid SFTC freeze duration or learning-rate multiplier')
    if cfg.SOLVER.GRAD_CLIP_NORM <= 0:
        raise ValueError('GRAD_CLIP_NORM must be positive')
    sensor_order(cfg.DATASETS.MODALITIES)


class BlurDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
        self.register_buffer('kernel', kernel[None, None], persistent=False)

    def forward(self, x):
        kernel = self.kernel.to(dtype=x.dtype).expand(x.shape[1], 1, 3, 3)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'),
                        kernel, stride=2, groups=x.shape[1])


def sincos_2d(height, width, dim):
    # Each axis: [sin(positions * omega), cos(positions * omega)].
    # Width varies fastest, matching flatten(2) of [B,C,H,W].
    if dim % 4:
        raise ValueError('2D sinusoidal dimension must be divisible by four')
    omega = 1. / (10000 ** (torch.arange(dim // 4).float() / (dim // 4)))
    y = torch.arange(height).repeat_interleave(width).float()[:, None] * omega
    x = torch.arange(width).repeat(height).float()[:, None] * omega
    return torch.cat([y.sin(), y.cos(), x.sin(), x.cos()], dim=1)[None]


class SFTCLocalEncoder(nn.Module):
    def __init__(self, pos_scale=0.1):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.stem_norm = nn.GroupNorm(8, 32, affine=False)
        self.modal_gamma = nn.Parameter(torch.ones(3, 32))
        self.modal_beta = nn.Parameter(torch.zeros(3, 32))
        self.act = nn.GELU()
        self.stage1 = self._stage(32, 64, 8)
        self.stage2 = self._stage(64, 128, 16)
        self.blur = BlurDownsample()
        self.fusion = nn.Sequential(nn.Conv2d(192, 256, 1, bias=False),
                                    nn.GroupNorm(32, 256), nn.GELU())
        self.projection = nn.Conv2d(256, 768, 1, bias=False)
        self.norm = nn.LayerNorm(768)
        self.pos_scale = pos_scale
        self.register_buffer('pos_embed', sincos_2d(16, 8, 768), persistent=False)
        self.reset_parameters()

    @staticmethod
    def _stage(in_channels, out_channels, groups):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1,
                      groups=in_channels, bias=False),
            nn.GroupNorm(8, in_channels), nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.GELU())

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)) and module.weight is not None:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)

    def forward(self, images, modal_ids):
        if images.shape[1:] != (3, 256, 128):
            raise ValueError('SFTC expects [B,3,256,128] images')
        x = self.stem_norm(self.stem_conv(images))
        # embedding rejects negative IDs (tensor indexing would accept -1).
        gamma = F.embedding(modal_ids, self.modal_gamma)[:, :, None, None]
        beta = F.embedding(modal_ids, self.modal_beta)[:, :, None, None]
        x = self.act(x * gamma + beta)
        f4 = self.stage1(x)
        f8 = self.stage2(f4)
        fused = self.fusion(torch.cat([self.blur(self.blur(f4)), self.blur(f8)], dim=1))
        tokens = self.projection(fused).flatten(2).transpose(1, 2)
        return self.norm(tokens + self.pos_scale * self.pos_embed.to(dtype=tokens.dtype))


class SFTCSource(nn.Module):
    def __init__(self, qkv, num_heads=12, pos_scale=0.1, bias_init=-2., fixed_epochs=5):
        super().__init__()
        self.encoder = SFTCLocalEncoder(pos_scale)
        self.local_k = nn.Linear(768, 768, bias=qkv.bias is not None)
        self.local_v = nn.Linear(768, 768, bias=qkv.bias is not None)
        self.source_bias = nn.Parameter(torch.full((3, num_heads), float(bias_init)))
        self.num_heads = num_heads
        self.bias_init = bias_init
        self.fixed_epochs = fixed_epochs
        self.bias_fixed = fixed_epochs > 0
        self.collect_stats = False
        self.last_stats = None
        # Called once, after ImageNet weights were loaded. Checkpoint loading
        # never calls this method and never recopies the backbone K/V.
        self.initialize_kv(qkv)

    @torch.no_grad()
    def initialize_kv(self, qkv):
        self.local_k.weight.copy_(qkv.weight[768:1536])
        self.local_v.weight.copy_(qkv.weight[1536:2304])
        if qkv.bias is not None:
            self.local_k.bias.copy_(qkv.bias[768:1536])
            self.local_v.bias.copy_(qkv.bias[1536:2304])

    @torch.no_grad()
    def set_epoch(self, epoch):
        if epoch < 1:
            raise ValueError('Epoch is one-based')
        self.bias_fixed = epoch <= self.fixed_epochs
        if self.bias_fixed:
            self.source_bias.fill_(self.bias_init)

    @torch.no_grad()
    def suppress_bias_gradient(self, optimizer):
        if self.bias_fixed:
            if self.source_bias.grad is not None:
                self.source_bias.grad.zero_()
            # Also handles stale momentum if an optimizer state is restored.
            optimizer.state.pop(self.source_bias, None)
            self.source_bias.fill_(self.bias_init)

    def forward(self, images, modal_ids):
        if modal_ids is None or modal_ids.shape != (images.shape[0],) or modal_ids.dtype != torch.long:
            raise ValueError('SFTC requires a long sensor-ID tensor [B] (RGB=0,NIR=1,TIR=2)')
        if modal_ids.device != images.device:
            raise ValueError('Sensor IDs and images must share a device')
        local = self.encoder(images, modal_ids)
        b, n, c = local.shape
        k = self.local_k(local).reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = self.local_v(local).reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        # Zero-valued graph edge retains DDP participation during fixed epochs.
        bias = (self.source_bias * 0. + self.bias_init
                if self.training and self.bias_fixed else self.source_bias)
        return k, v, F.embedding(modal_ids, bias)[:, :, None, None]

    @torch.no_grad()
    def record_attention(self, probs, num_backbone_tokens, modal_ids):
        mass = probs.detach()[..., num_backbone_tokens:].float().sum(-1)
        membership = F.one_hot(modal_ids, num_classes=3).float().t()
        self.last_stats = {
            'count': membership.sum(1),
            'cls_sum': membership @ mass[:, :, 0],
            'patch_sum': membership @ mass[:, :, 1:].mean(-1),
            'attention_shape': list(probs.shape),
        }
