"""Profile UAD/SFTC without data or downloaded weights; no claim about accuracy."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg as defaults
from model import make_model
from model.backbones.vit_pytorch import Attention
from model.backbones.sftc_local import BlurDownsample


@torch.no_grad()
def measure(enabled, args):
    cfg = defaults.clone()
    cfg.merge_from_file(str(ROOT / 'configs' / ('UAD_SFTC_V2_1.yml' if enabled else 'UAD.yml')))
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    model = make_model(cfg, args.identities, 7, 3).to(args.device).eval()
    images = torch.randn(args.batch_size, 3, 256, 128, device=args.device)
    ids = torch.arange(args.batch_size, device=args.device) % 3
    macs = {'conv_linear': 0, 'attention_matmul': 0, 'fixed_blur': 0}

    def hook(module, inputs, output):
        if isinstance(module, nn.Linear):
            macs['conv_linear'] += output.numel() * module.in_features
        elif isinstance(module, nn.Conv2d):
            macs['conv_linear'] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, Attention):
            b, n, c = inputs[0].shape
            local = inputs[1] if len(inputs) > 1 else None
            keys = n + (local[0].shape[-2] if local is not None else 0)
            macs['attention_matmul'] += 2 * b * n * keys * c
        elif isinstance(module, BlurDownsample):
            macs['fixed_blur'] += output.numel() * 9

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Linear, nn.Conv2d, Attention, BlurDownsample))]
    amp_dtype = torch.float16 if args.device.startswith('cuda') else torch.bfloat16

    def forward():
        with torch.autocast(device_type='cuda' if args.device.startswith('cuda') else 'cpu',
                            enabled=args.amp, dtype=amp_dtype):
            return model(images, mode=1, modal_ids=ids if enabled else None)

    feature = forward()
    if feature.shape != (args.batch_size, 768) or not torch.isfinite(feature).all():
        raise RuntimeError('Non-finite or incorrectly shaped retrieval feature')
    for handle in handles:
        handle.remove()
    for _ in range(args.warmup):
        forward()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.iterations):
        forward()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {'model': 'SFTC' if enabled else 'UAD',
              'parameters_total': sum(p.numel() for p in model.parameters()),
              'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'macs_per_image': {k: v // args.batch_size for k, v in macs.items()},
              'macs_per_image_total': sum(macs.values()) // args.batch_size,
              'flops_per_image_conv_linear_matmul': 2 * sum(macs.values()) // args.batch_size,
              'inference_ms_per_batch': elapsed * 1000 / args.iterations,
              'inference_ms_per_image': elapsed * 1000 / args.iterations / args.batch_size,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None}
    del model, images
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--identities', type=int, default=500)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error('batch size/iterations must be positive, warmup nonnegative')
    torch.set_num_threads(args.threads)
    result = {'runtime': {'torch': torch.__version__, 'device': args.device, 'amp': args.amp,
                          'batch_size': args.batch_size, 'threads': args.threads,
                          'identities': args.identities, 'iterations': args.iterations, 'warmup': args.warmup},
              'count_convention': '1 MAC=2 FLOPs; conv, linear, QK/AV matmul and fixed blur only; excludes norm/activation/softmax/elementwise operations',
              'results': [measure(False, args), measure(True, args)]}
    output = json.dumps(result, indent=2)
    if args.report:
        Path(args.report).write_text(output + '\n', encoding='utf-8')
    print(output)
