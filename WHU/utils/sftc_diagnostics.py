"""Sparse, rank-local diagnostics. No distributed collectives or training losses."""
import json
import os
import torch
from torch.nn import functional as F


def gradient_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt().item() if values else 0.


class SFTCDiagnostics:
    def __init__(self, model, output_dir, logger):
        self.source = model.base.sftc
        self.logger = logger
        self.path = os.path.join(output_dir or '.', 'sftc_diagnostics.jsonl')
        self.groups = {
            'cnn': list(self.source.encoder.parameters()),
            'kv': list(self.source.local_k.parameters()) + list(self.source.local_v.parameters()),
        }

    @torch.no_grad()
    def before_step(self):
        snapshot = {name: [p.detach().clone() for p in params] for name, params in self.groups.items()}
        norms = {name: gradient_norm(params) for name, params in self.groups.items()}
        return snapshot, norms

    @torch.no_grad()
    def after_step(self, epoch, iteration, optimizer, before, total_norm, clip_coefficient, skipped):
        snapshot, norms = before
        updates = {}
        for name, params in self.groups.items():
            delta = torch.stack([(p.detach().float() - old.float()).square().sum()
                                 for p, old in zip(params, snapshot[name])]).sum().sqrt().item()
            magnitude = torch.stack([old.float().square().sum() for old in snapshot[name]]).sum().sqrt().item()
            updates[name] = {'l2': delta, 'relative_l2': delta / max(magnitude, 1e-12)}
        row = {'kind': 'train', 'epoch': epoch, 'iteration': iteration,
               'rank': 0, 'bias_fixed': self.source.bias_fixed,
               'source_bias': self.source.source_bias.detach().cpu().tolist(),
               'gradient_pre_clip': norms, 'global_gradient_pre_clip': total_norm,
               'clip_coefficient': clip_coefficient, 'amp_step_skipped': skipped,
               'updates': updates,
               'new_lr': {g['name']: g['lr'] for g in optimizer.param_groups if 'name' in g}}
        stats = self.source.last_stats
        if stats is not None:
            count = stats['count'].cpu()
            row['attention_shape'] = stats['attention_shape']
            row['rho_local'] = {}
            for index, name in enumerate(('RGB', 'NIR', 'TIR')):
                if count[index] > 0:
                    row['rho_local'][name] = {
                        'cls_per_head': (stats['cls_sum'][index].cpu() / count[index]).tolist(),
                        'patch_per_head': (stats['patch_sum'][index].cpu() / count[index]).tolist()}
        self.write(row)
        self.logger.info('SFTC epoch=%d iter=%d fixed=%s grad(CNN/KV)=%s '
                         'global_norm=%.4g clip=%.4g updates=%s rho=%s',
                         epoch, iteration, row['bias_fixed'], norms, total_norm,
                         clip_coefficient, updates, row.get('rho_local', {}))

    def write(self, row):
        with open(self.path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


@torch.no_grad()
def cls_source_difference(feature_on, feature_off):
    cosine = 1. - F.cosine_similarity(feature_on.float(), feature_off.float(), dim=1)
    relative = (feature_on.float() - feature_off.float()).norm(dim=1) / feature_on.float().norm(dim=1).clamp_min(1e-12)
    return {'cosine_distance_mean': cosine.mean().item(),
            'relative_l2_mean': relative.mean().item()}


def write_cls_diagnostic(output_dir, row):
    # Validation assigns modalities to different ranks; rank>0 has a null
    # logger in UAD. Separate files keep every sensor's evidence without adding
    # a collective or simultaneous writes to a shared file.
    path = os.path.join(output_dir or '.', 'sftc_cls_diagnostics_rank{}.jsonl'.format(row['rank']))
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
