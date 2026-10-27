"""Eval-only interventions. No training/model files or parameter values change."""
from contextlib import contextmanager
import types

import torch
from torch.nn import functional as F

MODES = ('off', 'zero_local_value', 'on', 'cls_only', 'patch_only')


def source_statistics(q, kd, vd, kl, vl, bias, scale, mode):
    """FP32, pre-dropout, per-image/head/query-region sufficient statistics.

    logZ_local_minus_backbone excludes the source bias; the effective logZ
    difference includes it. A_D/A_L are conditional, before output projection.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        sd = (q.detach().float() @ kd.detach().float().transpose(-2, -1)) * scale
        sl = (q.detach().float() @ kl.detach().float().transpose(-2, -1)) * scale
        zd, zl = sd.logsumexp(-1), sl.logsumexp(-1)
        effective = zl - zd + bias.detach().float().squeeze(-1)
        rho = effective.sigmoid()
        allowed = torch.ones_like(rho)
        if mode == 'cls_only':
            allowed[:, :, 1:] = 0
        elif mode == 'patch_only':
            allowed[:, :, :1] = 0
        rho = rho * allowed
        ad = sd.softmax(-1) @ vd.detach().float()
        al = sl.softmax(-1) @ vl.detach().float()
        # ZeroV's actual local response is zero; also retain the unmodified
        # conditional response to distinguish removal from candidate content.
        actual_al = torch.zeros_like(al) if mode == 'zero_local_value' else al
        metrics = {
            'rho': rho,
            'local_allowed': allowed,
            'logZ_local_minus_backbone': zl - zd,
            'effective_logZ_difference': effective,
            'A_D_norm': ad.norm(dim=-1),
            'A_L_norm': actual_al.norm(dim=-1),
            'A_L_original_norm': al.norm(dim=-1),
            'A_L_A_D_cosine': F.cosine_similarity(actual_al, ad, dim=-1),
            'A_L_original_A_D_cosine': F.cosine_similarity(al, ad, dim=-1),
        }
        return {region: {key: value[:, :, selection].mean(-1).cpu()
                         for key, value in metrics.items()}
                for region, selection in [('CLS', slice(0, 1)), ('Patch', slice(1, None))]}


@contextmanager
def inference_probe(model, mode, collect_layers=False, collect_local=False,
                    collect_attention=False):
    """Temporarily bind only Block 7 Attention.forward; restore even on failure.

    This intentionally runs on one unwrapped model. No global hooks, optimizer,
    parameter copies, source-bias changes, or training call sites are involved.
    """
    if mode not in MODES:
        raise ValueError('Unknown inference intervention: ' + mode)
    if any(module.training for module in model.modules()):
        raise ValueError('SFTC interventions require the entire model in eval()')
    if not getattr(model, 'sftc_enabled', False):
        raise ValueError('A full SFTC model/checkpoint is required')
    base, captured, handles = model.base, {}, []
    attention = base.blocks[base.sftc_block_index].attn
    had_forward = 'forward' in attention.__dict__
    old_forward = attention.__dict__.get('forward')
    old_collect = base.sftc.collect_stats

    def intervened(self, x, local_context=None, stats_callback=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Interventions are inference-only; use inference_mode/no_grad')
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kd, vd = qkv[0], qkv[1], qkv[2]
        logits = (q @ kd.transpose(-2, -1)) * self.scale
        values = vd
        if mode != 'off':
            if local_context is None:
                raise RuntimeError('Enabled intervention received no local K/V')
            kl, vl, bias = local_context
            local_logits = (q @ kl.transpose(-2, -1)) * self.scale
            local_logits = local_logits + bias.to(local_logits.dtype)
            # Mask by QUERY; retain the original backbone candidates and their
            # logits. Disallowed rows normalize over exactly the original K/V.
            if mode == 'cls_only':
                local_logits = local_logits.clone()
                local_logits[:, :, 1:] = -torch.inf
            elif mode == 'patch_only':
                local_logits = local_logits.clone()
                local_logits[:, :, :1] = -torch.inf
            logits = torch.cat([logits, local_logits], dim=-1)
            values = torch.cat([vd, torch.zeros_like(vl) if mode == 'zero_local_value' else vl], dim=-2)
            if collect_attention:
                captured['attention'] = source_statistics(q, kd, vd, kl, vl, bias, self.scale, mode)
        probs = logits.softmax(dim=-1)
        if stats_callback is not None:
            stats_callback(probs, n)
        output = (self.attn_drop(probs) @ values).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(output))

    def save(name, selector=lambda value: value):
        def hook(module, inputs, output):
            captured[name] = selector(output).detach().float().cpu().clone()
        return hook

    try:
        attention.forward = types.MethodType(intervened, attention)
        base.sftc.collect_stats = False
        if collect_layers:
            def save_input(module, inputs):
                captured['block7_input'] = inputs[0].detach().float().cpu().clone()
            handles.append(base.blocks[6].register_forward_pre_hook(save_input))
            handles.append(attention.register_forward_hook(save('block7_attention_post_projection')))
            handles.append(base.blocks[6].register_forward_hook(save('block7_output')))
            for index in (7, 9, 11):
                handles.append(base.blocks[index].register_forward_hook(save('block{}_cls'.format(index + 1), lambda v: v[:, 0])))
            handles.append(base.norm.register_forward_hook(save('final_norm_cls', lambda v: v[:, 0])))
        if collect_local:
            handles.append(base.sftc.encoder.fusion.register_forward_hook(save('cnn_fused', lambda v: v.mean(dim=(2, 3)))))
            handles.append(base.sftc.local_v.register_forward_hook(save('local_value', lambda v: v.mean(dim=1))))
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        if had_forward:
            attention.forward = old_forward
        else:
            del attention.forward
        base.sftc.collect_stats = old_collect


def layer_views(captured):
    views = {}
    for key in ('block7_input', 'block7_attention_post_projection', 'block7_output'):
        views[key + '_cls'] = captured[key][:, 0]
        views[key + '_patch'] = captured[key][:, 1:].flatten(1)
    views.update({key: captured[key] for key in ('block8_cls', 'block10_cls', 'block12_cls', 'final_norm_cls')})
    return views
