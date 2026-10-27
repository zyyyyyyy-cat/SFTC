"""Meaningful dataset-free intervention, metric, export and restoration checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg as default_cfg
from model import make_model
from model.backbones.vit_pytorch import Attention, Block
from utils.sftc_inference_probe import inference_probe, layer_views
from utils.sftc_probe_metrics import evaluate_features, query_comparison, write_csv
from utils.metrics import compute_indices_chunked, eval_func
from diagnose_sftc import balanced_samples, state_digest, extract_features, propagation, check_existing_paths


def attention_checks():
    model = nn.Module()
    model.sftc_enabled = True
    model.base = nn.Module()
    model.base.sftc_block_index = 6
    model.base.blocks = nn.ModuleList([Block(32, 4, qkv_bias=True) for _ in range(12)])
    model.base.sftc = nn.Module()
    model.base.sftc.collect_stats = True
    model.eval()
    attention = model.base.blocks[6].attn
    x = torch.randn(2, 5, 32)
    kl, vl = torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8)
    bias = torch.full((2, 4, 1, 1), -2.)
    before = state_digest(model)
    with torch.inference_mode():
        original_on, original_off = attention(x, (kl, vl, bias)), attention(x)
        qkv = attention.qkv(x).reshape(2, 5, 3, 4, 8).permute(2, 0, 3, 1, 4)
        q, kd, vd = qkv
        sd = q @ kd.transpose(-2, -1) * attention.scale
        sl = q @ kl.transpose(-2, -1) * attention.scale + bias
        probs = torch.cat([sd, sl], -1).softmax(-1)
        expected_zero = attention.proj((probs[..., :5] @ vd).transpose(1, 2).reshape(2, 5, 32))
        for mode in ('on', 'off', 'zero_local_value', 'cls_only', 'patch_only'):
            with inference_probe(model, mode, collect_attention=mode != 'off') as captured:
                out = attention(x, None if mode == 'off' else (kl, vl, bias))
            if mode == 'on':
                torch.testing.assert_close(out, original_on, rtol=0, atol=0)
                stats = captured['attention']['CLS']
                torch.testing.assert_close(stats['rho'], probs[:, :, 0, 5:].sum(-1))
            elif mode == 'off':
                torch.testing.assert_close(out, original_off, rtol=0, atol=0)
            elif mode == 'zero_local_value':
                torch.testing.assert_close(out, expected_zero, atol=1e-6, rtol=1e-6)
                assert not torch.allclose(out, original_off)
                assert torch.count_nonzero(captured['attention']['CLS']['A_L_norm']) == 0
            elif mode == 'cls_only':
                torch.testing.assert_close(out[:, :1], original_on[:, :1], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(out[:, 1:], original_off[:, 1:], atol=1e-6, rtol=1e-6)
                assert torch.count_nonzero(captured['attention']['Patch']['rho']) == 0
            else:
                torch.testing.assert_close(out[:, :1], original_off[:, :1], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(out[:, 1:], original_on[:, 1:], atol=1e-6, rtol=1e-6)
                assert torch.count_nonzero(captured['attention']['CLS']['rho']) == 0
        assert 'forward' not in attention.__dict__ and model.base.sftc.collect_stats
        try:
            with inference_probe(model, 'on'):
                raise RuntimeError('intentional failure')
        except RuntimeError:
            pass
        torch.testing.assert_close(attention(x, (kl, vl, bias)), original_on, rtol=0, atol=0)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with inference_probe(model, 'on', collect_attention=True) as captured:
                attention(x, (kl, vl, bias))
        assert captured['attention']['CLS']['rho'].dtype == torch.float32
    assert state_digest(model) == before
    model.train()
    try:
        with inference_probe(model, 'on'):
            pass
        raise AssertionError('Training must be rejected')
    except ValueError:
        pass
    return 'exact on/off; ZeroV independent formula; query masks; FP32 stats; restoration and eval guard pass'


def metrics_checks():
    metadata = []
    for split, count in [('query', 4), ('gallery', 15)]:
        for i in range(count):
            metadata.append(dict(sample_index=len(metadata), split=split, pid=i % 3, camid=(0 if split == 'query' else i % 7),
                                 sensor_id=i % 3, dataset_modality_id=i % 3 + 1, modality=('RGB', 'NIR', 'TIR')[i % 3],
                                 platform='G' if (0 if split == 'query' else i % 7) < 5 else 'A', image_path=str(i) + split))
    metadata[3]['pid'] = 999  # No valid positive: must retain a blank Query row.
    features = torch.randn(len(metadata), 8)
    features = F.normalize(features, dim=1)
    # An impostor from the Query camera is closer than its positive. The
    # inherited WHU rule must exclude it too, not just same-ID/same-camera.
    features[4] = features[0]
    a, queries, stats = evaluate_features(features.numpy(), metadata, 4, 'on', 1, similarity=True)
    b, query_b, _ = evaluate_features(features.numpy(), metadata, 4, 'off', 3)
    assert a[:12] == [dict(row, mode='on') for row in b]
    assert queries == query_b and queries[3]['AP'] is None and not queries[3]['valid']
    qp, gp = np.array([m['pid'] for m in metadata[:4]]), np.array([m['pid'] for m in metadata[4:]])
    qc, gc = np.array([m['camid'] for m in metadata[:4]]), np.array([m['camid'] for m in metadata[4:]])
    indices = compute_indices_chunked(features[:4], features[4:], top_k=0, q_chunk_size=2)
    cmc, ap, inp = eval_func(indices, qp, gp, qc, gc, max_rank=10, return_minp=True)
    np.testing.assert_allclose([a[0]['mAP'], a[0]['mINP'], a[0]['R1'], a[0]['R5'], a[0]['R10']],
                               100 * np.array([ap, inp, cmc[0], cmc[4], cmc[9]]), rtol=1e-6)
    for row in stats:
        assert sum(json.loads(row['histogram_counts'])) == row['pair_count']
    comparison = query_comparison(metadata, 4, {'on': queries, 'off': query_b}, ['on', 'off'])
    assert [r['query_index'] for r in comparison] == list(range(4))
    assert comparison[3]['AP_on'] is None
    assert balanced_samples(metadata, 2) == balanced_samples(metadata, 2)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'empty.csv'
        write_csv(path, [], ['query_index', 'transition'])
        assert 'query_index,transition' in path.read_text(encoding='utf-8-sig')
    return 'official AP/CMC/mINP agreement; same-camera negative removal; invalid Query alignment; chunk invariance and histograms pass'


def model_checks():
    cfg = default_cfg.clone()
    cfg.merge_from_file(str(ROOT / 'configs/UAD_SFTC_V2_1.yml'))
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    model = make_model(cfg, 2, 7, 3).eval().requires_grad_(False)
    before = state_digest(model)
    metadata = [dict(sample_index=i, split='query' if i < 3 else 'gallery', pid=i % 2,
                     camid=0 if i < 3 else 5, sensor_id=i % 3, dataset_modality_id=i % 3 + 1,
                     modality=('RGB', 'NIR', 'TIR')[i % 3], platform='G' if i < 3 else 'A',
                     image_path='synthetic-' + str(i)) for i in range(6)]
    images = torch.randn(6, 3, 256, 128)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, torch.arange(6)), batch_size=3)
    checks = check_existing_paths(model, loader, metadata, torch.device('cpu'))
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for mode in ('on', 'off', 'zero_local_value', 'cls_only', 'patch_only'):
            extract_features(model, loader, metadata, mode, directory, torch.device('cpu'), local=mode == 'on', logger=lambda x: None)
            assert np.load(directory / (mode + '.npy')).shape == (6, 768)
        assert np.load(directory / 'cnn_fused.npy').shape == (6, 256)
        assert np.load(directory / 'local_value.npy').shape == (6, 768)
        propagation(model, loader, metadata, torch.device('cpu'), directory)
        import csv
        with (directory / '03_layer_propagation.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 6 * 4 * 10
        assert all(float(row['relative_l2']) == 0 for row in rows if row['location'].startswith('block7_input'))
    assert state_digest(model) == before
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    return dict(checks, all_five_modes_and_two_local_features='pass', layer_rows=240, complete_state_unchanged=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    results = dict(attention=attention_checks(), metrics=metrics_checks(), model=model_checks())
    path = ROOT / 'docs/sftc_probe_validation.json'
    path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
