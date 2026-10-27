"""Dataset-free SFTC checks. Run from WHU: python tests/check_sftc.py [--ddp]."""
import argparse
import gc
import importlib.util
import json
import logging
from pathlib import Path
import socket
import sys
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg as default_cfg
from model import make_model
from model.backbones.vit_pytorch import Attention, Block
from model.backbones.sftc_local import SFTCSource, sensor_order, sincos_2d
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss.wrt_loss import TripletLoss_WRT
from processor.processor import PCA, GPD, PrototypeBank
from utils.sftc_diagnostics import SFTCDiagnostics, cls_source_difference, write_cls_diagnostic


def config(enabled=True):
    cfg = default_cfg.clone()
    cfg.merge_from_file(str(ROOT / 'configs' / ('UAD_SFTC_V2_1.yml' if enabled else 'UAD.yml')))
    cfg.MODEL.PRETRAIN_CHOICE = 'self'  # synthetic weights; no download
    return cfg


def expect_error(call, exception=ValueError):
    try:
        call()
    except exception:
        return
    raise AssertionError('Expected {}'.format(exception.__name__))


def norm(parameters):
    return sum(p.grad.float().square().sum().item() for p in parameters if p.grad is not None) ** 0.5


def run_checks(original_dir=None):
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    results = {}
    cfg = config()
    baseline_cfg = config(False)
    # Only SFTC, output path and the exposed clip default differ from UAD.
    for section in ('MODEL', 'INPUT', 'DATASETS', 'DATALOADER', 'SOLVER', 'TEST'):
        assert cfg[section] == baseline_cfg[section], section
    model = make_model(cfg, 2, 7, 3)
    source = model.base.sftc
    baseline_keys = {k for k in model.state_dict() if not k.startswith('base.sftc.')}
    results['parameters_total_2_ids'] = sum(p.numel() for p in model.parameters())
    results['parameters_added'] = sum(p.numel() for p in source.parameters())
    results['parameters_encoder'] = sum(p.numel() for p in source.encoder.parameters())
    results['parameters_kv'] = sum(p.numel() for p in source.local_k.parameters()) + sum(p.numel() for p in source.local_v.parameters())
    qkv = model.base.blocks[6].attn.qkv
    assert torch.equal(source.local_k.weight, qkv.weight[768:1536])
    assert torch.equal(source.local_v.weight, qkv.weight[1536:])
    assert torch.equal(source.local_k.bias, qkv.bias[768:1536])
    assert source.local_k.weight.data_ptr() != qkv.weight.data_ptr()
    results['kv_initialization'] = 'exact independent copies of backbone K/V'

    center = nn.Linear(1, 1)
    optimizer, _ = make_optimizer(cfg, model, center)
    groups = {id(g['params'][0]): g for g in optimizer.param_groups}
    assert len(groups) == sum(p.requires_grad for p in model.parameters())
    for name, p in source.named_parameters():
        group = groups[id(p)]
        assert group['lr'] == 0.024
        no_decay = p.ndim == 1 or name.endswith(('source_bias', 'modal_gamma', 'modal_beta'))
        assert group['weight_decay'] == (0. if no_decay else 1e-4), name
    assert groups[id(qkv.weight)]['lr'] == 0.008
    assert groups[id(qkv.bias)]['lr'] == 0.016
    results['optimizer'] = 'all parameters included exactly once; LR/decay checked'
    scheduler = create_scheduler(cfg, optimizer)
    scheduler.step(1)
    results['actual_lr_epoch1_backbone_new'] = [groups[id(qkv.weight)]['lr'], groups[id(source.local_k.weight)]['lr']]
    scheduler.step(6)
    results['actual_lr_epoch6_backbone_new'] = [groups[id(qkv.weight)]['lr'], groups[id(source.local_k.weight)]['lr']]

    images = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    sensor_ids = model.training_modal_ids(images)
    assert sensor_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert model.evaluation_modal_ids((1, 2, 3), torch.device('cpu')).tolist() == [0, 1, 2]
    model.sensor_order = sensor_order(('Thermal', 'RGB', 'IR'))
    assert model.training_modal_ids(images).tolist() == [2, 2, 0, 0, 1, 1]
    assert model.evaluation_modal_ids((1, 2, 3), torch.device('cpu')).tolist() == [2, 0, 1]
    model.sensor_order = sensor_order(cfg.DATASETS.MODALITIES)
    expect_error(lambda: model.evaluation_modal_ids((0,), torch.device('cpu')))
    expect_error(lambda: sensor_order(('RGB', 'IR', 'NIR')))
    results['modality_mapping'] = 'training concatenation, one-based eval and reordered folders checked'

    shapes = {}
    handles = []
    for name, module in [('f4', source.encoder.stage1), ('f8', source.encoder.stage2),
                         ('fusion', source.encoder.fusion), ('tokens', source.encoder.norm)]:
        handles.append(module.register_forward_hook(lambda m, args, out, key=name: shapes.update({key: list(out.shape)})))
    for index, block in enumerate(model.base.blocks):
        handles.append(block.register_forward_hook(lambda m, args, out, key='block': shapes.update({key: list(out.shape)})))
    model.eval()
    source.collect_stats = True
    with torch.no_grad():
        full = torch.cat(images)
        feature_on = model(full, mode=1, modal_ids=sensor_ids)
        feature_off = model(full, mode=1, sftc_enabled=False)
    assert shapes == {'f4': [6, 64, 64, 32], 'f8': [6, 128, 32, 16],
                      'fusion': [6, 256, 16, 8], 'tokens': [6, 128, 768], 'block': [6, 129, 768]}
    assert source.last_stats['attention_shape'] == [6, 12, 129, 257]
    assert feature_on.shape == (6, 768)
    for handle in handles:
        handle.remove()
    results['shapes'] = shapes
    results['attention_shape'] = source.last_stats['attention_shape']
    results['cls_difference_random_weights'] = cls_source_difference(feature_on, feature_off)
    expect_error(lambda: model(full, mode=1))
    expect_error(lambda: source(images[0], torch.tensor([-1, 0])), IndexError)

    # Source-off must be bit-for-bit equal to the original backbone computation.
    saved_source = model.base.sftc
    del model.base.sftc
    with torch.no_grad():
        assert torch.equal(feature_off, model(full, mode=1))
    model.base.sftc = saved_source
    results['source_off_equivalence'] = 'bit-exact to same weights with local module removed'
    if original_dir:
        path = Path(original_dir) / 'model' / 'backbones' / 'vit_pytorch.py'
        spec = importlib.util.spec_from_file_location('original_vit', path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        original_base = original.vit_base_in(stride_size=[16, 16])
        original_base.load_state_dict({k: v for k, v in model.base.state_dict().items() if not k.startswith('sftc.')})
        original_base.eval()
        with torch.no_grad():
            assert torch.equal(feature_off, original_base(full))
        del original_base
        gc.collect()
        results['original_backbone_equivalence'] = 'bit-exact to supplied pre-edit backbone'

    # Synthetic pretrained state also exercises positional resizing and prefix stripping.
    with tempfile.TemporaryDirectory(prefix='sftc_checks_') as directory:
        pretrain = Path(directory) / 'pretrain.pth'
        weights = {k: v for k, v in model.base.state_dict().items() if not k.startswith('sftc.')}
        weights['pos_embed'] = torch.randn(1, 197, 768)
        torch.save({'model': {'module.' + k: v for k, v in weights.items()}}, pretrain)
        pretrained_cfg = config()
        pretrained_cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'
        pretrained_cfg.MODEL.PRETRAIN_PATH = str(pretrain)
        loaded_model = make_model(pretrained_cfg, 2, 7, 3)
        assert not loaded_model.base.pretrain_report['missing']
        assert torch.equal(loaded_model.base.sftc.local_k.weight, loaded_model.base.blocks[6].attn.qkv.weight[768:1536])
        del loaded_model
        weights.pop('blocks.6.attn.qkv.weight')
        torch.save(weights, pretrain)
        # Use a plain backbone so strict preload doesn't expect SFTC state.
        saved_source = model.base.sftc
        del model.base.sftc
        expect_error(lambda: model.base.load_param(str(pretrain), strict_backbone=True), RuntimeError)
        model.base.sftc = saved_source
        results['pretrain'] = 'all backbone keys checked; resized 197->129; missing QKV rejected'

        model.train()
        targets = torch.tensor([0, 1])
        bank = PrototypeBank(2, 768, device='cpu')
        loss_history = []
        for epoch in (1, 5, 6):
            source.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            score, feat, bn_feat = model(images)
            ce = F.cross_entropy(score, targets.repeat(3))
            wrt = TripletLoss_WRT()(feat, targets.repeat(3))[0]
            pca, ids, centers = PCA(feat, targets, 3, group_size=1)
            gpd = GPD(feat, targets, bank, ids, centers, 3)
            loss = ce + wrt + .01 * pca + gpd
            loss.backward()
            for name, params in [('cnn', source.encoder.parameters()), ('kv', list(source.local_k.parameters()) + list(source.local_v.parameters())),
                                 ('vit', [qkv.weight, model.base.patch_embed.proj.weight])]:
                value = norm(params)
                assert 0 < value < float('inf'), (name, epoch, value)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in source.parameters())
            if epoch <= 5:
                # Deliberately inject stale momentum; fixed-period guard must remove it.
                optimizer.state[source.source_bias]['momentum_buffer'] = torch.ones_like(source.source_bias)
            else:
                assert source.source_bias.grad.abs().max() > 0
            source.suppress_bias_gradient(optimizer)
            diag = SFTCDiagnostics(model, directory, logging.getLogger('check'))
            before = diag.before_step()
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            coefficient = min(1., 1. / (total_norm.item() + 1e-6))
            old_bias = source.source_bias.detach().clone()
            optimizer.step()
            diag.after_step(epoch, 1, optimizer, before, total_norm.item(), coefficient, False)
            if epoch <= 5:
                assert torch.equal(source.source_bias, torch.full_like(source.source_bias, -2.))
            else:
                assert not torch.equal(source.source_bias, old_bias)
            bank.update(ids, centers, ddp=False)
            loss_history.append({'epoch': epoch, 'ce': ce.item(), 'wrt': wrt.item(), 'proca': pca.item(), 'gpd': gpd.item(),
                                 'bias_min': source.source_bias.min().item(), 'bias_max': source.source_bias.max().item()})
        rows = [json.loads(line) for line in Path(diag.path).read_text().splitlines()]
        assert len(rows) == 3 and rows[-1]['updates']['cnn']['l2'] > 0 and rows[-1]['updates']['kv']['l2'] > 0
        results['synthetic_loss_and_freeze_checks'] = loss_history
        results['diagnostics'] = 'CNN/KV true update norms and pre-dropout per-sensor/head probabilities checked'
        write_cls_diagnostic(directory, {'rank': 2, 'epoch': 120, 'folder': 'Thermal'})
        assert json.loads((Path(directory) / 'sftc_cls_diagnostics_rank2.jsonl').read_text())['folder'] == 'Thermal'
        model.eval()
        with torch.no_grad():
            reference = model(full, mode=1, modal_ids=sensor_ids)
        checkpoint = Path(directory) / 'sftc.pth'
        with torch.no_grad():
            source.local_k.weight.add_(0.1)
        torch.save({'state_dict': {'module.' + k: v for k, v in model.state_dict().items()}}, checkpoint)
        saved_k = source.local_k.weight.detach().clone()
        source.local_k.weight.data.zero_()
        model.load_param(str(checkpoint))
        assert torch.equal(source.local_k.weight, saved_k)
        assert not torch.equal(source.local_k.weight, qkv.weight[768:1536])
        with torch.no_grad():
            checkpoint_feature = model(full, mode=1, modal_ids=sensor_ids)
        model.load_param(str(checkpoint))
        with torch.no_grad():
            assert torch.equal(checkpoint_feature, model(full, mode=1, modal_ids=sensor_ids))
        torch.save({k: v for k, v in model.state_dict().items() if k in baseline_keys}, checkpoint)
        expect_error(lambda: model.load_param(str(checkpoint)), RuntimeError)
        results['checkpoint'] = 'full load exact; trained local K/V not recopied; baseline checkpoint rejected'

    # Pure attention algebra: zero logits yield the expected cardinality-aware prior.
    attn = Attention(768, 12, qkv_bias=True, attn_drop=.5)
    with torch.no_grad():
        attn.qkv.weight.zero_()
        attn.qkv.bias.zero_()
    seen = {}
    attn.train()
    attn(torch.zeros(1, 129, 768),
         (torch.zeros(1, 12, 128, 64), torch.zeros(1, 12, 128, 64), torch.full((1, 12, 1, 1), -2.)),
         lambda p, n: seen.update({'rho': p[..., n:].sum(-1)}))
    expected = 128 * torch.exp(torch.tensor(-2.)) / (129 + 128 * torch.exp(torch.tensor(-2.)))
    assert torch.allclose(seen['rho'], expected.expand_as(seen['rho']), atol=1e-7)
    results['joint_softmax_prior'] = expected.item()
    pe = sincos_2d(16, 8, 768)
    assert torch.equal(pe[0, 0, :384], pe[0, 1, :384]) and torch.equal(pe[0, 0, 384:], pe[0, 8, 384:])
    results['position_encoding'] = 'fixed y/x halves; row-major grid checked'
    block = Block(768, 12, qkv_bias=True)
    amp_source = SFTCSource(block.attn.qkv)
    amp_source.set_epoch(6)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        context = amp_source(torch.randn(3, 3, 256, 128), torch.tensor([0, 1, 2]))
        amp_feature = block(torch.randn(3, 129, 768), context)[:, 0]
        amp_loss = amp_feature.square().mean()
    amp_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in amp_source.parameters())
    assert norm(amp_source.encoder.parameters()) > 0 and norm(amp_source.local_k.parameters()) > 0
    results['cpu_bfloat16_autocast'] = 'local source + ViT block forward/backward finite (not CUDA FP16 validation)'
    return results


class DistributedCheckModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block(768, 12, qkv_bias=True)
        self.sftc = SFTCSource(self.block.attn.qkv)
        self.classifier = nn.Linear(768, 2)
        self.unused_head = nn.Linear(768, 2)

    def forward(self, images, tokens, ids):
        return self.classifier(self.block(tokens, self.sftc(images, ids))[:, 0])


def ddp_worker(rank, port):
    import datetime
    import torch.distributed as dist
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:{}'.format(port),
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(1234)
        model = DistributedCheckModel()
        ddp = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.024, momentum=.9, weight_decay=0.)
        for epoch in (1, 5, 6, 7):
            model.sftc.set_epoch(epoch)
            for iteration in range(2):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(rank * 100 + epoch * 10 + iteration)
                score = ddp(torch.randn(3, 3, 256, 128), torch.randn(3, 129, 768), torch.tensor([0, 1, 2]))
                F.cross_entropy(score, torch.tensor([0, 1, 0])).backward()
                assert norm(model.sftc.encoder.parameters()) > 0
                model.sftc.suppress_bias_gradient(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                for p in (model.sftc.source_bias, model.sftc.encoder.stem_conv.weight, model.sftc.local_k.weight):
                    copies = [torch.empty_like(p) for _ in range(2)]
                    dist.all_gather(copies, p.detach())
                    assert torch.equal(copies[0], copies[1])
                if epoch <= 5:
                    assert torch.equal(model.sftc.source_bias, torch.full_like(model.sftc.source_bias, -2.))
                else:
                    assert not torch.equal(model.sftc.source_bias, torch.full_like(model.sftc.source_bias, -2.))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--ddp-only', action='store_true')
    parser.add_argument('--original-dir')
    parser.add_argument('--report')
    args = parser.parse_args()
    started = time.perf_counter()
    results = {} if args.ddp_only else run_checks(args.original_dir)
    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    if args.ddp or args.ddp_only:
        if not torch.distributed.is_gloo_available():
            raise RuntimeError('Gloo unavailable; run the DDP check on a supported host')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        torch.multiprocessing.spawn(ddp_worker, args=(port,), nprocs=2, join=True)
        results['ddp'] = '2 CPU/Gloo processes; 8 updates across fixed/released bias; synchronized CNN/KV/bias'
    results['elapsed_seconds'] = time.perf_counter() - started
    results['runtime'] = {'torch': torch.__version__, 'device': 'cpu', 'cuda_available': torch.cuda.is_available()}
    output = json.dumps(results, indent=2)
    if args.report:
        Path(args.report).write_text(output + '\n', encoding='utf-8')
    print(output)
