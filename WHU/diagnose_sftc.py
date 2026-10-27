"""Four checkpoint-only SFTC experiments; run from WHU, without torchrun."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from config import cfg as default_cfg
from model import make_model
from utils.sftc_inference_probe import inference_probe, layer_views
from utils.sftc_probe_metrics import evaluate_features, query_comparison, write_csv

ROOT = Path(__file__).resolve().parent


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def evaluation_metadata(dataset, modalities, sensor_order):
    records, num_query = [], 0
    for split, source in [('query', dataset.query), ('gallery', dataset.gallery)]:
        for folder, sensor in zip(modalities, sensor_order):
            for path, pid, camid, dataset_modal_id in source.get(folder, []):
                records.append(dict(sample_index=len(records), split=split, image_path=str(Path(path).resolve()),
                                    pid=int(pid), camid=int(camid), dataset_modality_id=int(dataset_modal_id),
                                    sensor_id=int(sensor), modality=('RGB', 'NIR', 'TIR')[sensor],
                                    platform='G' if camid < 5 else 'A'))
                if split == 'query':
                    num_query += 1
    return records, num_query


def build_loader(cfg, metadata, batch_size, sample_indices=None):
    # Module-level dataset also works with Windows spawn workers. Imports of
    # torchvision/timm are delayed until a real-data run.
    return torch.utils.data.DataLoader(_EvaluationImages(metadata, sample_indices, cfg),
                                      batch_size=batch_size, shuffle=False,
                                      num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=torch.cuda.is_available())


class _EvaluationImages(torch.utils.data.Dataset):
    def __init__(self, metadata, indices, cfg):
        import torchvision.transforms as T
        self.metadata = metadata
        self.indices = list(range(len(metadata))) if indices is None else list(indices)
        self.transform = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                                    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        from datasets.bases import read_image
        index = self.indices[item]
        return self.transform(read_image(self.metadata[index]['image_path'])), index


def balanced_samples(metadata, per_domain=50):
    selected = []
    for sensor in range(3):
        for platform in ('G', 'A'):
            identities = defaultdict(list)
            for i, item in enumerate(metadata):
                if item['sensor_id'] == sensor and item['platform'] == platform:
                    identities[item['pid']].append(i)
            queues = [deque(sorted(identities[pid], key=lambda i: hashlib.sha256(metadata[i]['image_path'].encode()).hexdigest()))
                      for pid in sorted(identities, key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())]
            count = 0
            while queues and count < per_domain:
                remaining = []
                for queue in queues:
                    if count < per_domain:
                        selected.append(queue.popleft())
                        count += 1
                    if queue:
                        remaining.append(queue)
                queues = remaining
    return sorted(selected)


class AttentionSummary:
    def __init__(self):
        self.groups = {}
    def add(self, mode, records, stats):
        for region, metrics in stats.items():
            for i, record in enumerate(records):
                for head in range(12):
                    key = (mode, record['modality'], record['platform'], region, head)
                    group = self.groups.setdefault(key, {'sample_count': 0, 'sums': {name: 0. for name in metrics}})
                    group['sample_count'] += 1
                    for name, values in metrics.items():
                        group['sums'][name] += float(values[i, head])
    def rows(self):
        return [dict(mode=k[0], modality=k[1], platform=k[2], query_region=k[3], head=k[4],
                     sample_count=v['sample_count'], **{m: total / v['sample_count'] for m, total in v['sums'].items()})
                for k, v in sorted(self.groups.items())]


@torch.inference_mode()
def extract_features(model, loader, metadata, mode, directory, device, local=False, logger=print):
    names = [(mode, 768)] + ([('cnn_fused', 256), ('local_value', 768)] if local else [])
    arrays = {name: np.lib.format.open_memmap(directory / (name + '.partial.npy'), mode='w+', dtype=np.float32,
                                             shape=(len(metadata), dim)) for name, dim in names}
    seen, summary = np.zeros(len(metadata), dtype=bool), AttentionSummary()
    with inference_probe(model, mode, collect_local=local, collect_attention=mode != 'off') as captured:
        for batch_index, (images, indices) in enumerate(loader):
            items = [metadata[i] for i in indices.tolist()]
            modal_ids = model.evaluation_modal_ids([x['dataset_modality_id'] for x in items], device)
            captured.clear()
            features = model(images.to(device), mode=1, modal_ids=modal_ids, sftc_enabled=mode != 'off')
            if not torch.isfinite(features).all():
                raise RuntimeError('Nonfinite final feature: ' + mode)
            positions = indices.numpy()
            if seen[positions].any():
                raise RuntimeError('Duplicate extraction index')
            seen[positions] = True
            arrays[mode][positions] = features.float().cpu().numpy()
            if local:
                for name in ('cnn_fused', 'local_value'):
                    if not torch.isfinite(captured[name]).all():
                        raise RuntimeError('Nonfinite local feature: ' + name)
                    arrays[name][positions] = F.normalize(captured[name], dim=1).numpy()
            if 'attention' in captured:
                summary.add(mode, items, captured['attention'])
            if batch_index % 50 == 0:
                logger('{} extraction: {}/{} batches'.format(mode, batch_index + 1, len(loader)))
    if not seen.all():
        raise RuntimeError('Incomplete full Query/Gallery extraction')
    for array in arrays.values():
        array.flush()
    arrays.clear()
    # Drop the final loop reference too, so Windows can rename the memmap.
    del array
    for name, _ in names:
        (directory / (name + '.partial.npy')).replace(directory / (name + '.npy'))
    save_json(directory / (mode + '_attention.json'), summary.rows())


@torch.inference_mode()
def check_existing_paths(model, loader, metadata, device):
    images, indices = next(iter(loader))
    images = images[:2].to(device)
    ids = model.evaluation_modal_ids([metadata[i]['dataset_modality_id'] for i in indices[:2].tolist()], device)
    errors = {}
    for mode in ('on', 'off'):
        original = model(images, mode=1, modal_ids=ids, sftc_enabled=mode == 'on')
        with inference_probe(model, mode):
            probed = model(images, mode=1, modal_ids=ids, sftc_enabled=mode == 'on')
        error = (original - probed).abs().max().item()
        torch.testing.assert_close(original, probed, rtol=1e-5, atol=1e-6)
        errors[mode + '_max_absolute_error'] = error
    return errors


@torch.inference_mode()
def propagation(model, loader, metadata, device, output):
    rows = []
    for images, indices in loader:
        images = images.to(device)
        ids = model.evaluation_modal_ids([metadata[i]['dataset_modality_id'] for i in indices.tolist()], device)
        captured_modes = {}
        for mode in ('off', 'on', 'cls_only', 'patch_only'):
            with inference_probe(model, mode, collect_layers=True) as captured:
                model(images, mode=1, modal_ids=ids, sftc_enabled=mode != 'off')
            captured_modes[mode] = captured
        reference = layer_views(captured_modes['off'])
        for mode, captured in captured_modes.items():
            if not torch.equal(captured['block7_input'], captured_modes['off']['block7_input']):
                raise RuntimeError('Block 7 input differs across interventions')
            for location, value in layer_views(captured).items():
                baseline = reference[location]
                cosine = 1. - F.cosine_similarity(value, baseline, dim=1)
                relative = (value - baseline).norm(dim=1) / (baseline.norm(dim=1) + 1e-12)
                for j, index in enumerate(indices.tolist()):
                    record = metadata[index]
                    rows.append(dict(sample_index=index, split=record['split'], image_path=record['image_path'],
                                     pid=record['pid'], camid=record['camid'], modality=record['modality'],
                                     platform=record['platform'], mode=mode, location=location,
                                     cosine_distance=0. if mode == 'off' else float(cosine[j]),
                                     relative_l2=float(relative[j]), block7_input_identical=True))
    write_csv(output / '03_layer_propagation.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/UAD_SFTC_V2_1.yml')
    parser.add_argument('--checkpoint', default=None, help='Full Epoch 120 SFTC checkpoint; alternatively TEST.WEIGHT')
    parser.add_argument('--output', required=True)
    parser.add_argument('--experiments', nargs='+', type=int, choices=[1, 2, 3, 4], default=[1, 2])
    parser.add_argument('--device', default='cuda:0', help='One device; do not use torchrun/DataParallel')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--query-chunk', type=int, default=32)
    parser.add_argument('--samples-per-domain', type=int, default=50)
    parser.add_argument('opts', nargs=argparse.REMAINDER, help='Original config overrides; put last')
    args = parser.parse_args()
    if min(args.batch_size, args.query_chunk, args.samples_per_domain) < 1:
        parser.error('Batch, distance chunk and sample count must be positive')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('This independent diagnostic entrypoint uses one process; do not launch with torchrun')
    cfg = default_cfg.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts[1:] if args.opts[:1] == ['--'] else args.opts)
    if not cfg.SFTC.ENABLED or cfg.TEST.NECK_FEAT != 'before' or cfg.TEST.RE_RANKING or cfg.TEST.TOP_K_EVAL != 0:
        parser.error('Require SFTC enabled, pre-BN, no reranking and TOP_K_EVAL=0')
    if str(cfg.TEST.FEAT_NORM).lower() not in ('yes', 'true', '1'):
        parser.error('Diagnostics require original L2 feature normalization')
    checkpoint = Path(args.checkpoint or cfg.TEST.WEIGHT).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error('Supply an existing full SFTC checkpoint')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICE', str(cfg.MODEL.DEVICE_ID))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu only for synthetic/small checks')
    torch.manual_seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    def log(message):
        print(message, flush=True)
        with (output / 'diagnostic_log.txt').open('a', encoding='utf-8') as stream:
            stream.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')
    from datasets.whu_mars import WHU_MARS
    from model.backbones.sftc_local import sensor_order
    dataset = WHU_MARS(root=cfg.DATASETS.ROOT_DIR, modalities=cfg.DATASETS.MODALITIES,
                      dataset_dir=cfg.DATASETS.DATASET_DIR, filter_gd=cfg.DATASETS.FILTER_GD)
    metadata, num_query = evaluation_metadata(dataset, cfg.DATASETS.MODALITIES, sensor_order(cfg.DATASETS.MODALITIES))
    if not num_query or len(metadata) <= num_query:
        parser.error('Original Query/Gallery must both be nonempty')
    # Do not load ImageNet or recreate a train sampler: the full trained state
    # supplies every used parameter. Existing strict SFTC loader rejects gaps.
    init_cfg = cfg.clone()
    init_cfg.MODEL.PRETRAIN_CHOICE = 'self'
    init_cfg.MODEL.DIST_TRAIN = False
    model = make_model(init_cfg, dataset.num_train_pids, dataset.num_train_cams, dataset.num_train_vids)
    model.load_param(str(checkpoint))
    model.eval().requires_grad_(False).to(device)
    before = state_digest(model)
    metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    code_files = ['diagnose_sftc.py', 'utils/sftc_inference_probe.py', 'utils/sftc_probe_metrics.py',
                  'model/make_model.py', 'model/backbones/vit_pytorch.py', 'model/backbones/sftc_local.py',
                  'datasets/whu_mars.py', 'datasets/bases.py']
    identity = dict(checkpoint_sha256=file_digest(checkpoint), metadata_sha256=metadata_hash,
                    config=str(cfg), code_sha256={p: file_digest(ROOT / p) for p in code_files})
    cache = output / 'features'
    cache.mkdir(exist_ok=True)
    identity_file = cache / 'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text(encoding='utf-8')) != identity:
        parser.error('Output cache belongs to different weights/data/config/code; choose a new --output')
    save_json(identity_file, identity)
    write_csv(output / 'evaluation_samples.csv', metadata)
    loader = build_loader(cfg, metadata, args.batch_size)
    self_checks = check_existing_paths(model, loader, metadata, device)
    metrics, queries = {}, {}
    def get(mode):
        path = cache / (mode + '.npy')
        want_local = mode == 'on' and 4 in args.experiments
        missing_local = want_local and not all((cache / (name + '.npy')).exists() for name in ('cnn_fused', 'local_value'))
        if not path.exists() or not (cache / (mode + '_attention.json')).exists() or missing_local:
            extract_features(model, loader, metadata, mode, cache, device, local=want_local, logger=log)
        features = np.load(path, mmap_mode='r')
        if features.shape != (len(metadata), 768):
            raise RuntimeError('Cached feature shape mismatch')
        if mode not in metrics:
            log('Full-gallery evaluation: ' + mode)
            metrics[mode], queries[mode], _ = evaluate_features(features, metadata, num_query, mode, args.query_chunk)
        return features
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=identity['checkpoint_sha256'],
                    query_count=num_query, gallery_count=len(metadata) - num_query, metadata_sha256=metadata_hash,
                    config=str(cfg), device=str(device), batch_size=args.batch_size, query_chunk=args.query_chunk,
                    experiments=args.experiments, code_sha256=identity['code_sha256'],
                    protocol=dict(feature='768-dimensional pre-BN CLS except experiment 4', normalization='L2',
                                  distance='squared Euclidean', camera_filter='exclude ALL same-camera gallery images',
                                  platform='zero-based camera <5 ground, >=5 aerial', reranking=False, TTA=False, top_k=0,
                                  query_order='configured modality order, sorted paths within each modality',
                                  metric_csv_units='percent (0..100)', query_AP_INP_units='fraction (0..1)',
                                  invalid_query='blank AP/R1/first_hit/INP; excluded from metric mean',
                                  feature_cache='float32 .npy memmaps; same-mode Query and Gallery only'),
                    self_checks=self_checks, state_sha256_before=before,
                    limitations=['CNN pooled feature retrieval is a diagnostic, not an independently trained ReID model',
                                 'Local construction skips ImageNet loading and loads the strict complete checkpoint once',
                                 'Classifier checkpoint is skipped by original eval loader; classifier is unused in pre-BN retrieval',
                                 'Dataset image content is not hashed; do not modify image files when reusing the cache'])
    if 1 in args.experiments:
        get('on'); get('off')
        write_csv(output / '01_retrieval_comparison.csv', metrics['on'] + metrics['off'])
        comparison = query_comparison(metadata, num_query, queries, ['on', 'off'])
        write_csv(output / '01_query_comparison.csv', comparison)
        transitions = [dict(row, transition='improved' if row['R1_on'] == 1 else 'regressed') for row in comparison
                       if row['valid_on'] and row['valid_off'] and row['R1_on'] != row['R1_off']]
        write_csv(output / '01_rank1_transitions.csv', transitions, list(comparison[0]) + ['transition'])
        save_json(output / '01_manifest.json', manifest)
    if 2 in args.experiments:
        for mode in ('off', 'zero_local_value', 'on'):
            get(mode)
        write_csv(output / '02_source_decomposition.csv', [row for mode in ('off', 'zero_local_value', 'on') for row in metrics[mode]])
        write_csv(output / '02_query_comparison.csv', query_comparison(metadata, num_query, queries, ['off', 'zero_local_value', 'on']))
    if 3 in args.experiments:
        for mode in ('off', 'on', 'cls_only', 'patch_only'):
            get(mode)
        write_csv(output / '03_path_retrieval.csv', [row for mode in ('off', 'on', 'cls_only', 'patch_only') for row in metrics[mode]])
        selected = balanced_samples(metadata, args.samples_per_domain)
        sample_rows = [metadata[i] for i in selected]
        write_csv(output / '03_fixed_samples.csv', sample_rows)
        counts = {m + '-' + p: sum(r['modality'] == m and r['platform'] == p for r in sample_rows)
                  for m in ('RGB', 'NIR', 'TIR') for p in ('G', 'A')}
        manifest['propagation_samples'] = dict(per_domain_requested=args.samples_per_domain, actual=counts,
                                               selector='deterministic identity round-robin, SHA256 path ordering',
                                               attention_location='after output projection/dropout, before residual and DropPath',
                                               patch_comparison='flatten 128x768 per image', relative_l2_denominator='off norm + 1e-12')
        propagation(model, build_loader(cfg, metadata, args.batch_size, selected), metadata, device, output)
    if 4 in args.experiments:
        if not all((cache / (name + '.npy')).exists() for name in ('cnn_fused', 'local_value')):
            extract_features(model, loader, metadata, 'on', cache, device, local=True, logger=log)
        local_metrics, similarities = [], []
        for name, dim in [('cnn_fused', 256), ('local_value', 768)]:
            features = np.load(cache / (name + '.npy'), mmap_mode='r')
            if features.shape != (len(metadata), dim):
                raise RuntimeError('Cached local feature shape mismatch')
            log('Local-feature retrieval and similarity distributions: ' + name)
            rows, _, stats = evaluate_features(features, metadata, num_query, name, args.query_chunk, similarity=True)
            local_metrics.extend(rows[:12])
            similarities.extend(stats)
        write_csv(output / '04_local_feature_retrieval.csv', local_metrics)
        write_csv(output / '04_identity_similarity.csv', similarities)
        save_json(output / '04_feature_manifest.json', dict(checkpoint_sha256=identity['checkpoint_sha256'],
                  features={'cnn_fused': dict(position='fusion Conv/GN/GELU output before projection and 2D PE',
                                             dimension=256, pooling='mean over H,W'),
                            'local_value': dict(position='trained local_v output; equivalent to merged 12 heads',
                                               dimension=768, pooling='mean over 128 tokens')},
                  normalization='L2', similarity='cosine, same-camera excluded for both positive and negative pairs',
                  histogram='40 fixed bins on [-1,1]; population std', labels='evaluation only; no fitted probe',
                  query_count=num_query, gallery_count=len(metadata)-num_query))
    attention_rows = []
    for path in sorted(cache.glob('*_attention.json')):
        attention_rows.extend(json.loads(path.read_text(encoding='utf-8')))
    if attention_rows:
        write_csv(output / 'attention_source_statistics.csv', attention_rows)
    after = state_digest(model)
    if after != before:
        raise RuntimeError('Model parameters or buffers changed during diagnostics')
    manifest.update(state_sha256_after=after, parameter_and_buffer_values_unchanged=True,
                    files={p.name: {'bytes': p.stat().st_size} for p in output.iterdir() if p.is_file()},
                    attention_statistics=dict(precision='FP32 autocast disabled, detached, pre-dropout',
                                              response_location='per-head conditional response before projection',
                                              logZ_local_minus_backbone='excludes bias; effective_logZ_difference includes bias',
                                              zeroV='A_L_norm/cosine zero; original candidate response also retained',
                                              off='local encoder not executed; no local statistics rows'))
    save_json(output / 'diagnostic_manifest.json', manifest)
    if 1 in args.experiments:
        save_json(output / '01_manifest.json', manifest)
    log('Finished. All parameter/buffer values unchanged. Outputs: ' + str(output))


if __name__ == '__main__':
    main()
