"""Chunked full-gallery diagnostics preserving WHU's all-same-camera exclusion."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

NAMES = {0: 'RGB', 1: 'NIR', 2: 'TIR'}


def write_csv(path, rows, fields=None):
    path = Path(path)
    rows = list(rows)
    if fields is None:
        if not rows:
            raise ValueError('Empty CSV requires explicit fields')
        fields = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protocols(metadata, num_query, similarity=False):
    query, gallery = metadata[:num_query], metadata[num_query:]
    def indices(items, sensor=None, platform=None):
        return np.array([i for i, item in enumerate(items)
                         if (sensor is None or item['sensor_id'] == sensor)
                         and (platform is None or item['platform'] == platform)], dtype=np.int64)
    yield 'AS-ReID', indices(query), indices(gallery)
    for qm in range(3):
        for gm in range(3):
            yield NAMES[qm] + '->' + NAMES[gm], indices(query, qm), indices(gallery, gm)
    for qp, gp in [('A', 'G'), ('G', 'A')]:
        yield qp + '->' + gp, indices(query, platform=qp), indices(gallery, platform=gp)
    if similarity:
        for qp in ('G', 'A'):
            yield qp + '->' + qp, indices(query, platform=qp), indices(gallery, platform=qp)
        for qm in range(3):
            for qp in ('G', 'A'):
                for gm in range(3):
                    for gp in ('G', 'A'):
                        yield '{}-{}->{}-{}'.format(NAMES[qm], qp, NAMES[gm], gp), indices(query, qm, qp), indices(gallery, gm, gp)


class Distribution:
    def __init__(self):
        self.count, self.total, self.squared = 0, 0., 0.
        self.minimum, self.maximum = None, None
        self.edges = np.linspace(-1, 1, 41)
        self.hist = np.zeros(40, dtype=np.int64)

    def update(self, values):
        values = np.clip(np.asarray(values, dtype=np.float64), -1., 1.)
        if not values.size:
            return
        self.count += values.size
        self.total += values.sum()
        self.squared += np.square(values).sum()
        self.minimum = float(values.min()) if self.minimum is None else min(self.minimum, float(values.min()))
        self.maximum = float(values.max()) if self.maximum is None else max(self.maximum, float(values.max()))
        self.hist += np.histogram(values, self.edges)[0]

    def row(self):
        mean = self.total / self.count if self.count else None
        std = max(0., self.squared / self.count - mean * mean) ** .5 if self.count else None
        return dict(pair_count=int(self.count), mean=mean, std=std, minimum=self.minimum, maximum=self.maximum,
                    histogram_edges=json.dumps(self.edges.tolist()), histogram_counts=json.dumps(self.hist.tolist()))


def evaluate_features(features, metadata, num_query, mode, chunk_size=32, similarity=False):
    """Return protocol metrics, stable overall Query records, optional histograms.

    Only chunk_size x current gallery distances/indices are materialized. AP and
    INP use all eligible gallery samples, never truncated top-k candidates.
    """
    if chunk_size < 1 or num_query < 1 or num_query >= len(metadata):
        raise ValueError('Require a positive chunk and nonempty Query/Gallery')
    if len(features) != len(metadata):
        raise ValueError('Feature and metadata lengths differ')
    for start in range(0, len(features), 4096):
        if not np.isfinite(features[start:start + 4096]).all():
            raise ValueError('Nonfinite feature encountered')
    qmeta, gmeta = metadata[:num_query], metadata[num_query:]
    qpids = np.array([m['pid'] for m in qmeta])
    gpids = np.array([m['pid'] for m in gmeta])
    qcams = np.array([m['camid'] for m in qmeta])
    gcams = np.array([m['camid'] for m in gmeta])
    # Features are CPU memmaps. Normalize only the selected gallery and current
    # query chunk; this avoids copies of every intervention's complete features.
    rows, overall, distributions = [], [], []
    retrieval_protocols = {p for p, _, _ in protocols(metadata, num_query)}
    for protocol, qi, gi in protocols(metadata, num_query, similarity):
        records, positive, negative = [], Distribution(), Distribution()
        distribution_only = protocol not in retrieval_protocols
        if len(gi):
            gf = F.normalize(torch.from_numpy(np.array(features[num_query + gi], dtype=np.float32)), dim=1)
            gt, gsq = gf.t().contiguous(), gf.square().sum(1)[None]
            for start in range(0, len(qi), chunk_size):
                current = qi[start:start + chunk_size]
                qf = F.normalize(torch.from_numpy(np.array(features[current], dtype=np.float32)), dim=1)
                distance = qf.square().sum(1)[:, None] + gsq
                distance.addmm_(qf, gt, beta=1, alpha=-2)
                orders = None if distribution_only else torch.argsort(distance, dim=1).numpy()
                for j, query_index in enumerate(current):
                    keep = gcams[gi] != qcams[query_index]
                    same = gpids[gi] == qpids[query_index]
                    if similarity:
                        # For normalized features: squared L2 = 2 - 2*cos.
                        values = (1. - distance[j].numpy() / 2.)
                        positive.update(values[keep & same])
                        negative.update(values[keep & ~same])
                    if distribution_only:
                        continue
                    order = orders[j]
                    valid_order = order[keep[order]]
                    hits = np.flatnonzero(same[valid_order])
                    record = dict(query_index=int(query_index), valid=bool(len(hits)),
                                  AP=None, R1=None, first_hit=None, INP=None)
                    if len(hits):
                        record.update(AP=float(np.mean(np.arange(1, len(hits) + 1) / (hits + 1))),
                                      R1=int(hits[0] == 0), first_hit=int(hits[0] + 1),
                                      INP=float(len(hits) / (hits[-1] + 1)))
                    records.append(record)
        else:
            records = [dict(query_index=int(i), valid=False, AP=None, R1=None, first_hit=None, INP=None) for i in qi]
        valid = [r for r in records if r['valid']]
        row = dict(mode=mode, protocol=protocol, query_count=len(qi), gallery_count=len(gi), valid_query_count=len(valid),
                   mAP=None, R1=None, R5=None, R10=None, mINP=None,
                   status='statistics_only' if distribution_only else ('ok' if valid else 'empty_or_no_valid_query'))
        if valid:
            row['mAP'] = 100. * np.mean([r['AP'] for r in valid])
            row['mINP'] = 100. * np.mean([r['INP'] for r in valid])
            for rank in (1, 5, 10):
                row['R' + str(rank)] = float(100. * np.mean([r['first_hit'] <= rank for r in valid])) if len(gi) >= rank else None
        rows.append(row)
        if protocol == 'AS-ReID':
            overall = records
        if similarity:
            for relation, dist in [('same_identity', positive), ('different_identity', negative)]:
                distributions.append(dict(feature=mode, protocol=protocol, relation=relation,
                                          same_camera_excluded=True, **dist.row()))
    return rows, overall, distributions


def query_comparison(metadata, num_query, results, modes):
    lookup = {mode: {r['query_index']: r for r in results[mode]} for mode in modes}
    rows = []
    for index, item in enumerate(metadata[:num_query]):
        row = dict(query_index=index, pid=item['pid'], camid=item['camid'],
                   modality=item['modality'], platform=item['platform'], image_path=item['image_path'])
        for mode in modes:
            record = lookup[mode][index]
            row.update({key + '_' + mode: record[key] for key in ('AP', 'R1', 'first_hit', 'INP', 'valid')})
        rows.append(row)
    return rows
