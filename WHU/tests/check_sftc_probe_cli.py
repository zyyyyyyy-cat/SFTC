"""Synthetic JPG dataset + complete checkpoint end-to-end CLI and cache check."""
import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg as default_cfg
from model import make_model


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def run():
    torch.set_num_threads(4)
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix='sftc-probe-cli-') as temporary:
        directory = Path(temporary)
        root = directory / 'data'
        rng = np.random.default_rng(7)
        for split, identities, cameras in [('train', [0, 1], [1]), ('query', [10, 11], [1, 6]), ('test', [10, 11], [2, 7])]:
            for folder in ('RGB', 'IR', 'Thermal'):
                target = root / 'WHU-MARS' / split / folder
                target.mkdir(parents=True)
                for pid in identities:
                    for camera in cameras:
                        Image.fromarray(rng.integers(0, 256, (32, 16, 3), dtype=np.uint8)).save(target / ('%04d_c%d.jpg' % (pid, camera)))
        cfg = default_cfg.clone()
        cfg.merge_from_file(str(ROOT / 'configs/UAD_SFTC_V2_1.yml'))
        cfg.MODEL.PRETRAIN_CHOICE = 'self'
        model = make_model(cfg, 2, 1, 3)
        checkpoint = directory / 'synthetic_120.pth'
        torch.save(model.state_dict(), checkpoint)
        del model
        gc.collect()
        output = directory / 'output'
        command = [sys.executable, '-B', str(ROOT / 'diagnose_sftc.py'), '--config_file', str(ROOT / 'configs/UAD_SFTC_V2_1.yml'),
                   '--experiments', '1', '2', '3', '4', '--checkpoint', str(checkpoint), '--output', str(output),
                   '--device', 'cpu', '--batch-size', '4', '--query-chunk', '2', '--samples-per-domain', '2',
                   '--', 'DATASETS.ROOT_DIR', str(root), 'DATALOADER.NUM_WORKERS', '0']
        # Limit CPU threads in the diagnostic child for this tiny fixture.
        bootstrap = "import torch,runpy,sys; torch.set_num_threads(4); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        command = [sys.executable, '-B', '-c', bootstrap] + command[2:]
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout, flush=True)
        assert completed.returncode == 0
        expected = ['01_retrieval_comparison.csv', '01_query_comparison.csv', '01_rank1_transitions.csv', '01_manifest.json',
                    '02_source_decomposition.csv', '02_query_comparison.csv', '03_path_retrieval.csv', '03_layer_propagation.csv',
                    '03_fixed_samples.csv', '04_local_feature_retrieval.csv', '04_identity_similarity.csv', '04_feature_manifest.json',
                    'attention_source_statistics.csv', 'evaluation_samples.csv', 'diagnostic_manifest.json']
        assert all((output / name).exists() for name in expected)
        assert len(rows(output / expected[0])) == 24
        assert len(rows(output / '02_source_decomposition.csv')) == 36
        assert len(rows(output / '03_path_retrieval.csv')) == 48
        assert len(rows(output / '04_local_feature_retrieval.csv')) == 24
        assert len(rows(output / '04_identity_similarity.csv')) == 200
        assert len(rows(output / '03_fixed_samples.csv')) == 12
        assert len(rows(output / '03_layer_propagation.csv')) == 12 * 4 * 10
        assert len(rows(output / '01_query_comparison.csv')) == 12
        assert len(rows(output / 'evaluation_samples.csv')) == 24
        assert len(rows(output / 'attention_source_statistics.csv')) == 4 * 3 * 2 * 2 * 12
        manifest = json.loads((output / 'diagnostic_manifest.json').read_text(encoding='utf-8'))
        assert manifest['parameter_and_buffer_values_unchanged']
        timestamps = {p.name: p.stat().st_mtime_ns for p in (output / 'features').glob('*.npy')}
        again = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert again.returncode == 0, again.stdout
        assert 'extraction:' not in again.stdout
        assert timestamps == {p.name: p.stat().st_mtime_ns for p in (output / 'features').glob('*.npy')}
        return dict(all_four_experiments='pass', all_required_files_present=True,
                    query_count=12, gallery_count=12, similarity_rows=200,
                    propagation_rows=480, cache_reuse_without_reextraction=True,
                    parameter_and_buffer_values_unchanged=True)


if __name__ == '__main__':
    result = run()
    path = ROOT / 'docs/sftc_probe_validation.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    data['synthetic_cli_end_to_end'] = result
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
