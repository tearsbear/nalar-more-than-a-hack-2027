import argparse
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

from .config import Config, official_layout
from .data import DataError, Features, digest, load_inputs
from .inference import D, LABELS, infer, interpolate_ball, summarize

ROOT = Path(__file__).resolve().parent.parent


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def csv_bytes(labels):
    if labels.ndim != 1 or not np.isin(labels, np.arange(4)).all():
        raise ValueError('invalid output labels')
    return ('frame,label\n' + ''.join(f'{i},{label}\n' for i, label in enumerate(LABELS[labels]))).encode('utf-8')


def source_hash():
    h = hashlib.sha256()
    for path in [ROOT / 'games.py', ROOT / 'starter/load_smart_tagging.py', *sorted((ROOT / 'nalar').glob('*.py'))]:
        h.update(path.relative_to(ROOT).as_posix().encode() + b'\0')
        h.update(path.read_bytes())
    return h.hexdigest()


def git_state():
    def run(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {'revision': run('rev-parse', 'HEAD'), 'tracked_changes': bool(run('diff', 'HEAD', '--name-only'))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'revision': None, 'tracked_changes': None}


def peak_memory():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def check_outputs(args):
    outputs = [Path(args.output).resolve()]
    if args.command == 'generate':
        outputs.append(Path(args.manifest).resolve())
    reserved = [Path(args.data_dir).resolve(), *(ROOT / name for name in
                ('nalar', 'configs', 'starter', 'evaluation/gt'))]
    protected = {Path(args.config).resolve(), ROOT / 'games.py'}
    protected.update((ROOT / 'evaluation/results').glob('baseline_*.csv'))
    protected.update(ROOT.glob('*.md'))
    protected.update((ROOT / 'evaluation').glob('*.py'))
    if len(set(outputs)) != len(outputs) or any(
            p in protected or any(p.is_relative_to(folder) for folder in reserved) for p in outputs):
        raise ValueError('outputs would overwrite protected inputs or source files')


def diagnose(features, events, layout, config):
    base = infer(features, events, layout, config)
    rng = np.random.default_rng(2027)
    observed = features.observed.copy()
    observed[rng.random(layout.n_frames) < 0.1] = False
    ball, distances = features.ball.copy(), features.distances.copy()
    ball[~observed], distances[~observed] = np.nan, np.inf
    noisy_distances = np.maximum(0, features.distances + rng.normal(0, 0.1, features.distances.shape))
    dropout_features = Features(ball, distances, observed)
    noisy_features = Features(features.ball, noisy_distances, features.observed)
    variants = {
        'without_tags': infer(features, events, layout, config, use_tags=False),
        'without_dead_ball': infer(features, events, layout, config, use_dead_ball=False),
        'without_interpolation': infer(features, events, layout, replace(config, interpolation_frames=0)),
        'ten_percent_ball_dropout': infer(dropout_features, events, layout, config),
        'ten_cm_distance_noise': infer(noisy_features, events, layout, config),
    }
    return {'purpose': 'label-free sensitivity only; not evidence of accuracy or model selection',
            'seed': 2027, 'base': summarize(base, features, layout),
            'variants': {name: {'changed_frames': int(np.sum(labels != base)),
                                'diagnostics': summarize(labels, dropout_features if name == 'ten_percent_ball_dropout'
                                                         else features, layout)}
                         for name, labels in variants.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='NALAR deterministic offline possession inference')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('generate', 'diagnose'):
        sub = commands.add_parser(name)
        sub.add_argument('--data-dir', default=os.environ.get('MTAH_DATA_DIR', str(ROOT / 'data')))
        sub.add_argument('--config', default=str(ROOT / 'configs/nalar_v1.json'))
        if name == 'generate':
            sub.add_argument('--output', default=str(ROOT / 'evaluation/results/NALAR.csv'))
            sub.add_argument('--manifest', default=str(ROOT / 'artifacts/run.json'))
        else:
            sub.add_argument('--output', default=str(ROOT / 'artifacts/robustness.json'))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        check_outputs(args)
        config, layout = Config.load(args.config), official_layout()
        features, events, paths = load_inputs(args.data_dir, layout)
        if args.command == 'diagnose':
            report = diagnose(features, events, layout, config)
            atomic_write(args.output, (json.dumps(report, indent=2, sort_keys=True) + '\n').encode())
            print(json.dumps(report, sort_keys=True))
            return 0
        labels = infer(features, events, layout, config)
        payload = csv_bytes(labels)
        output_hash = hashlib.sha256(payload).hexdigest()
        diagnostics = summarize(labels, features, layout)
        continuity = interpolate_ball(features, layout, config.interpolation_frames, blocked=labels == D)
        diagnostics['interpolated_continuity_frames'] = int(np.sum(
            np.isfinite(continuity).all(axis=1) & ~features.observed))
        manifest = {
            'method': 'nalar_v1', 'configuration': asdict(config),
            'configuration_sha256': digest(args.config), 'source_sha256': source_hash(),
            'code': git_state(), 'input_sha256': {key: digest(path) for key, path in sorted(paths.items())},
            'environment': {'python': platform.python_version(), 'numpy': version('numpy'),
                            'pandas': version('pandas'), 'platform': platform.platform()},
            'output_sha256': output_hash, 'diagnostics': diagnostics,
            'elapsed_seconds': round(time.perf_counter() - started, 3), 'peak_rss_bytes': peak_memory(),
            'ground_truth_used': False,
        }
        atomic_write(args.output, payload)
        atomic_write(args.manifest, (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode())
        print(json.dumps({'output_sha256': output_hash, 'diagnostics': manifest['diagnostics'],
                          'elapsed_seconds': manifest['elapsed_seconds'], 'peak_rss_bytes': manifest['peak_rss_bytes']}, sort_keys=True))
        return 0
    except DataError as error:
        print(f'NALAR failed: {error}', file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError):
        # Exception text from parsers can contain the offending restricted input.
        print('NALAR failed: check local input files, configuration, and output paths; no output is valid until a successful run.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
