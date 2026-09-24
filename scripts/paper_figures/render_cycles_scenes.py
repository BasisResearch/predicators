"""Render saved, non-EGL scene exports sequentially with Blender Cycles."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
GENERATED_BY = ('scripts/paper_figures/render_cycles_scenes.py; '
                'do not edit manually')


def digest(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(manifest: Path, key: str, entry: Dict[str, Any]) -> None:
    """Write one render's entry into the manifest.

    Render jobs may run at the same time, so each write re-reads the
    manifest under a lock and changes only its own entry. Retired
    renders leave the manifest with their files.
    """
    manifest.touch()
    with manifest.open('r+') as handle:
        fcntl.lockf(handle, fcntl.LOCK_EX)
        text = handle.read()
        report = json.loads(text) if text else {'files': {}}
        report['generated_by'] = GENERATED_BY
        report['files'] = {
            name: value
            for name, value in report['files'].items()
            if (ROOT / name).exists()
        }
        report['files'][key] = entry
        handle.seek(0)
        handle.write(json.dumps(report, indent=2) + '\n')
        handle.truncate()


def main() -> None:
    """Render the selected archived scenes and update their manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender-python', default=sys.executable)
    parser.add_argument(
        '--domains',
        nargs='+',
        default=['Bridge', 'Balloons', 'Boil', 'Fan', 'Domino'])
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip-trajectories', action='store_true')
    parser.add_argument(
        '--scenes',
        nargs='+',
        help='Render only these scene stems, such as trajectory_bridge_0.')
    parser.add_argument(
        '--log-dir',
        type=Path,
        help='Keep renderer logs here. By default logs are temporary.')
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Print selected scene stems without rendering.')
    args = parser.parse_args()
    manifest = ROOT / 'data/cycles-render-manifest.json'
    report = json.loads(manifest.read_text()) if manifest.exists() else {
        'files': {}
    }
    renderer = ROOT / 'render_cycles_scene.py'
    jobs = []
    selected_domains = {domain.lower() for domain in args.domains}
    for domain in args.domains:
        for frame in ('start', 'win'):
            source = ROOT / f'data/cycles_scenes/{domain.lower()}_{frame}.json'
            output = ROOT / ('figures/sources/'
                             f'{domain.lower()}_cycles_{frame}.png')
            jobs.append((domain, frame, source, output))
    if not args.skip_trajectories:
        for source in sorted(
            (ROOT / 'data/cycles_scenes').glob('trajectory_*.json')):
            name = source.stem
            domain = name.split('_')[1].title()
            if domain.lower() not in selected_domains:
                continue
            output = ROOT / f'figures/sources/{name}_cycles.png'
            jobs.append((domain, name, source, output))
        if 'bridge' in selected_domains:
            for name in ('bridge_pair_predicted', 'bridge_pair_observed'):
                source = ROOT / f'data/cycles_scenes/{name}.json'
                output = ROOT / f'figures/sources/{name}_cycles.png'
                jobs.append(('Bridge', name, source, output))
    if args.scenes:
        selected_scenes = set(args.scenes)
        jobs = [job for job in jobs if job[2].stem in selected_scenes]
        missing = selected_scenes - {job[2].stem for job in jobs}
        if missing:
            parser.error(f'Unknown or excluded scenes: {sorted(missing)}')
    if args.dry_run:
        for _, _, source, _ in jobs:
            print(source.stem)
        return

    temporary_logs = None
    if args.log_dir is None:
        temporary_logs = tempfile.TemporaryDirectory(
            prefix='empiric-cycles-logs-')
        log_dir = Path(temporary_logs.name)
    else:
        log_dir = args.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

    for domain, frame, source, output in jobs:
        key = str(output.relative_to(ROOT))
        stamp = dict(renderer_sha256=digest(renderer),
                     scene_sha256=digest(source),
                     samples=args.samples)
        old = report['files'].get(key, {})
        if (not args.force and all(old.get(k) == v for k, v in stamp.items())
                and output.exists() and old.get('sha256') == digest(output)):
            continue
        log = log_dir / f'{domain.lower()}_{frame}.log'
        with log.open('w') as stream:
            subprocess.run(
                [
                    args.blender_python,
                    str(renderer),
                    str(source), '--output',
                    str(output), '--samples',
                    str(args.samples), '--threads',
                    str(args.threads)
                ],
                cwd=ROOT,
                env={
                    **os.environ, 'OPENBLAS_NUM_THREADS': '1',
                    'OMP_NUM_THREADS': str(args.threads),
                    'EMPIRIC_RENDER_THREADS': str(args.threads)
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True)
        scene_metadata = json.loads(source.read_text()).get('metadata', {})
        record(
            manifest, key,
            dict(**stamp,
                 sha256=digest(output),
                 domain=domain,
                 frame=frame,
                 scene=str(source.relative_to(ROOT)),
                 scorecard_sha256=scene_metadata.get('scorecard_sha256'),
                 physics_steps_after_restore=0,
                 body_poses_and_joints_unchanged=True))
        print(f'Rendered {domain} {frame}', flush=True)
    if temporary_logs is not None:
        temporary_logs.cleanup()


if __name__ == '__main__':
    main()
