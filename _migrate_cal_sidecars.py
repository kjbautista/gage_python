"""One-shot migration: generate <stem>.cal / cal_<ts>.cal sidecars for data
captured before the new save behavior landed.

Walks a target tree, reads every parameter JSON, and saves the matching
calibration sidecar next to it using the calibration stem stored in the JSON.
Also renames any older ``<stem>_cal.npy`` / ``cal_<ts>.npy`` sidecars from a
prior run to the new ``.cal`` extension so a single pass converts everything.

Idempotent: skips files where the sidecar already exists, where no calibration
was selected, or where the referenced calibration is no longer in the catalog.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from read_gage.python.calibration_loader import load_calibration_data

BEAMMAP_NAME_RE = re.compile(r'^acquisition_parameters_(\d{8}_\d{6})\.json$')


def cal_stem_from_json(parameters_path: Path) -> str:
    payload = json.loads(parameters_path.read_text(encoding='utf-8'))
    acq = payload.get('acquisition_config', {})
    return str(acq.get('calibration_file', '') or '')


def beammap_timestamp_from_json(parameters_path: Path) -> str:
    payload = json.loads(parameters_path.read_text(encoding='utf-8'))
    metadata_ts = payload.get('timestamp')
    if isinstance(metadata_ts, str) and metadata_ts:
        return metadata_ts
    match = BEAMMAP_NAME_RE.match(parameters_path.name)
    return match.group(1) if match else ''


def write_sidecar(sidecar_path: Path, cal_stem: str) -> str:
    """Return a one-word status: 'wrote', 'skip-empty', 'skip-exists', 'skip-missing-cal'."""
    if not cal_stem:
        return 'skip-empty'
    if sidecar_path.exists():
        return 'skip-exists'
    cal_data = load_calibration_data(cal_stem)
    if cal_data.size == 0:
        return 'skip-missing-cal'
    # Use a file handle so np.save does not auto-append `.npy` to the `.cal` path.
    with open(sidecar_path, 'wb') as fp:
        np.save(fp, cal_data)
    return 'wrote'


def migrate(root: Path) -> dict[str, int]:
    counts = {
        'wrote': 0,
        'renamed-from-npy': 0,
        'skip-empty': 0,
        'skip-exists': 0,
        'skip-missing-cal': 0,
    }

    json_paths: set[Path] = set()
    json_paths.update(root.rglob('*_parameters.json'))
    json_paths.update(root.rglob('acquisition_parameters_*.json'))

    for json_path in sorted(json_paths):
        if BEAMMAP_NAME_RE.match(json_path.name):
            timestamp = beammap_timestamp_from_json(json_path)
            if not timestamp:
                print(f'  [warn] no timestamp resolvable for {json_path}')
                continue
            sidecar_path = json_path.parent / f'cal_{timestamp}.cal'
            legacy_npy_path = json_path.parent / f'cal_{timestamp}.npy'
        else:
            data_stem = json_path.name.removesuffix('_parameters.json')
            sidecar_path = json_path.parent / f'{data_stem}.cal'
            legacy_npy_path = json_path.parent / f'{data_stem}_cal.npy'

        # If a previous-format sidecar exists and the new one does not, just rename.
        if legacy_npy_path.is_file() and not sidecar_path.exists():
            legacy_npy_path.rename(sidecar_path)
            counts['renamed-from-npy'] += 1
            print(f'  [renamed]          {sidecar_path.relative_to(root)}  (from {legacy_npy_path.name})')
            continue

        cal_stem = cal_stem_from_json(json_path)
        status = write_sidecar(sidecar_path, cal_stem)
        counts[status] += 1
        rel = sidecar_path.relative_to(root)
        if status == 'wrote':
            print(f'  [wrote]            {rel}  (from {cal_stem})')
        elif status == 'skip-missing-cal':
            print(f'  [skip-missing-cal] {rel}  (cal stem "{cal_stem}" not in catalogue)')
        elif status == 'skip-exists':
            print(f'  [skip-exists]      {rel}')
        # skip-empty is silent — many older files have no calibration set

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Top-level data directory to walk')
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f'error: {args.root} is not a directory', file=sys.stderr)
        return 2

    print(f'Migrating calibration sidecars under: {args.root}')
    counts = migrate(args.root)
    print()
    print('Summary:')
    for key, n in counts.items():
        print(f'  {key:18s} {n}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
