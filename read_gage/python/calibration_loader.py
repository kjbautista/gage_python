from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / 'calibrations'
_CSV_PATH = CALIBRATION_DIR / 'calibration_lookup.csv'


@dataclass(frozen=True)
class CalibrationEntry:
    """One row from calibration_lookup.csv."""

    name: str                   # display label, e.g. 'HNC200_rightangle'
    file_stem: str              # e.g. 'h1177_p1477_rightangle'
    hydrophone_preamplifier: str  # e.g. 'HNC200-1177_AH2010-1477'
    connector: str              # e.g. 'rightangle' or 'None'
    attenuator: str             # e.g. 'None'
    gain: str                   # e.g. 'None'


@lru_cache(maxsize=1)
def get_catalog() -> tuple[CalibrationEntry, ...]:
    """Read ``calibration_lookup.csv`` and return all entries as an immutable tuple.

    The result is cached after the first call so the file is read only once per
    interpreter session.
    """
    entries: list[CalibrationEntry] = []
    with open(_CSV_PATH, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            entries.append(CalibrationEntry(
                name=row['name'],
                file_stem=Path(row['calibrationFile']).stem,
                hydrophone_preamplifier=row.get('hydrophone_preamplifier', 'None') or 'None',
                connector=row.get('connector', 'None') or 'None',
                attenuator=row.get('attenuator', 'None') or 'None',
                gain=row.get('gain', 'None') or 'None',
            ))
    return tuple(entries)


def get_names() -> list[str]:
    """Return ``['None']`` followed by the name of every entry in the catalog."""
    return ['None'] + [entry.name for entry in get_catalog()]


def get_entry_by_name(name: str) -> CalibrationEntry | None:
    """Return the ``CalibrationEntry`` whose *name* matches, or ``None``."""
    for entry in get_catalog():
        if entry.name == name:
            return entry
    return None


def _load_onda_txt(path: Path) -> np.ndarray:
    """Parse an Onda calibration .txt file.

    Returns an ``Nx2`` float array of ``[freq_mhz, sens_v_per_pa]``.
    Columns in the file: FREQ_MHZ  SENS_DB  SENS_VPERPA  SENS_V2CM2PERW.
    """
    rows: list[tuple[float, float]] = []
    past_header = False
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not past_header:
                if line.startswith('HEADER_END'):
                    past_header = True
                continue
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            rows.append((float(parts[0]), float(parts[2])))  # freq, SENS_VPERPA
    return np.array(rows, dtype=float)


def load_calibration_data(file_stem: str) -> np.ndarray:
    """Load the ``.txt`` calibration file and return an ``Nx2`` array of
    ``[freq_mhz, sensitivity]``.

    Returns an empty ``(0, 2)`` array if *file_stem* is empty or the file does
    not exist.
    """
    if not file_stem:
        return np.empty((0, 2))
    cal_file = CALIBRATION_DIR / f'{file_stem}.txt'
    if not cal_file.is_file():
        LOGGER.info('No calibration file found for stem %r', file_stem)
        return np.empty((0, 2))
    return _load_onda_txt(cal_file)
