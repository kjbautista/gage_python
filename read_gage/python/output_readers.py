from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np


BEAMMAP_PARAMETERS_PATTERN = re.compile(r'^acquisition_parameters_(\d{8}_\d{6})\.json$')
BEAMMAP_POSITION_PATTERN = re.compile(r'^voltage_data_pos(\d{4,})_(\d{8}_\d{6})\.npy$')


def read_aline_output(path: str | Path, *, calibration_file_path: str | Path | None = None) -> dict[str, Any]:
    """Load an A-line ``.npy`` export and its JSON sidecar."""

    data_path = Path(path)
    metadata_path = data_path.with_name(f'{data_path.stem}_parameters.json')
    metadata = _load_json(metadata_path)
    save_matrix = _load_numeric_matrix(data_path)
    acq_cfg = metadata.get('acquisition_config', {})
    voltage_data_v = save_matrix[:, 1:]
    cal_data, cal_label = _resolve_calibration(
        override=calibration_file_path,
        sidecar_path=data_path.with_suffix('.cal'),
    )
    return {
        'data_path': data_path,
        'metadata_path': metadata_path,
        'metadata': metadata,
        'time_axis_us': save_matrix[:, 0],
        'voltage_data_v': voltage_data_v,
        'pressure_data_pa': _voltage_to_pressure(
            voltage_data_v,
            float(acq_cfg.get('sample_rate_hz', 0.0)),
            cal_data,
            cal_label,
        ),
    }


def read_mmode_output(path: str | Path, *, calibration_file_path: str | Path | None = None) -> dict[str, Any]:
    """Load an M-mode ``.npy`` export and its JSON sidecar."""

    data_path = Path(path)
    metadata_path = data_path.with_name(f'{data_path.stem}_parameters.json')
    metadata = _load_json(metadata_path)
    save_matrix = _load_numeric_matrix(data_path)
    acq_cfg = metadata.get('acquisition_config', {})
    voltage_data_v = save_matrix[:, 1:]
    cal_data, cal_label = _resolve_calibration(
        override=calibration_file_path,
        sidecar_path=data_path.with_suffix('.cal'),
    )
    return {
        'data_path': data_path,
        'metadata_path': metadata_path,
        'metadata': metadata,
        'time_axis_us': save_matrix[:, 0],
        'voltage_data_v': voltage_data_v,
        'pressure_data_pa': _voltage_to_pressure(
            voltage_data_v,
            float(acq_cfg.get('sample_rate_hz', 0.0)),
            cal_data,
            cal_label,
        ),
    }


def read_beammap_output(path: str | Path, *, calibration_file_path: str | Path | None = None) -> dict[str, Any]:
    """Load a Beammap export folder."""

    parameters_path = _resolve_beammap_parameters_path(Path(path))
    save_dir = parameters_path.parent
    metadata = _load_json(parameters_path)
    timestamp = _beammap_timestamp(parameters_path, metadata)
    acq_cfg = metadata.get('acquisition_config', {})
    sample_rate_hz = float(acq_cfg.get('sample_rate_hz', 0.0))
    cal_data, cal_label = _resolve_calibration(
        override=calibration_file_path,
        sidecar_path=save_dir / f'cal_{timestamp}.cal',
    )

    coordinates_mm = _load_numeric_matrix(save_dir / f'coordinates_mm_{timestamp}.npy')

    position_entries: list[tuple[int, Path]] = []
    for file_path in save_dir.glob(f'voltage_data_pos*_{timestamp}.npy'):
        match = BEAMMAP_POSITION_PATTERN.match(file_path.name)
        if match is None:
            continue
        position_entries.append((int(match.group(1)), file_path))
    position_entries.sort(key=lambda entry: entry[0])

    raw_dim1 = np.unique(coordinates_mm[:, 0])
    raw_dim2 = np.unique(coordinates_mm[:, 1])
    raw_dim3 = np.unique(coordinates_mm[:, 2])
    dim1 = raw_dim1 - raw_dim1.mean()
    dim2 = raw_dim2 - raw_dim2.mean()
    dim3 = raw_dim3 - raw_dim3.mean()
    n1, n2, n3 = raw_dim1.size, raw_dim2.size, raw_dim3.size

    voltage_peak_pos = np.zeros((n1, n2, n3), dtype=float)
    voltage_peak_neg = np.zeros((n1, n2, n3), dtype=float)
    has_cal = cal_data is not None and cal_data.shape[0] > 0
    pressure_peak_pos: np.ndarray | None = np.zeros((n1, n2, n3), dtype=float) if has_cal else None
    pressure_peak_neg: np.ndarray | None = np.zeros((n1, n2, n3), dtype=float) if has_cal else None

    for i, (_, file_path) in enumerate(position_entries):
        position_matrix = _load_numeric_matrix(file_path)
        voltage_trace = position_matrix.mean(axis=1)

        coord = coordinates_mm[i]
        i1 = int(np.searchsorted(raw_dim1, coord[0]))
        i2 = int(np.searchsorted(raw_dim2, coord[1]))
        i3 = int(np.searchsorted(raw_dim3, coord[2]))
        voltage_peak_pos[i1, i2, i3] = float(voltage_trace.max())
        voltage_peak_neg[i1, i2, i3] = float(voltage_trace.min())

        if pressure_peak_pos is not None and pressure_peak_neg is not None:
            pressure_trace = _voltage_to_pressure(
                voltage_trace[:, np.newaxis], sample_rate_hz, cal_data, cal_label, coord,
            )
            if pressure_trace is not None:
                pressure_peak_pos[i1, i2, i3] = float(pressure_trace.max())
                pressure_peak_neg[i1, i2, i3] = float(pressure_trace.min())

    return {
        'coordinates_mm': coordinates_mm,
        'voltage_peak_pos': voltage_peak_pos,
        'voltage_peak_neg': voltage_peak_neg,
        'pressure_peak_pos': pressure_peak_pos,
        'pressure_peak_neg': pressure_peak_neg,
        'dim1': dim1,
        'dim2': dim2,
        'dim3': dim3,
        'save_dir': save_dir,
    }


def _voltage_to_pressure(
    voltage_data_v: np.ndarray,
    sample_rate_hz: float,
    calibration_data: np.ndarray | None,
    cal_label: str,
    coord: np.ndarray | None = None,
) -> np.ndarray | None:
    """Convert voltage data to pressure (Pa) via FFT-based calibration.

    *calibration_data* must be an ``Nx2`` array of ``[freq_mhz, sens_v_per_pa]``.
    Returns ``None`` if no calibration is available.
    """
    if calibration_data is None or calibration_data.shape[0] == 0:
        return None
    n_samples = voltage_data_v.shape[0]
    voltage_data_v = voltage_data_v - voltage_data_v.mean(axis=0)  # Remove DC offset before FFT
    nfft = int(2 ** np.ceil(np.log2(n_samples)))
    cal_freq_hz = calibration_data[:, 0] * 1e6
    press_sens_pa_v = 1.0 / calibration_data[:, 1]
    nyquist_hz = sample_rate_hz / 2.0
    cal_max_hz = float(cal_freq_hz.max())
    freq_hz = nyquist_hz * np.linspace(0.0, 1.0, nfft // 2 + 1)
    fdata = np.fft.fft(voltage_data_v, n=nfft, axis=0)
    fmag_onesided = np.abs(fdata[:nfft // 2 + 1, :]).mean(axis=1)
    peak_freq_hz = float(freq_hz[int(np.argmax(fmag_onesided))])
    cal_min_hz = float(cal_freq_hz.min())
    if not (cal_min_hz <= peak_freq_hz <= cal_max_hz):
        coord_prefix = f'Coordinate: {np.asarray(coord).tolist()}\n' if coord is not None else ''
        warnings.warn(
            f'{coord_prefix}'
            f'Peak signal frequency ({peak_freq_hz / 1e6:.2f} MHz) is outside the calibration '
            f'frequency range ({cal_min_hz / 1e6:.2f}–{cal_max_hz / 1e6:.2f} MHz) for "{cal_label}". '
            'Sensitivity is set to zero outside the calibration range.',
            UserWarning,
            stacklevel=3,
        )
    press_sens_interp = np.interp(freq_hz, cal_freq_hz, press_sens_pa_v, left=0.0, right=0.0)
    # Build two-sided spectrum matching numpy FFT bin ordering:
    # [DC, f1,...,f(N/2-1), Nyq, f(N/2-1),...,f1]  length = nfft
    sens2 = np.concatenate([press_sens_interp[:-1], press_sens_interp[::-1]])[:-1]
    fcorr = fdata * sens2[:, np.newaxis]
    pressure = np.real(np.fft.ifft(fcorr, n=nfft, axis=0))
    return pressure[:n_samples, :]


def _resolve_calibration(
    *,
    override: str | Path | None,
    sidecar_path: Path,
) -> tuple[np.ndarray | None, str]:
    """Load the Nx2 ``[freq_mhz, sens_v_per_pa]`` calibration array used for
    pressure conversion.

    Resolution order:
    1. ``override`` — full path to an Onda ``.txt`` file (highest priority).
    2. ``sidecar_path`` — the ``_cal.npy`` saved next to the data file.
    3. Neither available → warn and return ``(None, '')``.
    """

    if override:
        cal_path = Path(override)
        if not cal_path.is_file():
            warnings.warn(
                f'Calibration override not found: {cal_path}. Voltage data will not be converted to pressure.',
                UserWarning,
                stacklevel=3,
            )
            return None, cal_path.stem
        from .calibration_loader import _load_onda_txt
        return _load_onda_txt(cal_path), cal_path.stem
    if sidecar_path.is_file():
        cal_data = np.asarray(np.load(sidecar_path), dtype=float)
        return cal_data, sidecar_path.stem
    warnings.warn(
        f'Calibration sidecar not found: {sidecar_path.name}. '
        'Voltage data will not be converted to pressure.',
        UserWarning,
        stacklevel=3,
    )
    return None, ''


def _resolve_beammap_parameters_path(path: Path) -> Path:
    if path.is_dir():
        matches = sorted(path.glob('acquisition_parameters_*.json'))
        return matches[0]
    return path


def _beammap_timestamp(parameters_path: Path, metadata: dict[str, Any]) -> str:
    filename_match = BEAMMAP_PARAMETERS_PATTERN.match(parameters_path.name)
    filename_timestamp = filename_match.group(1) if filename_match else None
    metadata_timestamp = metadata.get('timestamp')
    if isinstance(metadata_timestamp, str) and metadata_timestamp:
        return metadata_timestamp
    return filename_timestamp or ''


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_numeric_matrix(path: Path) -> np.ndarray:
    return np.asarray(np.atleast_2d(np.load(path)), dtype=float)
