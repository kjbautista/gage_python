"""Runtime configuration loaded once per process.

Precedence: explicit constructor arg (when code passes one) > environment
variables > ``run_gage/config.local.json`` next to this file > documented
defaults. The merged ``GageConfig`` is logged at INFO on first load so every
debug session records the exact values that ran.

Two things are intentionally NOT in here:

- Hardware-SDK-level constants (``SAMPLE_ALIGNMENT``, ``CS_SAMPLE_OFFSET_DEFAULT``,
  ``TRIGGER_LEVEL_FULL_SCALE_V``) live in ``controller.py`` because they are
  tied to the Gage SDK contract and are not user-tunable.
- GUI widget cosmetics (fonts, margins) stay in ``gui/style_utils.py``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


LOGGER = logging.getLogger(__name__)

_CONFIG_FILE_NAME = 'config.local.json'
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / _CONFIG_FILE_NAME


@dataclass(frozen=True)
class XpsConnectionConfig:
    """Newport XPS connection parameters."""

    #: Controller IP on the lab network. (XPS factory default)
    ip: str = '192.168.0.254'
    #: Login user — defaults to the XPS factory account. (XPS factory default)
    username: str = 'Administrator'
    #: Login password — defaults to the XPS factory account. (XPS factory default)
    password: str = 'Administrator'
    #: TCP timeout in seconds when talking to the controller.
    socket_timeout_s: float = 10.0
    #: Per-group init/home polling deadline, enforced by ``xps_control_main._wait_for_group_state``.
    #: Stages with long travel can take 30 s+ to home from a far end, so the floor is generous.
    #: Override via ``GAGE_XPS_INIT_TIMEOUT_S`` or ``config.local.json`` for slower hardware.
    initialize_timeout_s: float = 60.0
    #: How often to poll group status while waiting for init/home to complete.
    initialize_poll_s: float = 0.5
    #: Position window (mm) within which the post-move position check is considered a match.
    #: Newport XPS move calls are blocking, so this is a verification threshold — not a
    #: settling wait — and it catches silent servo errors where the ack fired but the
    #: stage did not actually arrive.
    motion_tolerance_mm: float = 0.001
    #: Threshold below which a commanded delta is treated as "no move" (filters float noise).
    motion_epsilon_mm: float = 1e-4
    #: Seconds to wait for the stage to settle after motion is done before proceeding with a capture.
    motion_settle_timeout_s: float = 1.0


@dataclass(frozen=True)
class MotionScanConfig:
    """Velocities and progress-bar budgets for raster scans."""

    #: Reduced velocity (mm/s) used when repositioning to scan start or home.
    scan_velocity_mm_s: float = 10.0


@dataclass(frozen=True)
class GuiTuningConfig:
    """Tunable GUI timing knobs."""

    #: Debounce window for parameter-change signals before reconfiguring hardware.
    reconfigure_debounce_ms: int = 250


@dataclass(frozen=True)
class GageConfig:
    """Root configuration tree."""

    xps: XpsConnectionConfig = field(default_factory=XpsConnectionConfig)
    motion: MotionScanConfig = field(default_factory=MotionScanConfig)
    gui: GuiTuningConfig = field(default_factory=GuiTuningConfig)


_CACHED_CONFIG: GageConfig | None = None


def _apply_env_overrides(config: XpsConnectionConfig) -> XpsConnectionConfig:
    """Return ``config`` with any environment-variable overrides applied."""

    mapping = {
        'GAGE_XPS_IP': 'ip',
        'GAGE_XPS_USER': 'username',
        'GAGE_XPS_PASSWORD': 'password',
        'GAGE_XPS_SOCKET_TIMEOUT_S': ('socket_timeout_s', float),
        'GAGE_XPS_INIT_TIMEOUT_S': ('initialize_timeout_s', float),
    }
    updates: dict[str, object] = {}
    for env_key, target in mapping.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        if isinstance(target, tuple):
            field_name, coercer = target
            updates[field_name] = coercer(value)
        else:
            updates[target] = value
    return replace(config, **updates) if updates else config


def _apply_file_overrides(config: GageConfig, path: Path) -> GageConfig:
    """Return ``config`` merged with any values present in a local JSON file."""

    if not path.is_file():
        return config
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        LOGGER.warning('Failed to parse %s; using defaults', path, exc_info=True)
        return config

    xps_updates = payload.get('xps') or {}
    motion_updates = payload.get('motion') or {}
    gui_updates = payload.get('gui') or {}
    return GageConfig(
        xps=replace(config.xps, **{k: xps_updates[k] for k in xps_updates if k in config.xps.__dataclass_fields__}),
        motion=replace(config.motion, **{k: motion_updates[k] for k in motion_updates if k in config.motion.__dataclass_fields__}),
        gui=replace(config.gui, **{k: gui_updates[k] for k in gui_updates if k in config.gui.__dataclass_fields__}),
    )


def load_gage_config(config_path: Path | None = None, *, refresh: bool = False) -> GageConfig:
    """Load the merged configuration, caching after the first call.

    Pass ``refresh=True`` to force a re-read (used by tests). The merged result
    is logged at INFO the first time it is loaded so the exact runtime values
    always appear in logs.
    """

    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not refresh:
        return _CACHED_CONFIG

    path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
    base = GageConfig()
    file_merged = _apply_file_overrides(base, path)
    xps_with_env = _apply_env_overrides(file_merged.xps)
    merged = replace(file_merged, xps=xps_with_env)

    safe_dump = asdict(merged)
    safe_dump['xps']['password'] = '***'
    LOGGER.info('Loaded gage_python configuration: %s (source: %s)', safe_dump, path)
    _CACHED_CONFIG = merged
    return merged
