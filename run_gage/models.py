"""Typed configuration, exception hierarchy, and acquisition-frame container.

Exception hierarchy:
    GageGuiError (RuntimeError)
    ├── ConfigurationError            — invalid settings, hardware config commit failed
    ├── TriggerTimeoutError           — board did not trigger within the allowed timeout
    ├── AcquisitionStoppedError       — user-initiated stop while capture in flight
    ├── TransferError                 — board failed to transfer captured data
    ├── AcquisitionInconsistencyError — saved frames disagree on time axis or geometry
    ├── MotionTimeoutError            — stage did not reach commanded position in time
    └── StageInitializationError      — stage failed to initialize or home in time
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


ErrorCategory = Literal['transient', 'config', 'motion', 'hardware', 'cancelled', 'internal']


#: Default Gage CompuScope sample rate in Hz.
DEFAULT_SAMPLE_RATE_HZ = 200e6
#: Digitizer full-scale half-range options in volts (GUI combo entries).
SUPPORTED_INPUT_RANGES_V = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
#: Minimum effective acquisition timeout floor in seconds (used when config leaves it unset).
MIN_ACQUISITION_TIMEOUT_S = 5.0
#: Multiplicative safety margin applied to the expected frame duration when auto-computing a timeout.
ACQUISITION_TIMEOUT_SAFETY_FACTOR = 5.0
#: Fixed slack added after the scaled frame duration to allow for trigger-wait variance.
ACQUISITION_TIMEOUT_TRIGGER_SLACK_S = 5.0


class GageGuiError(RuntimeError):
    """Base class for application-specific GUI and acquisition errors."""


class ConfigurationError(GageGuiError):
    """Raised when GUI settings cannot be converted into valid hardware settings."""


class TriggerTimeoutError(GageGuiError):
    """Raised when the board does not trigger within the allowed timeout."""


class AcquisitionStoppedError(GageGuiError):
    """Raised when an in-flight acquisition is interrupted by the user."""


class TransferError(GageGuiError):
    """Raised when the board fails to transfer captured data."""


class AcquisitionInconsistencyError(GageGuiError):
    """Raised when captured frames disagree on time axis or geometry at save time."""


class MotionTimeoutError(GageGuiError):
    """Raised when a motion-stage move does not settle within the allowed timeout."""


class StageInitializationError(GageGuiError):
    """Raised when a motion-stage group fails to initialize or home within the allowed timeout."""


@dataclass(frozen=True)
class WorkerError:
    """Structured error payload emitted on worker ``error`` signals.

    The ``category`` lets a GUI decide whether to allow a retry vs. surface a
    hard-failure dialog. ``traceback_text`` captures the full traceback at the
    point the exception was caught so post-hoc debugging does not depend on
    live logs. ``recovery_errors`` carries non-fatal failures from cleanup
    paths (disconnects, home moves) that would otherwise be silently logged.
    """

    #: High-level bucket used by GUI to choose a response.
    category: ErrorCategory
    #: Short human-readable description of the failure (the exception's ``str``).
    message: str
    #: Full traceback captured at the catch site — non-empty except for synthesised errors.
    traceback_text: str = ''
    #: ``type(exc).__qualname__`` so handlers can distinguish related exceptions without imports.
    cause_qualname: str = ''
    #: Messages accumulated from non-fatal cleanup warnings during failure recovery.
    recovery_errors: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        recovery_errors: 'list[str] | tuple[str, ...] | None' = None,
    ) -> 'WorkerError':
        """Build a ``WorkerError`` from a live exception and optional recovery messages."""

        category = _classify_exception(exc)
        tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(
            category=category,
            message=str(exc),
            traceback_text=tb_text.rstrip(),
            cause_qualname=type(exc).__qualname__,
            recovery_errors=tuple(recovery_errors or ()),
        )

    def __str__(self) -> str:
        lines = [f'{self.cause_qualname or "Error"}: {self.message}']
        if self.traceback_text:
            lines.extend(['', self.traceback_text])
        if self.recovery_errors:
            lines.extend(['', 'Recovery warnings:', *[f'  - {m}' for m in self.recovery_errors]])
        return '\n'.join(lines)


def _classify_exception(exc: BaseException) -> ErrorCategory:
    """Map an exception to a ``WorkerError.category`` for GUI dispatch."""

    if isinstance(exc, ConfigurationError):
        return 'config'
    if isinstance(exc, (MotionTimeoutError, StageInitializationError)):
        return 'motion'
    if isinstance(exc, TriggerTimeoutError):
        return 'transient'
    if isinstance(exc, (TransferError, AcquisitionInconsistencyError)):
        return 'hardware'
    if isinstance(exc, AcquisitionStoppedError):
        return 'cancelled'
    return 'internal'


@dataclass(frozen=True)
class AcquisitionConfig:
    """Strongly typed acquisition settings collected from the GUI.

    The meaning of ``n_alines`` depends on ``data_type``:
    - ``'aline'``: number of frames to save (SaveAcquisitionWorker loop bound).
    - ``'mmode'``: ring buffer capacity (column count in the M-mode buffer).
    - ``'beammap'``: frames averaged per scan position (lines_average_spin value).
    """

    #: Inclusive start of the captured time window in microseconds (may be negative for pre-trigger).
    t_start_us: float
    #: Exclusive end of the captured time window in microseconds.
    t_end_us: float
    #: Digitizer sample rate in Hz.
    sample_rate_hz: float
    #: External trigger level in volts (must fall within [0.1, 5.0] for the current hardware).
    trigger_level_v: float
    #: Channel input half-range in volts; must equal one of SUPPORTED_INPUT_RANGES_V.
    input_range_v: float
    #: DC offset applied to the channel in millivolts.
    dc_offset_mv: float
    #: Meaning depends on data_type — see class docstring.
    n_alines: int
    #: Kind of acquisition this config parameterises: 'aline' | 'mmode' | 'beammap'.
    data_type: str = 'aline'
    #: File stem of the matched calibration entry (e.g. 'h1177_p1477_rightangle').
    #: Resolved at config-collection time by calibration_loader.lookup_file_stem().
    #: Empty string means no matching calibration entry was found.
    calibration_file: str = ''
    #: Per-acquisition timeout in seconds. None → compute from depth/sample-rate via effective_timeout_s().
    acquisition_timeout_s: float | None = None

    def validate(self) -> None:
        """Validate settings before attempting to configure hardware."""

        if self.t_end_us <= self.t_start_us:
            raise ConfigurationError('Time end must be greater than time start.')
        if self.t_end_us - self.t_start_us > 20e6:
            raise ConfigurationError('Time duration must be less than or equal to 20 seconds.')
        if self.sample_rate_hz <= 0:
            raise ConfigurationError('Sample rate must be positive.')
        if self.input_range_v not in SUPPORTED_INPUT_RANGES_V:
            raise ConfigurationError(f'Unsupported input range: {self.input_range_v}')
        if not 0.1 <= self.trigger_level_v <= 5.0:
            raise ConfigurationError(
                f'Trigger level must be between 0.1 V and 5.0 V (got {self.trigger_level_v} V).'
            )
        if self.n_alines < 1:
            raise ConfigurationError('Number of A-lines must be at least 1.')
        if self.acquisition_timeout_s is not None and self.acquisition_timeout_s <= 0:
            raise ConfigurationError('Acquisition timeout, when set, must be positive.')

    def effective_timeout_s(self) -> float:
        """Return the per-acquisition timeout in seconds.

        Uses ``acquisition_timeout_s`` when the caller set it; otherwise computes
        ``max(MIN_ACQUISITION_TIMEOUT_S, safety_factor * frame_duration + trigger_slack)``
        so long captures do not trip spurious ``TriggerTimeoutError``.
        """

        if self.acquisition_timeout_s is not None:
            return float(self.acquisition_timeout_s)
        frame_duration_s = max(0.0, (self.t_end_us - self.t_start_us) * 1e-6)
        computed = ACQUISITION_TIMEOUT_SAFETY_FACTOR * frame_duration_s + ACQUISITION_TIMEOUT_TRIGGER_SLACK_S
        return float(max(MIN_ACQUISITION_TIMEOUT_S, computed))


@dataclass(frozen=True)
class DisplayConfig:
    """Display-only settings that do not require hardware reconfiguration."""

    #: Symmetric voltage half-range displayed on the A-line plot, in volts.
    displayed_voltage_range_v: float
    #: Vertical offset subtracted from waveforms for display AND FFT computation, in volts.
    plot_offset_v: float
    #: Lower bound of the FFT frequency display window in MHz.
    freq_start_mhz: float
    #: Upper bound of the FFT frequency display window in MHz.
    freq_end_mhz: float
    #: Number of frames to average for display smoothing and per-position capture counts.
    lines_to_average: int

    def normalized_frequency_limits(self) -> tuple[float, float]:
        """Return a valid frequency interval even if the UI range is inverted."""

        if self.freq_end_mhz <= self.freq_start_mhz:
            return self.freq_start_mhz, self.freq_start_mhz + 1.0
        return self.freq_start_mhz, self.freq_end_mhz

    def apply_plot_offset(self, volts_data):
        """Return data transformed exactly as it should appear in the UI/export path."""

        return np.asarray(volts_data) - self.plot_offset_v


def compute_fft_mag(data: np.ndarray) -> np.ndarray:
    fft_length = int(2 ** np.ceil(np.log2(data.size)))
    fft_mag = np.abs(np.fft.rfft(data, n=fft_length))
    return 20 * np.log10(fft_mag / np.max(fft_mag))


@dataclass
class AcquisitionFrame:
    """Container for one acquired A-line and its derived plot data."""

    time_axis_us: np.ndarray
    volts_data: np.ndarray
    freq_axis_mhz: np.ndarray
    fft_mag: np.ndarray
    min_voltage: float
