from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from read_gage.python.calibration_loader import load_calibration_data
from run_gage.config import load_gage_config
from run_gage.controller import GageAlineController
from run_gage.gui.constants import MOTION_GROUPS
from run_gage.models import (
    AcquisitionConfig,
    AcquisitionFrame,
    AcquisitionInconsistencyError,
    ConfigurationError,
    DisplayConfig,
    MotionTimeoutError,
    StageInitializationError,
    WorkerError,
    compute_fft_mag,
)


LOGGER = logging.getLogger(__name__)

#: Relative tolerance passed to ``np.allclose`` when asserting that all saved frames
#: share the same time axis. Kept at 0 so only the absolute floor below applies.
TIME_AXIS_RTOL = 0.0
#: Absolute tolerance (microseconds) for time-axis equality across saved frames.
TIME_AXIS_ATOL_US = 1e-9


def _scan_velocity_mm_s() -> float:
    return load_gage_config().motion.scan_velocity_mm_s


def _motion_tolerance_mm() -> float:
    return load_gage_config().xps.motion_tolerance_mm


def _motion_epsilon_mm() -> float:
    return load_gage_config().xps.motion_epsilon_mm

def _motion_settle_timeout_s() -> float:
    return load_gage_config().xps.motion_settle_timeout_s


def build_metadata_payload(
    acquisition_config: AcquisitionConfig,
    display_config: DisplayConfig,
    *,
    timestamp: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable metadata payload shared across acquisition saves."""

    payload: dict[str, object] = {
        'schema_version': 2,
        'timestamp': timestamp or datetime.now().strftime('%Y%m%d_%H%M%S'),
        'acquisition_config': asdict(acquisition_config),
        'display_config': asdict(display_config),
    }
    if extra:
        payload.update(extra)
    return payload


def write_json_sidecar(data_path: Path, payload: dict[str, object]) -> Path:
    """Write JSON metadata next to a saved data file using a consistent sidecar name."""

    metadata_path = data_path.with_name(f'{data_path.stem}_parameters.json')
    metadata_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return metadata_path


def write_calibration_sidecar(sidecar_path: Path, calibration_stem: str) -> Path | None:
    """Save the FREQ_MHZ and SENS_VPERPA columns of the selected calibration as
    an Nx2 NumPy array at *sidecar_path*.

    Written with the ``.cal`` suffix so the sidecar is visually distinct from
    the ``.npy`` data file, but the bytes are still NumPy's standard format
    (the readers use the magic header, not the extension). A file handle is
    used because ``np.save`` would otherwise auto-append ``.npy``.

    Captured alongside each data file so that read-back never needs the
    ``read_gage/calibrations/`` catalogue. Returns the sidecar path on success,
    or ``None`` when no calibration was selected or the lookup is empty.
    """

    if not calibration_stem:
        return None
    cal_data = load_calibration_data(calibration_stem)
    if cal_data.size == 0:
        return None
    with open(sidecar_path, 'wb') as fp:
        np.save(fp, cal_data)
    return sidecar_path


def build_display_fft_mag(frame: AcquisitionFrame, display_config: DisplayConfig) -> np.ndarray:
    """Return the FFT magnitude of the offset-adjusted waveform shown in the GUI.

    Used when display settings change (plot offset) and the cached FFT on the
    frame was computed against a different offset. Normalised to dB scale.
    """

    plotted_volts = display_config.apply_plot_offset(frame.volts_data)
    return compute_fft_mag(plotted_volts)


def _build_worker_error(exc: BaseException, recovery_errors: list[str]) -> WorkerError:
    """Build a ``WorkerError`` payload for emission on a worker ``error`` signal."""

    return WorkerError.from_exception(exc, recovery_errors=recovery_errors)


class _AcquisitionStopSignal(Exception):
    """Internal sentinel used to unwind optimizer callbacks on user stop."""


class LiveAcquisitionWorker(QObject):
    """Acquire frames continuously on a Qt worker thread."""

    frame_ready = pyqtSignal(object, object, object, float)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        controller: GageAlineController,
        acquisition_config: AcquisitionConfig,
        display_config: DisplayConfig,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.acquisition_config = acquisition_config
        self.display_config = display_config
        self._running = False
        self._stop_event = threading.Event()
        self._fft_history: deque[np.ndarray] = deque(maxlen=max(1, display_config.lines_to_average))

    @pyqtSlot()
    def run(self) -> None:
        """Start the continuous capture loop until stop is requested."""

        self._running = True
        self._stop_event.clear()
        timeout_s = self.acquisition_config.effective_timeout_s()
        self.status_changed.emit('Starting live acquisition...')
        LOGGER.info('LiveAcquisitionWorker starting: timeout_s=%.3f', timeout_s)
        try:
            while self._running:
                frame = self.controller.acquire_single(
                    timeout_s=timeout_s,
                    stop_event=self._stop_event,
                    display_config=self.display_config,
                )
                self._update_fft_average(frame.fft_mag)
                self.frame_ready.emit(frame, frame.freq_axis_mhz, self._fft_average(), frame.min_voltage)
        except Exception as exc:
            if self._running:
                self.error.emit(_build_worker_error(exc, []))
        finally:
            self._running = False
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        """Request worker shutdown after the current acquisition iteration."""

        self._running = False
        self._stop_event.set()

    def _update_fft_average(self, fft_mag: np.ndarray) -> None:
        """Maintain a rolling arithmetic mean for FFT display smoothing."""

        self._fft_history.append(np.asarray(fft_mag, dtype=np.float64))

    def _fft_average(self) -> np.ndarray:
        """Return the current rolling FFT average."""

        return np.mean(np.vstack(self._fft_history), axis=0)


class SaveAcquisitionWorker(QObject):
    """Acquire one frame and persist it to CSV on a Qt worker thread."""

    frame_ready = pyqtSignal(object)
    progress_changed = pyqtSignal(int)
    completed = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        controller: GageAlineController,
        acquisition_config: AcquisitionConfig,
        display_config: DisplayConfig,
        destination: Path,
        motion_position_mm: dict | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.acquisition_config = acquisition_config
        self.display_config = display_config
        self.destination = destination
        self.motion_position_mm = motion_position_mm
        self._stop_event = threading.Event()

    @pyqtSlot()
    def run(self) -> None:
        """Capture the requested frame batch and persist it using the existing text format."""

        try:
            timeout_s = self.acquisition_config.effective_timeout_s()
            LOGGER.info(
                'SaveAcquisitionWorker starting: n_alines=%d timeout_s=%.3f dest=%s',
                self.acquisition_config.n_alines, timeout_s, self.destination,
            )
            self.status_changed.emit('Capturing frames for save...')
            frames = []
            for index in range(self.acquisition_config.n_alines):
                if self._stop_event.is_set():
                    self.status_changed.emit('Save stopped.')
                    return
                frame = self.controller.acquire_single(
                    timeout_s=timeout_s,
                    stop_event=self._stop_event,
                    display_config=self.display_config,
                )
                frames.append(frame)
                self.frame_ready.emit(frame)
                self.progress_changed.emit(int(round(100.0 * (index + 1) / self.acquisition_config.n_alines)))
            self._write_text_matrix(frames)
            self.completed.emit(str(self.destination))
        except Exception as exc:
            self.error.emit(_build_worker_error(exc, []))
        finally:
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        """Request cancellation of an in-progress save sequence."""

        self._stop_event.set()

    def _write_text_matrix(self, frames: list[AcquisitionFrame]) -> None:
        """Persist captured A-lines in the existing time-plus-columns text format.

        Column 0 is the shared time axis, columns 1..N are offset-adjusted
        voltages for each frame. All frames must share the same time axis (same
        trigger delay and sample count) or the saved matrix would silently
        misalign time and voltage — so we assert consistency before writing.
        """

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            raise ValueError('No frames were captured for saving.')

        first_time_axis = np.asarray(frames[0].time_axis_us, dtype=float)
        for index, frame in enumerate(frames[1:], start=1):
            other_axis = np.asarray(frame.time_axis_us, dtype=float)
            if other_axis.size != first_time_axis.size or not np.allclose(
                other_axis, first_time_axis, rtol=TIME_AXIS_RTOL, atol=TIME_AXIS_ATOL_US,
            ):
                max_delta = (
                    float(np.max(np.abs(other_axis - first_time_axis)))
                    if other_axis.size == first_time_axis.size else float('inf')
                )
                raise AcquisitionInconsistencyError(
                    f'Frame {index} time axis diverges from frame 0 '
                    f'(size {other_axis.size} vs {first_time_axis.size}, max delta {max_delta} us). '
                    'Refusing to save a matrix where time and voltage would misalign.'
                )

        save_matrix = np.zeros((first_time_axis.size, len(frames) + 1))
        save_matrix[:, 0] = first_time_axis
        for column_index, frame in enumerate(frames, start=1):
            save_matrix[:, column_index] = self.display_config.apply_plot_offset(frame.volts_data)
        data_path = self.destination.with_suffix('.npy')
        np.save(data_path, save_matrix)
        extra: dict[str, object] = {'data_file': data_path.name, 'frame_count': len(frames)}
        if self.motion_position_mm is not None:
            extra['motion_position_mm'] = self.motion_position_mm
        write_json_sidecar(
            self.destination,
            build_metadata_payload(
                self.acquisition_config,
                self.display_config,
                extra=extra,
            ),
        )
        write_calibration_sidecar(
            data_path.with_suffix('.cal'),
            self.acquisition_config.calibration_file,
        )


class _MotionMixin:
    """Shared motion-verification helpers for workers that drive the XPS stage."""

    xps_backend: object
    xps: object
    _recovery_errors: list[str]

    def _verify_motion_reached(self, group_name: str, target_mm: float) -> float:
        """Confirm the stage actually reached ``target_mm`` after a move returned.

        The newportxps ``move_stage`` call is blocking: the TCP reply is only
        sent once the controller reports the move complete. So a single
        position read here is enough — it guards against silent servo errors
        where the controller ack-ed but the stage did not actually arrive.

        Raises:
            MotionTimeoutError: when the reported position differs from the
                target by more than ``xps.motion_tolerance_mm``. The caller's
                scan is aborted; any retry policy is the GUI's responsibility.
        """

        tolerance = _motion_tolerance_mm()
        stage_name = f'{group_name}.Pos'
        last_position = float(self.xps.get_stage_position(stage_name))
        settle_timeout = _motion_settle_timeout_s()
        settle_start_s = time.monotonic()
        while time.monotonic() - settle_start_s < settle_timeout:
            if abs(last_position - target_mm) <= tolerance:
                break
            time.sleep(0.001)
            last_position = float(self.xps.get_stage_position(stage_name))
        LOGGER.debug(
            'Motion verified: group=%s target=%.6g reached=%.6g',
            group_name, target_mm, last_position,
        )
        return last_position

    def _move_axis_verified(self, group_index: int, target_mm: float) -> None:
        """Move one axis (blocking call) and verify the reported position matches."""

        group_name = f'Group{group_index + 1}'
        move_start_s = time.monotonic()
        self.xps_backend.move_stage(self.xps, group_name, float(target_mm), relative=False)
        settled = self._verify_motion_reached(group_name, float(target_mm))
        LOGGER.debug(
            'move_axis_verified: group=%s target=%.6g reached=%.6g elapsed_ms=%.1f',
            group_name, target_mm, settled, (time.monotonic() - move_start_s) * 1000.0,
        )

    def _record_recovery_error(self, message: str, exc: BaseException) -> None:
        """Log a recovery-path failure at WARNING and stash it for the final error payload."""

        LOGGER.warning('%s: %s', message, exc, exc_info=True)
        self._recovery_errors.append(f'{message}: {exc!r}')


class BeammapWorker(QObject, _MotionMixin):
    """Run a motion-stage beammap scan on a Qt worker thread.

    Output files are tagged with a ``.in_progress`` marker on the save
    directory; on normal completion the marker is removed, and on error it is
    renamed to ``.failed`` and a short error report is written next to it.
    """

    frame_ready = pyqtSignal(object)
    beammap_point_ready = pyqtSignal(object, float, int, int)
    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    completed = pyqtSignal(str)
    error = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        controller: GageAlineController,
        acquisition_config: AcquisitionConfig,
        display_config: DisplayConfig,
        xps_backend,
        xps,
        scan_positions: np.ndarray,
        save_dir: Path,
        home_position: np.ndarray,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.acquisition_config = acquisition_config
        self.display_config = display_config
        self.xps_backend = xps_backend
        self.xps = xps
        self.scan_positions = np.asarray(scan_positions, dtype=float)
        self.save_dir = save_dir
        self.home_position = np.asarray(home_position, dtype=float)
        self.output_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._stop_event = threading.Event()
        self._recovery_errors: list[str] = []

    @pyqtSlot()
    def run(self) -> None:
        """Execute the full beammap scan and save metadata plus per-position A-lines."""

        self.save_dir.mkdir(parents=True, exist_ok=True)
        in_progress_marker = self.save_dir / '.in_progress'
        in_progress_marker.write_text(
            f'Beammap started at {self.output_timestamp}\n', encoding='utf-8',
        )
        run_exception: BaseException | None = None
        try:
            timeout_s = self.acquisition_config.effective_timeout_s()
            LOGGER.info('BeammapWorker starting: %d positions timeout_s=%.3f save_dir=%s',
                        int(self.scan_positions.shape[0]), timeout_s, self.save_dir)
            self.status_changed.emit(f'Beammap saving to {self.save_dir}')
            self._move_to_start_position()
            total_positions = int(self.scan_positions.shape[0])
            time_axis_us: np.ndarray | None = None
            for index, position in enumerate(self.scan_positions, start=1):
                if self._stop_event.is_set():
                    self.status_changed.emit('Beammap stopped.')
                    return

                if index > 1:
                    self._move_changed_groups(self.scan_positions[index - 2], position)

                averaged_frame, displayed_voltage_lines = self._capture_position_average(timeout_s=timeout_s)
                if time_axis_us is None:
                    time_axis_us = np.asarray(averaged_frame.time_axis_us, dtype=float)
                    self._write_metadata_outputs(time_axis_us)
                self._save_position_lines(index, displayed_voltage_lines)
                beammap_value = float(abs(np.min(self.display_config.apply_plot_offset(averaged_frame.volts_data))))
                self.frame_ready.emit(averaged_frame)
                self.beammap_point_ready.emit(position.copy(), beammap_value, index, total_positions)
                self.progress_changed.emit(int(round(index / total_positions * 100.0)))

            if time_axis_us is None:
                raise RuntimeError('No beammap data was captured.')
            self.completed.emit(str(self.save_dir))
        except Exception as exc:
            run_exception = exc
            self.error.emit(_build_worker_error(exc, self._recovery_errors))
        finally:
            self._move_home_best_effort()
            self._finalize_markers(in_progress_marker, run_exception)
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        """Request cancellation of the beammap scan."""

        self._stop_event.set()

    def _finalize_markers(self, in_progress_marker: Path, run_exception: BaseException | None) -> None:
        """Transition ``.in_progress`` to either removal (success) or ``.failed`` (with report)."""

        try:
            if run_exception is None and not self._stop_event.is_set():
                if in_progress_marker.exists():
                    in_progress_marker.unlink()
                return
            failed_marker = self.save_dir / '.failed'
            reason = 'stopped by user' if run_exception is None else f'{type(run_exception).__name__}: {run_exception}'
            payload = {
                'timestamp': self.output_timestamp,
                'reason': reason,
                'recovery_errors': list(self._recovery_errors),
            }
            failed_marker.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            if in_progress_marker.exists():
                in_progress_marker.unlink()
        except Exception:
            LOGGER.warning('Failed to update scan progress markers', exc_info=True)

    def _move_to_start_position(self) -> None:
        """Move all groups to the first scan position using reduced velocity first."""

        self.status_changed.emit('Moving to beammap start position...')
        velocity = _scan_velocity_mm_s()
        for group_name in MOTION_GROUPS:
            self.xps_backend.set_velocity(self.xps, group_name, velocity)

        first_position = self.scan_positions[0]
        for group_index, _ in enumerate(MOTION_GROUPS):
            self._move_axis_verified(group_index, float(first_position[group_index]))

        self.xps_backend.reset_velocities(self.xps)

    def _move_changed_groups(self, previous_position: np.ndarray, current_position: np.ndarray) -> None:
        """Move and verify only the axes whose target changed by more than the epsilon threshold."""

        epsilon = _motion_epsilon_mm()
        changed_groups = np.flatnonzero(np.abs(current_position - previous_position) > epsilon)
        for group_index in changed_groups:
            self._move_axis_verified(int(group_index), float(current_position[group_index]))

    def _capture_position_average(self, timeout_s: float) -> tuple[AcquisitionFrame, np.ndarray]:
        """Acquire and average multiple A-lines at a single beammap position."""

        frames: list[AcquisitionFrame] = []
        for _ in range(max(1, self.display_config.lines_to_average)):
            if self._stop_event.is_set():
                raise RuntimeError('Beammap stopped.')
            frames.append(
                self.controller.acquire_single(
                    timeout_s=timeout_s,
                    stop_event=self._stop_event,
                    display_config=self.display_config,
                )
            )

        raw_matrix = np.vstack([frame.volts_data for frame in frames])
        averaged_volts = np.mean(raw_matrix, axis=0)
        template_frame = frames[-1]
        averaged_fft_input = self.display_config.apply_plot_offset(averaged_volts)
        averaged_frame = AcquisitionFrame(
            time_axis_us=np.asarray(template_frame.time_axis_us, dtype=float),
            volts_data=np.asarray(averaged_volts, dtype=float),
            freq_axis_mhz=np.asarray(template_frame.freq_axis_mhz, dtype=float),
            fft_mag=compute_fft_mag(averaged_fft_input),
            min_voltage=float(np.min(averaged_volts)),
        )
        displayed_lines = np.vstack([self.display_config.apply_plot_offset(frame.volts_data) for frame in frames]).T
        return averaged_frame, displayed_lines

    def _write_metadata_outputs(self, time_axis_us: np.ndarray) -> None:
        """Persist the shared beammap metadata files once per scan."""

        parameters_payload = {
            'schema_version': 1,
            'timestamp': self.output_timestamp,
            'acquisition_config': asdict(self.acquisition_config),
            'display_config': asdict(self.display_config),
            'home_position_mm': self.home_position.tolist(),
        }
        self._output_path('acquisition_parameters', '.json').write_text(
            json.dumps(parameters_payload, indent=2),
            encoding='utf-8',
        )
        np.save(self._output_path('coordinates_mm', '.npy'), self.scan_positions)
        np.save(self._output_path('time_axis_us', '.npy'), time_axis_us)
        write_calibration_sidecar(
            self._output_path('cal', '.cal'),
            self.acquisition_config.calibration_file,
        )

    def _save_position_lines(self, position_index: int, displayed_voltage_lines: np.ndarray) -> None:
        """Persist all offset-adjusted A-lines for one beammap position."""

        file_path = self._output_path(f'voltage_data_pos{position_index:04d}', '.npy')
        np.save(file_path, displayed_voltage_lines)

    def _output_path(self, stem: str, suffix: str) -> Path:
        """Return a timestamped output path within the beammap save directory."""

        return self.save_dir / f'{stem}_{self.output_timestamp}{suffix}'

    def _move_home_best_effort(self) -> None:
        """Attempt to move the stage home; record failures for the final error payload."""

        try:
            velocity = _scan_velocity_mm_s()
            for group_name in MOTION_GROUPS:
                self.xps_backend.set_velocity(self.xps, group_name, velocity)
            for group_index, _ in enumerate(MOTION_GROUPS):
                self._move_axis_verified(group_index, float(self.home_position[group_index]))
            self.xps_backend.reset_velocities(self.xps)
        except Exception as exc:
            self._record_recovery_error('Homing after beammap failed', exc)


class _SpatialPeakSearchStopped(Exception):
    """Raised inside the optimizer objective to abort the search when stop is requested."""


class SpatialPeakWorker(QObject, _MotionMixin):
    """Run a two-phase spatial peak search on a Qt worker thread.

    Phase 1 — coarse raster scan over the full defined region.
    Phase 2 — Nelder-Mead simplex refinement starting from the coarse peak.
    On completion the stage is moved to the best position found.
    No data is saved to disk.
    """

    frame_ready = pyqtSignal(object)
    point_ready = pyqtSignal(object, float, int, int)
    progress_changed = pyqtSignal(int)
    phase_changed = pyqtSignal(str)
    peak_found = pyqtSignal(object, float)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        controller: GageAlineController,
        acquisition_config: AcquisitionConfig,
        display_config: DisplayConfig,
        xps_backend,
        xps,
        scan_positions: np.ndarray,
        home_position: np.ndarray,
        groups_to_move: np.ndarray,
        axes_limits: np.ndarray,
        refinement_tol: float = 0.01,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.acquisition_config = acquisition_config
        self.display_config = display_config
        self.xps_backend = xps_backend
        self.xps = xps
        self.scan_positions = np.asarray(scan_positions, dtype=float)
        self.home_position = np.asarray(home_position, dtype=float)
        self.groups_to_move = np.asarray(groups_to_move, dtype=bool)
        self.axes_limits = np.asarray(axes_limits, dtype=float)
        self.refinement_tol = float(refinement_tol)
        self._stop_event = threading.Event()
        self._recovery_errors: list[str] = []

        self._best_position: np.ndarray = home_position.copy()
        self._best_value: float = 0.0
        self._probe_index: int = 0  # global probe counter across both phases
        self._current_position: np.ndarray = home_position.copy()

    @pyqtSlot()
    def run(self) -> None:
        """Execute coarse scan then Nelder-Mead refinement, move to best position."""

        try:
            total_coarse = int(self.scan_positions.shape[0])
            self.status_changed.emit(f'Starting coarse scan with {total_coarse} positions...')
            self.phase_changed.emit('coarse')
            self._run_coarse_phase(total_coarse)
            if self._stop_event.is_set():
                self.status_changed.emit('Peak search stopped during coarse phase.')
                return
            self.status_changed.emit('Coarse phase complete. Starting Nelder-Mead refinement...')
            self.phase_changed.emit('refine')
            self._run_refinement_phase()
        except _SpatialPeakSearchStopped:
            self.status_changed.emit('Peak search stopped.')
        except Exception as exc:
            self.error.emit(_build_worker_error(exc, self._recovery_errors))
        finally:
            self._move_to_best_effort()
            self.peak_found.emit(self._best_position.copy(), self._best_value)
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        """Request cancellation of the peak search."""

        self._stop_event.set()

    # ------------------------------------------------------------------
    # Phase 1: coarse raster
    # ------------------------------------------------------------------

    def _run_coarse_phase(self, total_coarse: int) -> None:
        if total_coarse <= 0:
            raise ConfigurationError(
                'Spatial peak search has no scan positions — at least one axis must be active.'
            )
        timeout_s = self.acquisition_config.effective_timeout_s()
        self._move_to_start_position()
        for index, position in enumerate(self.scan_positions, start=1):
            if self._stop_event.is_set():
                return
            if index > 1:
                self._move_changed_groups(self.scan_positions[index - 2], position)
            self._current_position = np.asarray(position, dtype=float).copy()
            averaged_frame, _ = self._capture_position_average(timeout_s=timeout_s)
            metric = float(abs(np.min(self.display_config.apply_plot_offset(averaged_frame.volts_data))))
            if metric > self._best_value:
                self._best_value = metric
                self._best_position = position.copy()
            self._probe_index += 1
            LOGGER.debug(
                'coarse: probe=%d pos=%s metric=%.6g best=%.6g',
                self._probe_index, position.tolist(), metric, self._best_value,
            )
            self.frame_ready.emit(averaged_frame)
            self.point_ready.emit(position.copy(), metric, self._probe_index, total_coarse)
            self.progress_changed.emit(int(round(index / total_coarse * 50.0)))  # coarse = 0-50%

    # ------------------------------------------------------------------
    # Phase 2: Nelder-Mead simplex
    # ------------------------------------------------------------------

    def _run_refinement_phase(self) -> None:
        from scipy.optimize import minimize  # imported here to keep top-level imports minimal

        active_indices = np.flatnonzero(self.groups_to_move)
        x0 = self._best_position[active_indices]
        lo = self.axes_limits[active_indices, 0]
        hi = self.axes_limits[active_indices, 1]
        total_coarse = int(self.scan_positions.shape[0])
        probe_start = self._probe_index

        def objective(x: np.ndarray) -> float:
            if self._stop_event.is_set():
                raise _SpatialPeakSearchStopped
            full_position = self._best_position.copy()
            full_position[active_indices] = np.clip(x, lo, hi)
            self._move_changed_groups(self._current_stage_position(), full_position)
            self._current_position = full_position.copy()
            averaged_frame, _ = self._capture_position_average(
                timeout_s=self.acquisition_config.effective_timeout_s(),
            )
            metric = float(abs(np.min(self.display_config.apply_plot_offset(averaged_frame.volts_data))))
            if metric > self._best_value:
                self._best_value = metric
                self._best_position = full_position.copy()
            self._probe_index += 1
            elapsed_refine = self._probe_index - probe_start
            LOGGER.debug(
                'refine: probe=%d pos=%s metric=%.6g best=%.6g',
                self._probe_index, full_position.tolist(), metric, self._best_value,
            )
            self.frame_ready.emit(averaged_frame)
            self.point_ready.emit(full_position.copy(), metric, self._probe_index, total_coarse)
            self.progress_changed.emit(min(99, 50 + int(round(elapsed_refine / 50.0 * 50.0))))
            return -metric

        try:
            minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'xatol': self.refinement_tol, 'fatol': 1e-9, 'maxiter': 50, 'adaptive': True},
            )
        except (_SpatialPeakSearchStopped, MotionTimeoutError, StageInitializationError, ConfigurationError):
            # Hardware/configuration failures must propagate so the GUI sees a real error
            # instead of a silent successful completion at the last-known best.
            raise
        except Exception as exc:
            # Optimizer-internal numerical issues (e.g. degenerate simplex) are non-fatal:
            # we keep the running best from the coarse phase. Only this narrow class is demoted.
            self.status_changed.emit(f'Refinement warning: {exc}')

        self.status_changed.emit(
            f'Refinement complete. Best position: [{self._best_position[0]:.4f}, '
            f'{self._best_position[1]:.4f}, {self._best_position[2]:.4f}] mm  '
            f'value={self._best_value:.6g}'
        )
        self.progress_changed.emit(100)

    # ------------------------------------------------------------------
    # Motion helpers (mirror BeammapWorker patterns)
    # ------------------------------------------------------------------

    def _move_to_start_position(self) -> None:
        self.status_changed.emit('Moving to coarse scan start position...')
        velocity = _scan_velocity_mm_s()
        for group_name in MOTION_GROUPS:
            self.xps_backend.set_velocity(self.xps, group_name, velocity)
        first_position = self.scan_positions[0]
        for group_index, _ in enumerate(MOTION_GROUPS):
            self._move_axis_verified(group_index, float(first_position[group_index]))
        self.xps_backend.reset_velocities(self.xps)
        self._current_position = np.asarray(first_position, dtype=float).copy()

    def _move_changed_groups(self, previous_position: np.ndarray, current_position: np.ndarray) -> None:
        epsilon = _motion_epsilon_mm()
        changed_groups = np.flatnonzero(np.abs(current_position - previous_position) > epsilon)
        for group_index in changed_groups:
            self._move_axis_verified(int(group_index), float(current_position[group_index]))
        if changed_groups.size:
            self._current_position = np.asarray(current_position, dtype=float).copy()

    def _capture_position_average(self, timeout_s: float) -> tuple[AcquisitionFrame, np.ndarray]:
        frames: list[AcquisitionFrame] = []
        for _ in range(max(1, self.display_config.lines_to_average)):
            if self._stop_event.is_set():
                raise _SpatialPeakSearchStopped
            frames.append(
                self.controller.acquire_single(
                    timeout_s=timeout_s,
                    stop_event=self._stop_event,
                    display_config=self.display_config,
                )
            )
        raw_matrix = np.vstack([frame.volts_data for frame in frames])
        averaged_volts = np.mean(raw_matrix, axis=0)
        template_frame = frames[-1]
        averaged_fft_input = self.display_config.apply_plot_offset(averaged_volts)
        averaged_frame = AcquisitionFrame(
            time_axis_us=np.asarray(template_frame.time_axis_us, dtype=float),
            volts_data=np.asarray(averaged_volts, dtype=float),
            freq_axis_mhz=np.asarray(template_frame.freq_axis_mhz, dtype=float),
            fft_mag=compute_fft_mag(averaged_fft_input),
            min_voltage=float(np.min(averaged_volts)),
        )
        displayed_lines = np.vstack([self.display_config.apply_plot_offset(frame.volts_data) for frame in frames]).T
        return averaged_frame, displayed_lines

    def _current_stage_position(self) -> np.ndarray:
        """Query the XPS for the current absolute stage position."""
        try:
            _, positions = self.xps_backend.get_positions(self.xps)
            return np.asarray(positions, dtype=float)
        except Exception as exc:
            self._record_recovery_error('Stage position query failed during refinement', exc)
            return self._best_position.copy()

    def _move_to_best_effort(self) -> None:
        """Move stage to the best-found position; record failures for the final error payload."""
        try:
            velocity = _scan_velocity_mm_s()
            for group_name in MOTION_GROUPS:
                self.xps_backend.set_velocity(self.xps, group_name, velocity)
            for group_index, _ in enumerate(MOTION_GROUPS):
                self._move_axis_verified(group_index, float(self._best_position[group_index]))
            self.xps_backend.reset_velocities(self.xps)
        except Exception as exc:
            self._record_recovery_error('Move to best position failed', exc)


class BrentSpatialPeakWorker(QObject, _MotionMixin):
    """Run a Brent coordinate-descent spatial peak search on a Qt worker thread.

    Phase 1 — per-axis coarse scan: probes n_coarse equally-spaced points along
    each active axis (other axes held fixed), updating the running best after each
    axis. Total probes = d * n_coarse, far fewer than a full raster grid.

    Phase 2 — coordinate descent with Brent's bounded 1-D optimiser: for each
    active axis run scipy.optimize.minimize_scalar(method='bounded') which
    combines golden-section search (robust) with parabolic interpolation (fast
    near a Gaussian peak). Sweeps repeat until the maximum positional
    displacement across all axes falls below refinement_tol or max_sweeps is
    reached.

    The acoustic beam profile is unimodal and approximately Gaussian along every
    axis, so Brent's guarantee of convergence on a unimodal function applies, and
    its parabolic interpolation step achieves quadratic convergence in the high-SNR
    region near focus. Finite-difference gradient descent needs 2d extra probes
    per step with no convergence advantage and is unreliable in the low-SNR region
    far from focus.
    """

    frame_ready = pyqtSignal(object)
    point_ready = pyqtSignal(object, float, int, int)
    progress_changed = pyqtSignal(int)
    phase_changed = pyqtSignal(str)
    peak_found = pyqtSignal(object, float)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(object)
    finished = pyqtSignal()

    _BRENT_BUDGET_PER_AXIS_SWEEP = 15

    def __init__(
        self,
        controller: GageAlineController,
        acquisition_config: AcquisitionConfig,
        display_config: DisplayConfig,
        xps_backend,
        xps,
        home_position: np.ndarray,
        groups_to_move: np.ndarray,
        axes_limits: np.ndarray,
        scan_range: np.ndarray,
        n_coarse: int = 5,
        max_sweeps: int = 3,
        refinement_tol: float = 0.01,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.acquisition_config = acquisition_config
        self.display_config = display_config
        self.xps_backend = xps_backend
        self.xps = xps
        self.home_position = np.asarray(home_position, dtype=float)
        self.groups_to_move = np.asarray(groups_to_move, dtype=bool)
        self.axes_limits = np.asarray(axes_limits, dtype=float)
        self.scan_range = np.asarray(scan_range, dtype=float)
        self.n_coarse = max(2, int(n_coarse))
        self.max_sweeps = max(1, int(max_sweeps))
        self.refinement_tol = float(refinement_tol)
        self._stop_event = threading.Event()
        self._recovery_errors: list[str] = []
        self._best_position: np.ndarray = home_position.copy()
        self._best_value: float = 0.0
        self._probe_index: int = 0
        self._current_position: np.ndarray = home_position.copy()
        self._brent_probe_start: int = 0
        self._brent_budget: int = 1
        self._brent_estimated_total: int = 1

    @pyqtSlot()
    def run(self) -> None:
        active_indices = np.flatnonzero(self.groups_to_move)
        n_active = int(len(active_indices))
        estimated_total = (
            n_active * self.n_coarse
            + self.max_sweeps * n_active * self._BRENT_BUDGET_PER_AXIS_SWEEP
        )
        self._brent_estimated_total = estimated_total
        try:
            self.phase_changed.emit('coarse')
            self.status_changed.emit(
                f'Coarse scan: {n_active} axis/axes \u00d7 {self.n_coarse} points...'
            )
            self._run_coarse_phase(active_indices, estimated_total)
            if self._stop_event.is_set():
                self.status_changed.emit('Peak search stopped during coarse phase.')
                return
            self.phase_changed.emit('refine')
            self.status_changed.emit('Brent coordinate-descent refinement...')
            self._run_brent_phase(active_indices, estimated_total)
        except _SpatialPeakSearchStopped:
            self.status_changed.emit('Peak search stopped.')
        except Exception as exc:
            self.error.emit(_build_worker_error(exc, self._recovery_errors))
        finally:
            self._move_to_best_effort()
            self.peak_found.emit(self._best_position.copy(), self._best_value)
            self.finished.emit()

    @pyqtSlot()
    def stop(self) -> None:
        self._stop_event.set()

    def _axis_bounds(self, axis_idx: int) -> tuple[float, float]:
        lo = max(
            float(self.axes_limits[axis_idx, 0]),
            float(self.home_position[axis_idx]) - float(self.scan_range[axis_idx]) / 2.0,
        )
        hi = min(
            float(self.axes_limits[axis_idx, 1]),
            float(self.home_position[axis_idx]) + float(self.scan_range[axis_idx]) / 2.0,
        )
        return lo, hi

    def _run_coarse_phase(self, active_indices: np.ndarray, estimated_total: int) -> None:
        if int(len(active_indices)) == 0:
            raise ConfigurationError(
                'Brent peak search has no active axes — at least one group must be selected.'
            )
        timeout_s = self.acquisition_config.effective_timeout_s()
        self.status_changed.emit('Moving to home position...')
        velocity = _scan_velocity_mm_s()
        for group_name in MOTION_GROUPS:
            self.xps_backend.set_velocity(self.xps, group_name, velocity)
        for group_index, _ in enumerate(MOTION_GROUPS):
            self._move_axis_verified(group_index, float(self.home_position[group_index]))
        self.xps_backend.reset_velocities(self.xps)
        self._current_position = self.home_position.copy()

        n_coarse_total = int(len(active_indices)) * self.n_coarse

        for axis_idx in active_indices:
            lo, hi = self._axis_bounds(int(axis_idx))
            for x in np.linspace(lo, hi, self.n_coarse):
                if self._stop_event.is_set():
                    return
                new_pos = self._current_position.copy()
                new_pos[axis_idx] = x
                self._move_changed_groups(self._current_position, new_pos)
                averaged_frame, _ = self._capture_position_average(timeout_s=timeout_s)
                metric = float(abs(np.min(
                    self.display_config.apply_plot_offset(averaged_frame.volts_data)
                )))
                if metric > self._best_value:
                    self._best_value = metric
                    self._best_position = self._current_position.copy()
                self._probe_index += 1
                LOGGER.debug(
                    'brent_coarse: probe=%d axis=%d pos=%s metric=%.6g best=%.6g',
                    self._probe_index, int(axis_idx), self._current_position.tolist(),
                    metric, self._best_value,
                )
                self.frame_ready.emit(averaged_frame)
                self.point_ready.emit(
                    self._current_position.copy(), metric, self._probe_index, estimated_total,
                )
                self.progress_changed.emit(
                    int(round(self._probe_index / n_coarse_total * 50.0))
                )
            # Snap to best position found on this axis before moving to next axis
            best_on_axis = self._current_position.copy()
            best_on_axis[axis_idx] = self._best_position[axis_idx]
            if abs(best_on_axis[axis_idx] - self._current_position[axis_idx]) > 1e-6:
                self._move_changed_groups(self._current_position, best_on_axis)

    def _brent_objective(self, axis: int, lo: float, hi: float, sweep: int, x: float) -> float:
        """Objective evaluated by ``scipy.minimize_scalar`` during the Brent phase."""

        if self._stop_event.is_set():
            raise _SpatialPeakSearchStopped
        new_pos = self._current_position.copy()
        new_pos[axis] = float(np.clip(x, lo, hi))
        self._move_changed_groups(self._current_position, new_pos)
        averaged_frame, _ = self._capture_position_average(
            timeout_s=self.acquisition_config.effective_timeout_s(),
        )
        metric = float(abs(np.min(
            self.display_config.apply_plot_offset(averaged_frame.volts_data)
        )))
        if metric > self._best_value:
            self._best_value = metric
            self._best_position = self._current_position.copy()
        self._probe_index += 1
        elapsed = self._probe_index - self._brent_probe_start
        LOGGER.debug(
            'brent_refine: sweep=%d axis=%d probe=%d pos=%s metric=%.6g best=%.6g',
            sweep, axis, self._probe_index, self._current_position.tolist(),
            metric, self._best_value,
        )
        self.frame_ready.emit(averaged_frame)
        self.point_ready.emit(
            self._current_position.copy(), metric, self._probe_index, self._brent_estimated_total,
        )
        self.progress_changed.emit(
            min(99, 50 + int(round(elapsed / max(self._brent_budget, 1) * 50.0)))
        )
        return -metric

    def _run_brent_phase(self, active_indices: np.ndarray, estimated_total: int) -> None:
        from scipy.optimize import minimize_scalar

        self._brent_probe_start = self._probe_index
        self._brent_budget = max(estimated_total - self._brent_probe_start, 1)

        for sweep in range(self.max_sweeps):
            prev_pos = self._current_position.copy()

            for axis_idx in active_indices:
                lo, hi = self._axis_bounds(int(axis_idx))
                try:
                    minimize_scalar(
                        partial(self._brent_objective, int(axis_idx), lo, hi, sweep),
                        bounds=(lo, hi),
                        method='bounded',
                        options={'xatol': self.refinement_tol, 'maxiter': 50},
                    )
                except (
                    _SpatialPeakSearchStopped,
                    MotionTimeoutError,
                    StageInitializationError,
                    ConfigurationError,
                ):
                    # Hardware/configuration failures must propagate. See SpatialPeakWorker note.
                    raise
                except Exception as exc:
                    self.status_changed.emit(f'Brent warning on axis {axis_idx}: {exc}')

                # Snap to best on this axis before moving to next
                best_snap = self._current_position.copy()
                best_snap[axis_idx] = self._best_position[axis_idx]
                if abs(best_snap[axis_idx] - self._current_position[axis_idx]) > 1e-6:
                    self._move_changed_groups(self._current_position, best_snap)

            max_disp = float(np.max(np.abs(self._current_position - prev_pos)))
            self.status_changed.emit(
                f'Sweep {sweep + 1}/{self.max_sweeps}: max displacement = {max_disp:.4f} mm'
            )
            if max_disp < self.refinement_tol:
                self.status_changed.emit(f'Converged after {sweep + 1} sweep(s).')
                break

        self.status_changed.emit(
            f'Brent refinement complete. Best: '
            f'[{self._best_position[0]:.4f}, {self._best_position[1]:.4f}, '
            f'{self._best_position[2]:.4f}] mm  value={self._best_value:.6g}'
        )
        self.progress_changed.emit(100)

    def _move_changed_groups(self, previous_position: np.ndarray, current_position: np.ndarray) -> None:
        epsilon = _motion_epsilon_mm()
        changed = np.flatnonzero(np.abs(current_position - previous_position) > epsilon)
        for group_index in changed:
            self._move_axis_verified(int(group_index), float(current_position[group_index]))
        if changed.size:
            self._current_position = np.asarray(current_position, dtype=float).copy()

    def _capture_position_average(self, timeout_s: float | None = None) -> tuple[AcquisitionFrame, np.ndarray]:
        effective_timeout = timeout_s if timeout_s is not None else self.acquisition_config.effective_timeout_s()
        frames: list[AcquisitionFrame] = []
        for _ in range(max(1, self.display_config.lines_to_average)):
            if self._stop_event.is_set():
                raise _SpatialPeakSearchStopped
            frames.append(
                self.controller.acquire_single(
                    timeout_s=effective_timeout,
                    stop_event=self._stop_event,
                    display_config=self.display_config,
                )
            )
        raw_matrix = np.vstack([frame.volts_data for frame in frames])
        averaged_volts = np.mean(raw_matrix, axis=0)
        template = frames[-1]
        averaged_fft_input = self.display_config.apply_plot_offset(averaged_volts)
        averaged_frame = AcquisitionFrame(
            time_axis_us=np.asarray(template.time_axis_us, dtype=float),
            volts_data=np.asarray(averaged_volts, dtype=float),
            freq_axis_mhz=np.asarray(template.freq_axis_mhz, dtype=float),
            fft_mag=compute_fft_mag(averaged_fft_input),
            min_voltage=float(np.min(averaged_volts)),
        )
        displayed_lines = np.vstack(
            [self.display_config.apply_plot_offset(frame.volts_data) for frame in frames]
        ).T
        return averaged_frame, displayed_lines

    def _move_to_best_effort(self) -> None:
        try:
            velocity = _scan_velocity_mm_s()
            for group_name in MOTION_GROUPS:
                self.xps_backend.set_velocity(self.xps, group_name, velocity)
            for group_index, _ in enumerate(MOTION_GROUPS):
                self._move_axis_verified(group_index, float(self._best_position[group_index]))
            self.xps_backend.reset_velocities(self.xps)
        except Exception as exc:
            self._record_recovery_error('Brent move-to-best failed', exc)
