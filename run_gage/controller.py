from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from read_gage.python.calibration_loader import load_calibration_data
from run_gage.models import (
    AcquisitionConfig,
    AcquisitionFrame,
    AcquisitionStoppedError,
    ConfigurationError,
    DisplayConfig,
    TriggerTimeoutError,
    TransferError,
    compute_fft_mag,
)


LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
GAGE_PATH = SCRIPT_DIR / 'from_gage'

if str(GAGE_PATH) not in sys.path:
    sys.path.append(str(GAGE_PATH))

try:
    import GageConstants as gc
except ImportError as exc:
    raise RuntimeError(f'Could not import GageConstants from {GAGE_PATH}') from exc

import platform

os_name = platform.system()
if os_name == 'Windows':
    is_64_bits = sys.maxsize > 2**32
    if is_64_bits:
        if sys.version_info >= (3, 0):
            import PyGage3_64 as PyGage
        else:
            import PyGage2_64 as PyGage
    else:
        if sys.version_info > (3, 0):
            import PyGage3_32 as PyGage
        else:
            import PyGage2_32 as PyGage
else:
    import PyGage


#: Single-channel acquisition — the CSE1222 board used here is operated in 1-channel mode.
CHANNEL_NUM = 1
#: Gage CompuScope requires Depth, TriggerDelay, and TriggerHoldoff to be multiples of this value.
#: See Gage CompuScope SDK manual §"Buffer Alignment"; the 128-sample granularity is a hardware
#: constraint inherited by all PyGage calls that accept a sample count.
SAMPLE_ALIGNMENT = 128
#: Offset subtracted from the midpoint ADC code so that mid-scale maps to 0 V. Matches the Gage
#: PyGage example ``gage_aline_capture.py``; the value is board-family specific.
CS_SAMPLE_OFFSET_DEFAULT = -16
#: External trigger full-scale in volts for the CS_TRIG_SOURCE_EXT path used here. The hardware
#: expresses the trigger level as an integer percent of full scale, so the conversion is
#: ``level_percent = volts / TRIGGER_LEVEL_FULL_SCALE_V * TRIGGER_LEVEL_PERCENT_MAX``.
TRIGGER_LEVEL_FULL_SCALE_V = 5.0
#: Maximum integer value the hardware accepts for Trigger['Level']: represents 100% of full scale.
TRIGGER_LEVEL_PERCENT_MAX = 100


def get_channel_input_range_mv(volts: float) -> int:
    """Convert the displayed half-range in volts to full-scale millivolts."""

    return int(round(float(volts) * 2.0 * 1000.0))


def linear_interp_extrap(x_values: np.ndarray, y_values: np.ndarray, x_query: float) -> float:
    """Linearly interpolate or extrapolate a calibration value at the requested x position."""

    if x_values.size < 2:
        return float(y_values[0]) if y_values.size else 0.0
    if x_query <= x_values[0]:
        x0, x1 = x_values[0], x_values[1]
        y0, y1 = y_values[0], y_values[1]
    elif x_query >= x_values[-1]:
        x0, x1 = x_values[-2], x_values[-1]
        y0, y1 = y_values[-2], y_values[-1]
    else:
        return float(np.interp(x_query, x_values, y_values))
    slope = (y1 - y0) / (x1 - x0)
    return float(y0 + slope * (x_query - x0))


class GageAlineController:
    """Wrap low-level Gage hardware setup, capture, and calibration loading."""

    def __init__(self) -> None:
        self.handle: int | None = None
        self.system_info: dict | None = None
        self.acq_info: dict | None = None
        self.chan_info: dict | None = None
        self.lock = threading.Lock()

    def initialize(self) -> tuple[str, str | None]:
        """Initialize the digitizer and return the detected board identity."""

        with self.lock:
            LOGGER.info('Initializing Gage system')
            status = PyGage.Initialize()
            if status < 0:
                raise ConfigurationError(PyGage.GetErrorString(status))

            handle = PyGage.GetSystem(0, 0, 0, 0)
            if handle < 0:
                raise ConfigurationError(PyGage.GetErrorString(handle))

            self.handle = handle
            self.system_info = PyGage.GetSystemInfo(handle)
            if not isinstance(self.system_info, dict):
                raise ConfigurationError(PyGage.GetErrorString(self.system_info))

            serial_number = None
            get_serial_number = getattr(PyGage, 'GetSerialNumber', None)
            if callable(get_serial_number):
                try:
                    serial_number = str(get_serial_number(handle))
                except Exception:
                    LOGGER.debug('Serial number query failed', exc_info=True)

            LOGGER.info('Connected to board=%s serial=%s', self.system_info['BoardName'], serial_number)
            return self.system_info['BoardName'], serial_number if serial_number else None

    def close(self) -> None:
        """Release the active digitizer handle if one is open."""

        with self.lock:
            if self.handle is not None:
                LOGGER.info('Closing Gage handle')
                PyGage.FreeSystem(self.handle)
                self.handle = None

    def __enter__(self) -> 'GageAlineController':
        """Initialize the digitizer on context entry and return self."""

        if self.handle is None:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release the digitizer handle regardless of how the context exits."""

        self.close()

    def configure(self, config: AcquisitionConfig) -> tuple[dict, dict]:
        """Apply acquisition, channel, and trigger settings derived from the GUI state."""

        config.validate()

        with self.lock:
            if self.handle is None:
                raise ConfigurationError('Gage system is not initialized.')

            LOGGER.info('Configuring acquisition: %s', config)

            if config.t_start_us >= 0:
                depth = int(np.ceil((config.t_end_us - config.t_start_us) * 1e-6 * config.sample_rate_hz / SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT)
                trigger_delay = int(np.floor(config.t_start_us * 1e-6 * config.sample_rate_hz / SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT)
                trigger_holdoff = 0
            else:
                depth = int(np.ceil(config.t_end_us * 1e-6 * config.sample_rate_hz / SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT)
                trigger_delay = 0
                trigger_holdoff = int(np.floor((-config.t_start_us) * 1e-6 * config.sample_rate_hz / SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT)

            segment_size = depth + trigger_holdoff

            acq_config = PyGage.GetAcquisitionConfig(self.handle)
            if not isinstance(acq_config, dict):
                raise ConfigurationError(PyGage.GetErrorString(acq_config))

            if config.t_start_us < 0 and 'TriggerHoldoff' not in acq_config:
                raise ConfigurationError('Negative time start requires TriggerHoldoff support on this digitizer.')

            acq_config['Mode'] = gc.CS_MODE_SINGLE
            acq_config['SampleRate'] = int(config.sample_rate_hz)
            acq_config['Depth'] = depth
            acq_config['SegmentSize'] = segment_size
            acq_config['SegmentCount'] = 1
            acq_config['TriggerTimeout'] = -1
            if 'TriggerHoldoff' in acq_config:
                acq_config['TriggerHoldoff'] = trigger_holdoff
            acq_config['TriggerDelay'] = trigger_delay
            if 'ExtClk' in acq_config:
                acq_config['ExtClk'] = 0
            if 'TimeStampConfig' in acq_config:
                acq_config['TimeStampConfig'] = 0
            if 'SampleOffset' in acq_config:
                acq_config['SampleOffset'] = CS_SAMPLE_OFFSET_DEFAULT

            ret = PyGage.SetAcquisitionConfig(self.handle, acq_config)
            if ret < 0:
                raise ConfigurationError(PyGage.GetErrorString(ret))

            chan_config = PyGage.GetChannelConfig(self.handle, CHANNEL_NUM)
            if not isinstance(chan_config, dict):
                raise ConfigurationError(PyGage.GetErrorString(chan_config))

            chan_config['Coupling'] = gc.CS_COUPLING_DC
            chan_config['InputRange'] = get_channel_input_range_mv(config.input_range_v)
            chan_config['Impedance'] = 50
            chan_config['DcOffset'] = int(config.dc_offset_mv)
            chan_config['Filter'] = 0

            ret = PyGage.SetChannelConfig(self.handle, CHANNEL_NUM, chan_config)
            if ret < 0:
                raise ConfigurationError(PyGage.GetErrorString(ret))

            trig_config = PyGage.GetTriggerConfig(self.handle, 1)
            if not isinstance(trig_config, dict):
                raise ConfigurationError(PyGage.GetErrorString(trig_config))

            trig_config['Condition'] = gc.CS_TRIG_COND_POS_SLOPE
            trigger_level_percent = int(np.floor(
                config.trigger_level_v / TRIGGER_LEVEL_FULL_SCALE_V * TRIGGER_LEVEL_PERCENT_MAX
            ))
            trig_config['Level'] = trigger_level_percent
            trig_config['Source'] = gc.CS_TRIG_SOURCE_EXT
            trig_config['ExtCoupling'] = gc.CS_COUPLING_DC
            if 'ExtRange' in trig_config:
                trig_config['ExtRange'] = 10000
            if 'ExtImpedance' in trig_config:
                trig_config['ExtImpedance'] = 1000000

            ret = PyGage.SetTriggerConfig(self.handle, 1, trig_config)
            if ret < 0:
                raise ConfigurationError(PyGage.GetErrorString(ret))

            ret = PyGage.Commit(self.handle)
            if ret < 0:
                raise ConfigurationError(PyGage.GetErrorString(ret))

            self.acq_info = PyGage.GetAcquisitionConfig(self.handle)
            self.chan_info = PyGage.GetChannelConfig(self.handle, CHANNEL_NUM)
            committed_level = int(PyGage.GetTriggerConfig(self.handle, 1).get('Level', trigger_level_percent))
            if committed_level != trigger_level_percent:
                LOGGER.warning(
                    'Trigger level mismatch: requested %d%% (%.3f V), committed %d%%',
                    trigger_level_percent, config.trigger_level_v, committed_level,
                )
            LOGGER.info(
                'Configuration committed: depth=%d segment=%d trigger_level=%d%% (%.3f V) acq=%s chan=%s',
                depth, segment_size, trigger_level_percent, config.trigger_level_v,
                self.acq_info, self.chan_info,
            )
            return self.acq_info, self.chan_info

    def acquire_single(
        self,
        timeout_s: float,
        stop_event: threading.Event | None = None,
        display_config: DisplayConfig | None = None,
    ) -> AcquisitionFrame:
        """Capture one A-line, convert it to volts, and compute FFT display data.
        
        Args:
            timeout_s: Maximum time to wait for trigger in seconds.
            stop_event: Optional threading event to abort acquisition.
            display_config: Optional display settings to apply plot offset before FFT.
                If None, FFT is computed on raw voltage data.
        """

        with self.lock:
            if self.handle is None or self.acq_info is None or self.chan_info is None:
                raise ConfigurationError('Acquisition has not been configured.')

            capture_start_s = time.time()
            ret = PyGage.StartCapture(self.handle)
            if ret < 0:
                raise TransferError(PyGage.GetErrorString(ret))

            wait_start = capture_start_s
            status = PyGage.GetStatus(self.handle)
            while status != gc.ACQ_STATUS_READY:
                if stop_event is not None and stop_event.is_set():
                    self._abort_capture_locked()
                    raise AcquisitionStoppedError('Acquisition stopped.')
                if time.time() - wait_start > timeout_s:
                    self._abort_capture_locked()
                    raise TriggerTimeoutError('Timeout waiting for trigger')
                time.sleep(0.01)
                status = PyGage.GetStatus(self.handle)

            transfer_start = int(self.acq_info['TriggerDelay'] + self.acq_info['Depth'] - self.acq_info['SegmentSize'])
            raw_data = PyGage.TransferData(
                self.handle,
                CHANNEL_NUM,
                gc.TxMODE_DEFAULT,
                1,
                transfer_start,
                self.acq_info['SegmentSize'],
            )
            if isinstance(raw_data, int):
                raise TransferError(PyGage.GetErrorString(raw_data))

            raw_counts = np.asarray(raw_data[0])
            actual_start = int(raw_data[1])
            actual_length = int(raw_data[2])
            if actual_length < raw_counts.size:
                raw_counts = raw_counts[:actual_length]

            volts_data = self.convert_to_volts(raw_counts)
            sample_points = np.arange(actual_start, actual_start + actual_length)
            time_axis_us = sample_points / self.acq_info['SampleRate'] * 1e6

            # Apply plot offset to waveform before FFT if display config is provided
            fft_input_data = volts_data
            if display_config is not None:
                fft_input_data = display_config.apply_plot_offset(volts_data)

            fft_length = int(2 ** np.ceil(np.log2(actual_length)))
            freq_axis_mhz = np.fft.rfftfreq(fft_length, d=1.0 / self.acq_info['SampleRate']) * 1e-6
            fft_mag = compute_fft_mag(fft_input_data)

            elapsed_ms = (time.time() - capture_start_s) * 1000.0
            LOGGER.debug(
                'acquire_single: depth=%d trigger_delay=%d elapsed_ms=%.1f min_v=%.6g',
                self.acq_info['Depth'], self.acq_info['TriggerDelay'], elapsed_ms,
                float(np.min(volts_data)),
            )

            return AcquisitionFrame(
                time_axis_us=time_axis_us,
                volts_data=volts_data,
                freq_axis_mhz=freq_axis_mhz,
                fft_mag=fft_mag,
                min_voltage=float(np.min(volts_data)),
            )

    def _abort_capture_locked(self) -> None:
        """Abort an in-flight capture while holding the controller lock.

        Any failure from PyGage.AbortCapture is logged at WARNING so that the
        underlying cause remains visible when a capture is being aborted due to
        a timeout or user stop. The method does not re-raise: the caller has
        already decided to abort and expects this to be best-effort.
        """

        abort_capture = getattr(PyGage, 'AbortCapture', None)
        if not callable(abort_capture) or self.handle is None:
            return
        try:
            abort_capture(self.handle)
            LOGGER.info('Capture aborted')
        except Exception:
            LOGGER.warning('AbortCapture failed during recovery', exc_info=True)

    def convert_to_volts(self, raw_counts: np.ndarray) -> np.ndarray:
        """Convert raw ADC counts to voltage using the committed channel settings."""

        scale_factor = self.chan_info['InputRange'] / 2000.0
        offset_volts = self.chan_info['DcOffset'] / 1000.0
        return (((self.acq_info['SampleOffset'] - raw_counts) / self.acq_info['SampleResolution']) * scale_factor) + offset_volts

    def load_calibration(self, config: AcquisitionConfig) -> np.ndarray:
        """Load hydrophone calibration data matching the current hardware selection."""

        return load_calibration_data(config.calibration_file)
