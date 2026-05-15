from __future__ import annotations

import logging
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from run_gage.acquisition_workers import LiveAcquisitionWorker, build_display_fft_mag, build_metadata_payload, write_calibration_sidecar, write_json_sidecar
from run_gage.gui.app_motion_control import read_motion_position_mm, show_motion_control_window
from run_gage.controller import GageAlineController, linear_interp_extrap
from run_gage.models import AcquisitionConfig, AcquisitionFrame, DEFAULT_SAMPLE_RATE_HZ, DisplayConfig, SUPPORTED_INPUT_RANGES_V
from read_gage.python.calibration_loader import get_names, get_entry_by_name
from run_gage.gui.constants import RECONFIGURE_DEBOUNCE_MS, VOLTAGE_RANGE_OPTIONS
from run_gage.gui.style_utils import apply_gui_scaling
from run_gage.gui.plot_widgets import AlinePlotCanvas, FftPlotCanvas, MmodePlotCanvas


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


class StatusEmitter(QObject):
    """Bridge background log records into the Qt main thread."""

    message = pyqtSignal(str)


class GuiLogHandler(logging.Handler):
    """Logging handler that forwards formatted records to a Qt signal."""

    def __init__(self, emitter: StatusEmitter) -> None:
        super().__init__(level=logging.INFO)
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a formatted log record to the GUI signal."""

        try:
            self.emitter.message.emit(self.format(record))
        except Exception:
            self.handleError(record)


class MmodeGui(QMainWindow):
    """Interactive PyQt front end for live M-mode acquisition and buffered saves."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('M-mode Acquisition')
        self.resize(1800, 1000)

        self.controller = GageAlineController()
        self.calibration_vals = np.empty((0, 2))
        self.last_frame: AcquisitionFrame | None = None
        self.last_fft_freq_axis = np.array([], dtype=float)
        self.last_fft_values = np.array([], dtype=float)
        self.fft_frame_history: deque[AcquisitionFrame] = deque()
        self.current_mode: str | None = None
        self.hardware_ready = False
        self.pending_live_restart = False
        self.pending_buffer_save = False
        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None
        self._last_save_folder = Path.home() / 'Documents'
        self.xps = None

        self.mmode_buffer = np.empty((0, 0), dtype=float)
        self.mmode_time_axis_us = np.array([], dtype=float)
        self.mmode_write_index = 0
        self.mmode_valid_count = 0
        self.mmode_last_write_index = -1

        self.log_emitter = StatusEmitter()
        self.log_handler = GuiLogHandler(self.log_emitter)

        self._build_ui()
        self.hardware_input_range_v = float(self.voltage_range_combo.currentText())
        apply_gui_scaling(self)
        self._connect_signals()
        self._configure_logging()
        self._update_tof()

        self.reconfigure_timer = QTimer(self)
        self.reconfigure_timer.setSingleShot(True)
        self.reconfigure_timer.timeout.connect(self._apply_parameter_change)

        QTimer.singleShot(10, self._initialize_hardware)

    def _build_ui(self) -> None:
        """Assemble the top-level GUI layout and embed the plot canvases."""

        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QGridLayout(central)
        root_layout.setColumnStretch(0, 0)
        root_layout.setColumnStretch(1, 1)
        root_layout.setRowStretch(0, 1)

        controls_layout = QVBoxLayout()
        display_layout = QVBoxLayout()
        root_layout.addLayout(controls_layout, 0, 0)
        root_layout.addLayout(display_layout, 0, 1)

        self.board_label = QLineEdit('Connecting...')
        self.board_label.setReadOnly(True)
        self.board_label.setStyleSheet(
            """
            background-color: #e6eaf2;
            border: 1px solid #e6eaf2;
            """
        )
        controls_layout.addWidget(self._build_system_group())
        controls_layout.addWidget(self._build_run_group())
        controls_layout.addLayout(self._build_controls_row(self._build_acquisition_group(), self._build_fft_group()))
        controls_layout.addWidget(self._build_save_group())
        controls_layout.addLayout(self._build_controls_row(self._build_hydrophone_group(), self._build_tof_group()))

        motion_button = QPushButton('Motion Controller')
        motion_button.clicked.connect(self._show_motion_controller)
        controls_layout.addWidget(motion_button)
        controls_layout.addStretch(1)

        self.fft_canvas = FftPlotCanvas()
        self.aline_canvas = AlinePlotCanvas()
        self.mmode_canvas = MmodePlotCanvas()
        display_layout.addWidget(self._wrap_group('Frequency Spectrum', self.fft_canvas), 2)
        display_layout.addWidget(self._wrap_group('A-line', self.aline_canvas), 3)
        display_layout.addWidget(self._wrap_group('M-mode Buffer', self.mmode_canvas), 4)

        status_group = QGroupBox('Status')
        status_layout = QVBoxLayout(status_group)
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        display_layout.addWidget(status_group, 1)

    def _build_controls_row(self, left_group: QGroupBox, right_group: QGroupBox) -> QHBoxLayout:
        """Place two control panels on one row to reduce vertical space usage."""

        layout = QHBoxLayout()
        layout.addWidget(left_group, 1)
        layout.addWidget(right_group, 1)
        return layout

    def _build_system_group(self) -> QGroupBox:
        group = QGroupBox('System')
        layout = QFormLayout(group)
        layout.addRow('Board', self.board_label)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox('Control')
        layout = QHBoxLayout(group)
        self.run_button = QPushButton('Run')
        self.stop_button = QPushButton('Stop')
        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        return group

    def _build_acquisition_group(self) -> QGroupBox:
        group = QGroupBox('Acquisition')
        layout = QFormLayout(group)

        self.time_start_spin = QDoubleSpinBox()
        self.time_start_spin.setRange(-20e6, 20e6)
        self.time_start_spin.setDecimals(2)
        self.time_start_spin.setValue(0.0)

        self.time_end_spin = QDoubleSpinBox()
        self.time_end_spin.setRange(-20e6, 20e6)
        self.time_end_spin.setDecimals(2)
        self.time_end_spin.setValue(50.0)

        self.voltage_range_combo = QComboBox()
        self.voltage_range_combo.addItems(VOLTAGE_RANGE_OPTIONS)
        self.voltage_range_combo.setCurrentText('0.1')

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-5000.0, 5000.0)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setSingleStep(1.0)

        self.trigger_spin = QDoubleSpinBox()
        self.trigger_spin.setRange(0.1, 5.0)
        self.trigger_spin.setDecimals(2)
        self.trigger_spin.setSingleStep(0.1)
        self.trigger_spin.setValue(2.5)

        layout.addRow('Time start (us)', self.time_start_spin)
        layout.addRow('Time end (us)', self.time_end_spin)
        layout.addRow('Input voltage range (V)', self.voltage_range_combo)
        layout.addRow('Plot offset (mV)', self.offset_spin)
        layout.addRow('Trigger level (V)', self.trigger_spin)
        return group

    def _build_fft_group(self) -> QGroupBox:
        group = QGroupBox('Frequency Spectrum')
        layout = QFormLayout(group)

        self.lines_average_spin = QSpinBox()
        self.lines_average_spin.setRange(1, 99999)
        self.lines_average_spin.setValue(3)

        self.freq_start_spin = QDoubleSpinBox()
        self.freq_start_spin.setRange(0.0, DEFAULT_SAMPLE_RATE_HZ / 2e6)
        self.freq_start_spin.setDecimals(3)

        self.freq_end_spin = QDoubleSpinBox()
        self.freq_end_spin.setRange(0.0, DEFAULT_SAMPLE_RATE_HZ / 2e6)
        self.freq_end_spin.setDecimals(3)
        self.freq_end_spin.setValue(10.0)

        self.max_frequency_edit = QLineEdit('0.0')
        self.max_frequency_edit.setReadOnly(True)
        self.max_frequency_edit.setStyleSheet(
            """
            background-color: #e6eaf2;
            border: 1px solid #e6eaf2;
            """
        )

        layout.addRow('Lines to average', self.lines_average_spin)
        layout.addRow('Freq start (MHz)', self.freq_start_spin)
        layout.addRow('Freq end (MHz)', self.freq_end_spin)
        layout.addRow('Max frequency (MHz)', self.max_frequency_edit)
        return group

    def _build_save_group(self) -> QGroupBox:
        group = QGroupBox('Save')
        layout = QFormLayout(group)

        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(1, 99999)
        self.buffer_size_spin.setValue(100)

        self.save_button = QPushButton('Save')

        layout.addRow('Buffer size', self.buffer_size_spin)
        layout.addRow(self.save_button)
        return group

    def _build_hydrophone_group(self) -> QGroupBox:
        group = QGroupBox('Hydrophone')
        layout = QFormLayout(group)

        self.calibration_name_combo = QComboBox()
        self.calibration_name_combo.addItems(get_names())

        _ro_style = 'background-color: #e6eaf2; border: 1px solid #e6eaf2; min-width: 180px; padding-left: 4px;'

        self.calibration_file_edit = QLineEdit('\u2014')
        self.calibration_file_edit.setReadOnly(True)
        self.calibration_file_edit.setStyleSheet(_ro_style)

        self.hydro_preamp_edit = QLineEdit('\u2014')
        self.hydro_preamp_edit.setReadOnly(True)
        self.hydro_preamp_edit.setStyleSheet(_ro_style)

        self.connector_edit = QLineEdit('\u2014')
        self.connector_edit.setReadOnly(True)
        self.connector_edit.setStyleSheet(_ro_style)

        self.attenuator_info_edit = QLineEdit('\u2014')
        self.attenuator_info_edit.setReadOnly(True)
        self.attenuator_info_edit.setStyleSheet(_ro_style)

        self.gain_edit = QLineEdit('\u2014')
        self.gain_edit.setReadOnly(True)
        self.gain_edit.setStyleSheet(_ro_style)

        self.sensitivity_edit = QLineEdit('0.0')
        self.sensitivity_edit.setReadOnly(True)
        self.sensitivity_edit.setStyleSheet(_ro_style)

        self.pressure_edit = QLineEdit('0.0')
        self.pressure_edit.setReadOnly(True)
        self.pressure_edit.setStyleSheet(_ro_style)

        self.mi_edit = QLineEdit('0.0')
        self.mi_edit.setReadOnly(True)
        self.mi_edit.setStyleSheet(_ro_style)

        layout.addRow('Hydrophone', self.calibration_name_combo)
        layout.addRow('Calibration file', self.calibration_file_edit)
        layout.addRow('Hydrophone / Preamp', self.hydro_preamp_edit)
        layout.addRow('Connector', self.connector_edit)
        layout.addRow('Attenuator', self.attenuator_info_edit)
        layout.addRow('Gain', self.gain_edit)
        layout.addRow('Sensitivity', self.sensitivity_edit)
        layout.addRow('Peak neg. pressure (kPa)', self.pressure_edit)
        layout.addRow('MI', self.mi_edit)
        return group

    def _build_tof_group(self) -> QGroupBox:
        group = QGroupBox('Time of Flight')
        layout = QFormLayout(group)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 10000.0)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1480.0)

        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setRange(0.0, 1000.0)
        self.focal_spin.setDecimals(3)

        self.tof_edit = QLineEdit('0.0')
        self.tof_edit.setReadOnly(True)
        self.tof_edit.setStyleSheet(
            """
            background-color: #e6eaf2;
            border: 1px solid #e6eaf2;
            """
        )

        layout.addRow('Speed of sound (m/s)', self.speed_spin)
        layout.addRow('Focal length (mm)', self.focal_spin)
        layout.addRow('Time of flight (us)', self.tof_edit)
        return group

    def _wrap_group(self, title: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(widget)
        return group

    def _connect_signals(self) -> None:
        self.run_button.clicked.connect(self._start_live)
        self.stop_button.clicked.connect(self._stop_worker)
        self.save_button.clicked.connect(self._save_buffer)
        self.speed_spin.valueChanged.connect(self._update_tof)
        self.focal_spin.valueChanged.connect(self._update_tof)
        self.lines_average_spin.valueChanged.connect(self._refresh_existing_plots)
        self.freq_start_spin.valueChanged.connect(self._refresh_existing_plots)
        self.freq_end_spin.valueChanged.connect(self._refresh_existing_plots)
        self.buffer_size_spin.valueChanged.connect(self._handle_buffer_size_changed)
        self.voltage_range_combo.currentTextChanged.connect(self._on_voltage_range_changed)
        self.log_emitter.message.connect(self._append_status)

        self.time_start_spin.valueChanged.connect(self._schedule_parameter_change)
        self.time_end_spin.valueChanged.connect(self._schedule_parameter_change)
        self.offset_spin.valueChanged.connect(self._refresh_existing_plots)
        self.trigger_spin.valueChanged.connect(self._schedule_parameter_change)
        self.calibration_name_combo.currentTextChanged.connect(self._on_calibration_selection_changed)

    def _configure_logging(self) -> None:
        formatter = logging.Formatter('%(name)s: %(message)s')
        self.log_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.log_handler)

    def _initialize_hardware(self) -> None:
        try:
            board_name, serial_number = self.controller.initialize()
            label = board_name if not serial_number else f'{board_name} | SN: {serial_number}'
            self.board_label.setText(label)
            self.hardware_ready = True
            self._log_status(f'Connected to {board_name}')
            self._apply_parameter_change()
        except Exception as exc:
            self.board_label.setText('Unavailable')
            self._log_status(f'Initialization failed: {exc}')
            QMessageBox.critical(self, 'Initialization Failed', str(exc))

    def _collect_acquisition_config(self) -> AcquisitionConfig:
        entry = get_entry_by_name(self.calibration_name_combo.currentText())
        config = AcquisitionConfig(
            t_start_us=self.time_start_spin.value(),
            t_end_us=self.time_end_spin.value(),
            sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ,
            trigger_level_v=self.trigger_spin.value(),
            input_range_v=self.hardware_input_range_v,
            dc_offset_mv=0.0,
            n_alines=self.buffer_size_spin.value(),
            data_type='mmode',
            calibration_file=entry.file_stem if entry is not None else '',
        )
        config.validate()
        return config

    def _collect_display_config(self) -> DisplayConfig:
        return DisplayConfig(
            displayed_voltage_range_v=float(self.voltage_range_combo.currentText()),
            plot_offset_v=self.offset_spin.value() / 1000.0,
            freq_start_mhz=self.freq_start_spin.value(),
            freq_end_mhz=self.freq_end_spin.value(),
            lines_to_average=self.lines_average_spin.value(),
        )

    def _selected_voltage_range_v(self) -> float:
        return float(self.voltage_range_combo.currentText())

    def _on_voltage_range_changed(self) -> None:
        selected_range_v = self._selected_voltage_range_v()
        self._refresh_existing_plots()
        if selected_range_v in SUPPORTED_INPUT_RANGES_V:
            self.hardware_input_range_v = selected_range_v
            self._schedule_parameter_change()
            return
        self._log_status(
            f'Voltage range {selected_range_v:g} V is display-only; hardware range remains {self.hardware_input_range_v:g} V.'
        )

    def _prepare_acquisition(self) -> tuple[AcquisitionConfig, DisplayConfig]:
        acquisition_config = self._collect_acquisition_config()
        display_config = self._collect_display_config()
        acq_info, _ = self.controller.configure(acquisition_config)
        self.fft_frame_history.clear()
        self._reset_mmode_buffer()
        self._refresh_calibration_state(acquisition_config)
        holdoff = acq_info.get('TriggerHoldoff', 0)
        self._log_status(
            f"Configured acquisition: Fs={acq_info['SampleRate']}, Depth={acq_info['Depth']}, SegmentSize={acq_info['SegmentSize']}, Holdoff={holdoff}"
        )
        return acquisition_config, display_config

    def _start_live(self) -> None:
        if self._worker_is_active():
            self._log_status('Worker already active.')
            return
        try:
            acquisition_config, display_config = self._prepare_acquisition()
        except Exception as exc:
            self._handle_configuration_error(exc)
            return

        worker = LiveAcquisitionWorker(self.controller, acquisition_config, display_config)
        worker.frame_ready.connect(self._handle_live_frame)
        worker.status_changed.connect(self._log_status)
        worker.error.connect(self._handle_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self.current_mode = 'live'
        self._start_worker_thread(worker)
        self._log_status('Live acquisition started.')

    def _save_buffer(self) -> None:
        if self._worker_is_active():
            self.pending_live_restart = self.current_mode == 'live'
            self.pending_buffer_save = True
            self._log_status('Pausing live acquisition before saving buffer...')
            self._stop_worker()
            return

        self._save_buffer_after_stop()

    def _save_buffer_after_stop(self) -> None:
        snapshot = self._snapshot_buffer_for_save()
        if snapshot is None:
            QMessageBox.information(self, 'Save Buffer', 'No buffered A-lines are available to save yet.')
            return

        time_axis_us, ordered_alines, metadata = snapshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_path = str(self._last_save_folder / f'mmode_{timestamp}.npy')
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save M-mode data', default_path, 'Numpy files (*.npy);;All files (*)')
        if not save_path:
            self._log_status('Save cancelled.')
            return

        save_matrix = np.zeros((time_axis_us.size, ordered_alines.shape[1] + 1), dtype=float)
        save_matrix[:, 0] = time_axis_us
        save_matrix[:, 1:] = ordered_alines
        destination = Path(save_path)
        self._last_save_folder = destination.parent
        np.save(destination, save_matrix)
        acquisition_config = self._collect_acquisition_config()
        motion_position, self.xps = read_motion_position_mm(self.xps)
        extra: dict[str, object] = {
            'data_file': destination.name,
            'valid_alines': metadata['valid_count'],
            'next_write_index': metadata['next_write_index'],
            'last_write_index': metadata['last_write_index'],
        }
        if motion_position is not None:
            extra['motion_position_mm'] = motion_position
        write_json_sidecar(
            destination,
            build_metadata_payload(
                acquisition_config,
                self._collect_display_config(),
                extra=extra,
            ),
        )
        write_calibration_sidecar(
            destination.with_suffix('.cal'),
            acquisition_config.calibration_file,
        )
        self._log_status(f'Saved {ordered_alines.shape[1]} buffered A-lines to {save_path}')
        QMessageBox.information(self, 'Save Complete', f'Data saved to:\n{save_path}')

    def _start_worker_thread(self, worker: QObject) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        thread.finished.connect(thread.deleteLater)
        self.worker_thread = thread
        self.worker = worker
        thread.start()

    def _stop_worker(self) -> None:
        if self.worker is None:
            return
        stop_method = getattr(self.worker, 'stop', None)
        if callable(stop_method):
            stop_method()
            self._log_status('Stop requested.')

    def _handle_live_frame(self, frame: AcquisitionFrame, freq_axis_mhz: np.ndarray, fft_average: np.ndarray, min_voltage: float) -> None:
        frame.min_voltage = min_voltage
        self._remember_fft_frame(frame)
        self._update_mmode_buffer(frame)
        self._render_frame(frame, freq_axis_mhz)

    def _render_frame(self, frame: AcquisitionFrame, freq_axis_mhz: np.ndarray) -> None:
        display_config = self._collect_display_config()
        self.last_frame = frame
        self.last_fft_freq_axis = np.asarray(freq_axis_mhz)
        self.last_fft_values = self._build_rolling_fft_average(display_config)

        self.aline_canvas.update_frame(frame.time_axis_us, frame.volts_data, frame.min_voltage, display_config)
        self.fft_canvas.update_spectrum(self.last_fft_freq_axis, self.last_fft_values, display_config)
        if self.mmode_valid_count and self.mmode_time_axis_us.size:
            self.mmode_canvas.update_image(
                self.mmode_buffer,
                self.mmode_time_axis_us,
                self.buffer_size_spin.value(),
                self.mmode_write_index,
            )

        if self.last_fft_values.size:
            max_index = int(np.argmax(self.last_fft_values))
            max_frequency_mhz = float(self.last_fft_freq_axis[max_index])
            self.max_frequency_edit.setText(f'{max_frequency_mhz:.6f}')
            self._update_pressure_estimate(max_frequency_mhz, frame.min_voltage)

    def _update_mmode_buffer(self, frame: AcquisitionFrame) -> None:
        display_config = self._collect_display_config()
        displayed_volts = np.asarray(display_config.apply_plot_offset(frame.volts_data), dtype=float)
        buffer_size = self.buffer_size_spin.value()
        if (
            self.mmode_buffer.size == 0
            or self.mmode_buffer.shape[0] != displayed_volts.size
            or self.mmode_buffer.shape[1] != buffer_size
        ):
            self.mmode_buffer = np.zeros((displayed_volts.size, buffer_size), dtype=float)
            self.mmode_time_axis_us = np.asarray(frame.time_axis_us, dtype=float)
            self.mmode_write_index = 0
            self.mmode_valid_count = 0
            self.mmode_last_write_index = -1

        self.mmode_buffer[:, self.mmode_write_index] = displayed_volts
        self.mmode_last_write_index = self.mmode_write_index
        self.mmode_write_index = (self.mmode_write_index + 1) % buffer_size
        self.mmode_valid_count = min(self.mmode_valid_count + 1, buffer_size)

    def _snapshot_buffer_for_save(self) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
        if self.mmode_valid_count == 0 or self.mmode_time_axis_us.size == 0 or self.mmode_buffer.size == 0:
            return None

        if self.mmode_valid_count < self.buffer_size_spin.value():
            ordered_alines = self.mmode_buffer[:, :self.mmode_valid_count].copy()
        else:
            ordered_alines = np.hstack(
                (
                    self.mmode_buffer[:, self.mmode_write_index :],
                    self.mmode_buffer[:, : self.mmode_write_index],
                )
            ).copy()

        metadata = {
            'valid_count': self.mmode_valid_count,
            'next_write_index': self.mmode_write_index,
            'last_write_index': self.mmode_last_write_index,
        }
        return self.mmode_time_axis_us.copy(), ordered_alines, metadata

    def _reset_mmode_buffer(self) -> None:
        self.mmode_buffer = np.empty((0, 0), dtype=float)
        self.mmode_time_axis_us = np.array([], dtype=float)
        self.mmode_write_index = 0
        self.mmode_valid_count = 0
        self.mmode_last_write_index = -1

    def _handle_buffer_size_changed(self) -> None:
        self._reset_mmode_buffer()
        self._refresh_existing_plots()

    def _update_pressure_estimate(self, max_frequency_mhz: float, min_voltage: float) -> None:
        if self.calibration_vals.size == 0:
            self.sensitivity_edit.setText('0.0')
            self.pressure_edit.setText('0.0')
            return

        hydro_sensitivity = linear_interp_extrap(
            self.calibration_vals[:, 0],
            self.calibration_vals[:, 1],
            max_frequency_mhz,
        )
        pressure_estimate_kpa = abs(min_voltage / hydro_sensitivity) * 1e-3
        self.sensitivity_edit.setText(f'{hydro_sensitivity:.6g}')
        self.pressure_edit.setText(f'{pressure_estimate_kpa:.6g}')
        self.mi_edit.setText(f'{pressure_estimate_kpa * 1e-3 / np.sqrt(max_frequency_mhz):.6g}')

    def _update_tof(self) -> None:
        speed_of_sound = self.speed_spin.value()
        focal_length_mm = self.focal_spin.value()
        tof_s = (focal_length_mm * 1e-3) / speed_of_sound
        self.tof_edit.setText(f'{tof_s * 1e6:.6g}')

    def _schedule_parameter_change(self) -> None:
        if not self.hardware_ready:
            return
        self.reconfigure_timer.start(RECONFIGURE_DEBOUNCE_MS)

    def _on_calibration_selection_changed(self) -> None:
        name = self.calibration_name_combo.currentText()
        entry = get_entry_by_name(name)
        if entry is None:
            self.calibration_file_edit.setText('\u2014')
            self.hydro_preamp_edit.setText('\u2014')
            self.connector_edit.setText('\u2014')
            self.attenuator_info_edit.setText('\u2014')
            self.gain_edit.setText('\u2014')
        else:
            self.calibration_file_edit.setText(f'{entry.file_stem}.txt')
            self.hydro_preamp_edit.setText(entry.hydrophone_preamplifier)
            self.connector_edit.setText(entry.connector)
            self.attenuator_info_edit.setText(entry.attenuator)
            self.gain_edit.setText(entry.gain)
        if not self.hardware_ready:
            return
        acquisition_config = self._collect_acquisition_config()
        self._refresh_calibration_state(acquisition_config)

    def _refresh_calibration_state(self, acquisition_config: AcquisitionConfig) -> None:
        self.calibration_vals = self.controller.load_calibration(acquisition_config)
        if self.last_frame is None or not self.last_fft_values.size:
            self.sensitivity_edit.setText('0.0')
            self.pressure_edit.setText('0.0')
            self.mi_edit.setText('0.0')
            return

        max_index = int(np.argmax(self.last_fft_values))
        max_frequency_mhz = float(self.last_fft_freq_axis[max_index])
        self._update_pressure_estimate(max_frequency_mhz, self.last_frame.min_voltage)

    def _apply_parameter_change(self) -> None:
        if not self.hardware_ready:
            return

        if self._worker_is_active():
            self.pending_live_restart = self.current_mode == 'live'
            self._log_status('Parameter changed. Reconfiguring acquisition...')
            self._stop_worker()
            return

        try:
            self._prepare_acquisition()
        except Exception as exc:
            self._handle_configuration_error(exc)
            return

        if self.pending_live_restart:
            self.pending_live_restart = False
            self._start_live()

    def _refresh_existing_plots(self) -> None:
        if self.last_frame is not None and self.last_fft_values.size:
            self._render_frame(self.last_frame, self.last_fft_freq_axis)

    def _remember_fft_frame(self, frame: AcquisitionFrame) -> None:
        self.fft_frame_history.append(frame)
        max_history = max(1, self.lines_average_spin.value())
        while len(self.fft_frame_history) > max_history:
            self.fft_frame_history.popleft()

    def _build_rolling_fft_average(self, display_config: DisplayConfig) -> np.ndarray:
        if not self.fft_frame_history:
            if self.last_frame is None:
                return np.array([], dtype=float)
            return build_display_fft_mag(self.last_frame, display_config)

        frames = list(self.fft_frame_history)[-max(1, display_config.lines_to_average) :]
        fft_magnitudes = [build_display_fft_mag(frame, display_config) for frame in frames]
        return np.mean(np.vstack(fft_magnitudes), axis=0)

    def _handle_worker_error(self, payload) -> None:
        short_message = getattr(payload, 'message', None) or str(payload)
        self._log_status(f'Error: {payload}')
        QMessageBox.critical(self, 'Acquisition Error', short_message)

    def _handle_configuration_error(self, exc: Exception) -> None:
        self._log_status(f'Configuration error: {exc}')
        QMessageBox.critical(self, 'Configuration Error', str(exc))

    def _on_worker_finished(self) -> None:
        thread = self.worker_thread
        worker = self.worker
        self.worker_thread = None
        self.worker = None
        self.current_mode = None
        if thread is not None:
            thread.quit()
            thread.wait()
        if worker is not None:
            worker.deleteLater()
        if self.pending_buffer_save:
            self.pending_buffer_save = False
            self._save_buffer_after_stop()
            if self.pending_live_restart:
                self.pending_live_restart = False
                self._start_live()
            return
        if self.pending_live_restart:
            self._apply_parameter_change()

    def _worker_is_active(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.isRunning()

    def _show_motion_controller(self) -> None:
        show_motion_control_window(self)

    def _log_status(self, message: str) -> None:
        LOGGER.info(message)

    def _append_status(self, message: str) -> None:
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.appendPlainText(f'[{timestamp}] {message}')

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.reconfigure_timer.stop()
        self.pending_live_restart = False
        self.pending_buffer_save = False
        self._stop_worker()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
        logging.getLogger().removeHandler(self.log_handler)
        self.controller.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MmodeGui()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()