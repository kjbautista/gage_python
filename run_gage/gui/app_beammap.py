from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from run_gage.acquisition_workers import BeammapWorker, LiveAcquisitionWorker, build_display_fft_mag
from run_gage.gui.app_motion_control import get_xps_backend, show_motion_control_window
from run_gage.beammap_utils import calculate_beammap_scan_positions
from run_gage.controller import GageAlineController, linear_interp_extrap
from run_gage.models import AcquisitionConfig, AcquisitionFrame, DEFAULT_SAMPLE_RATE_HZ, DisplayConfig, SUPPORTED_INPUT_RANGES_V
from read_gage.python.calibration_loader import get_names, get_entry_by_name
from run_gage.gui.constants import MOTION_GROUPS, RECONFIGURE_DEBOUNCE_MS, VOLTAGE_RANGE_OPTIONS
from run_gage.gui.style_utils import apply_gui_scaling
from run_gage.gui.plot_widgets import AlinePlotCanvas, BeammapPlotCanvas, FftPlotCanvas


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
        try:
            self.emitter.message.emit(self.format(record))
        except Exception:
            self.handleError(record)


class BeammapGui(QMainWindow):
    """PyQt front end for live A-line preview and motion-stage beammapping."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Beammap Acquisition')
        self.resize(1800, 1200)

        self.controller = GageAlineController()
        self.calibration_vals = np.empty((0, 2))
        self.last_frame: AcquisitionFrame | None = None
        self.last_fft_freq_axis = np.array([], dtype=float)
        self.last_fft_values = np.array([], dtype=float)
        self.fft_frame_history: deque[AcquisitionFrame] = deque()
        self.current_mode: str | None = None
        self.hardware_ready = False
        self.pending_live_restart = False
        self.pending_beammap_start = False
        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None
        self.beammap_start_time: float | None = None

        self.xps = None
        self.axes_limits = np.zeros((3, 2), dtype=float)
        self.center_pos = np.zeros(3, dtype=float)
        self.home_pos = np.zeros(3, dtype=float)
        self.scan_positions = np.empty((0, 3), dtype=float)
        self.groups_to_move = np.zeros(3, dtype=bool)
        self.dim1 = np.array([], dtype=float)
        self.dim2 = np.array([], dtype=float)
        self.beammap_x_label = ''
        self.beammap_y_label = ''
        self.beammap_reverse_x = False
        self.live_beammap = np.empty((0, 0), dtype=float)
        self.save_folder = Path.home() / 'Documents'
        self.current_beammap_save_dir: Path | None = None
        # Per-user temp file. The PID suffix prevents two GUI instances on the same account
        # from clobbering each other's recovery state.
        self.home_coordinates_tempfile = (
            Path(tempfile.gettempdir())
            / f'gage_beammap_home_coordinates_{os.getpid()}.json'
        )

        self.log_emitter = StatusEmitter()
        self.log_handler = GuiLogHandler(self.log_emitter)

        self._build_ui()
        self.hardware_input_range_v = float(self.voltage_range_combo.currentText())
        apply_gui_scaling(self)
        self._connect_signals()
        self._configure_logging()
        self._update_group_controls()
        self._update_calculator()

        self.reconfigure_timer = QTimer(self)
        self.reconfigure_timer.setSingleShot(True)
        self.reconfigure_timer.timeout.connect(self._apply_parameter_change)

        QTimer.singleShot(10, self._initialize_hardware)

    def _build_ui(self) -> None:
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
        self.board_label.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')

        controls_layout.addWidget(self._build_system_group())
        controls_layout.addWidget(self._build_run_group())
        controls_layout.addLayout(self._build_controls_row(self._build_acquisition_group(), self._build_fft_group()))
        controls_layout.addWidget(self._build_beammap_group())
        controls_layout.addLayout(self._build_controls_row(self._build_hydrophone_group(), self._build_calculator_group()))

        motion_button = QPushButton('Motion Controller')
        motion_button.clicked.connect(self._show_motion_controller)
        controls_layout.addWidget(motion_button)
        controls_layout.addStretch(1)

        self.fft_canvas = FftPlotCanvas()
        self.aline_canvas = AlinePlotCanvas()
        self.beammap_canvas = BeammapPlotCanvas()
        display_layout.addWidget(self._wrap_group('Frequency Spectrum', self.fft_canvas), 2)
        display_layout.addWidget(self._wrap_group('A-line', self.aline_canvas), 2)
        display_layout.addWidget(self._wrap_group('Beammap', self.beammap_canvas), 5)

        progress_group = QGroupBox('Beammap Progress')
        progress_layout = QHBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar, 1)
        self.etr_label = QLabel('Est. time remaining: —')
        self.etr_label.setMinimumWidth(120)
        progress_layout.addWidget(self.etr_label)
        display_layout.addWidget(progress_group)

        status_group = QGroupBox('Status')
        status_layout = QVBoxLayout(status_group)
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        display_layout.addWidget(status_group, 1)

    def _build_controls_row(self, left_group: QGroupBox, right_group: QGroupBox) -> QHBoxLayout:
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

        self.freq_start_spin = QDoubleSpinBox()
        self.freq_start_spin.setRange(0.0, DEFAULT_SAMPLE_RATE_HZ / 2e6)
        self.freq_start_spin.setDecimals(3)

        self.freq_end_spin = QDoubleSpinBox()
        self.freq_end_spin.setRange(0.0, DEFAULT_SAMPLE_RATE_HZ / 2e6)
        self.freq_end_spin.setDecimals(3)
        self.freq_end_spin.setValue(10.0)

        self.max_frequency_edit = QLineEdit('0.0')
        self.max_frequency_edit.setReadOnly(True)
        self.max_frequency_edit.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')

        layout.addRow('Freq start (MHz)', self.freq_start_spin)
        layout.addRow('Freq end (MHz)', self.freq_end_spin)
        layout.addRow('Max frequency (MHz)', self.max_frequency_edit)
        return group

    def _build_beammap_group(self) -> QGroupBox:
        group = QGroupBox('Beammap')
        layout = QGridLayout(group)

        self.group1_checkbox = QCheckBox('Group 1')
        self.group1_checkbox.setChecked(False)
        self.group2_checkbox = QCheckBox('Group 2')
        self.group2_checkbox.setChecked(False)
        self.group3_checkbox = QCheckBox('Group 3')
        self.group3_checkbox.setChecked(False)

        self.range_1_spin = QDoubleSpinBox(); self.range_1_spin.setRange(0.0, 100.0); self.range_1_spin.setValue(1.0); self.range_1_spin.setDecimals(3)
        self.range_2_spin = QDoubleSpinBox(); self.range_2_spin.setRange(0.0, 100.0); self.range_2_spin.setValue(1.0); self.range_2_spin.setDecimals(3)
        self.range_3_spin = QDoubleSpinBox(); self.range_3_spin.setRange(0.0, 100.0); self.range_3_spin.setValue(1.0); self.range_3_spin.setDecimals(3)

        self.step_1_spin = QDoubleSpinBox(); self.step_1_spin.setRange(0.0, 10.0); self.step_1_spin.setValue(0.1); self.step_1_spin.setDecimals(4)
        self.step_2_spin = QDoubleSpinBox(); self.step_2_spin.setRange(0.0, 10.0); self.step_2_spin.setValue(0.1); self.step_2_spin.setDecimals(4)
        self.step_3_spin = QDoubleSpinBox(); self.step_3_spin.setRange(0.0, 10.0); self.step_3_spin.setValue(0.1); self.step_3_spin.setDecimals(4)

        self.start_beammap_button = QPushButton('Start Beammap')
        self.home_button = QPushButton('Move to Home')
        self.home_button.setEnabled(False)
        self.home_coordinates_edit = QLineEdit('[]')
        self.home_coordinates_edit.setReadOnly(True)
        self.home_coordinates_edit.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')

        self.lines_average_spin = QSpinBox()
        self.lines_average_spin.setRange(1, 99999)
        self.lines_average_spin.setValue(1)

        layout.addWidget(self.group1_checkbox, 0, 0, 1, 2)
        layout.addWidget(self.group2_checkbox, 0, 2, 1, 2)
        layout.addWidget(self.group3_checkbox, 0, 4, 1, 2)

        layout.addWidget(QLabel('Range (mm)'), 1, 0)
        layout.addWidget(QLabel('Step (mm)'), 1, 1)
        layout.addWidget(QLabel('Range (mm)'), 1, 2)
        layout.addWidget(QLabel('Step (mm)'), 1, 3)
        layout.addWidget(QLabel('Range (mm)'), 1, 4)
        layout.addWidget(QLabel('Step (mm)'), 1, 5)

        layout.addWidget(self.range_1_spin, 2, 0)
        layout.addWidget(self.step_1_spin, 2, 1)
        layout.addWidget(self.range_2_spin, 2, 2)
        layout.addWidget(self.step_2_spin, 2, 3)
        layout.addWidget(self.range_3_spin, 2, 4)
        layout.addWidget(self.step_3_spin, 2, 5)

        layout.addWidget(QLabel('Lines to average'), 3, 0)
        layout.addWidget(self.lines_average_spin, 3, 1)

        layout.addWidget(QLabel('Home coordinates'), 3, 2)
        layout.addWidget(self.home_coordinates_edit, 3, 3, 1, 2)
        layout.addWidget(self.home_button, 3, 5, 1, 2)
        layout.addWidget(self.start_beammap_button, 4, 0, 1, 6)

        
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

    def _build_calculator_group(self) -> QGroupBox:
        group = QGroupBox('Calculator')
        layout = QFormLayout(group)

        self.speed_spin = QDoubleSpinBox(); self.speed_spin.setRange(1.0, 10000.0); self.speed_spin.setDecimals(2); self.speed_spin.setValue(1480.0)
        self.focal_spin = QDoubleSpinBox(); self.focal_spin.setRange(0.0, 1000.0); self.focal_spin.setDecimals(3)
        self.tof_edit = QLineEdit('0.0'); self.tof_edit.setReadOnly(True); self.tof_edit.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')
        self.frequency_spin = QDoubleSpinBox(); self.frequency_spin.setRange(0.0, 1000.0); self.frequency_spin.setDecimals(3); self.frequency_spin.setValue(0.0)
        self.wavelength_edit = QLineEdit('0.0'); self.wavelength_edit.setReadOnly(True); self.wavelength_edit.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')
        self.ppw_spin = QDoubleSpinBox(); self.ppw_spin.setRange(0.1, 1000.0); self.ppw_spin.setDecimals(3); self.ppw_spin.setValue(4.0)
        self.max_step_size_edit = QLineEdit('0.0'); self.max_step_size_edit.setReadOnly(True); self.max_step_size_edit.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')

        layout.addRow('Speed of sound (m/s)', self.speed_spin)
        layout.addRow('Focal length (mm)', self.focal_spin)
        layout.addRow('Time of flight (us)', self.tof_edit)
        layout.addRow('Frequency (MHz)', self.frequency_spin)
        layout.addRow('Wavelength (mm)', self.wavelength_edit)
        layout.addRow('PPW', self.ppw_spin)
        layout.addRow('Max step size (mm)', self.max_step_size_edit)
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
        self.start_beammap_button.clicked.connect(self._start_beammap)
        self.home_button.clicked.connect(self._move_stage_home)

        self.speed_spin.valueChanged.connect(self._update_calculator)
        self.focal_spin.valueChanged.connect(self._update_calculator)
        self.frequency_spin.valueChanged.connect(self._update_calculator)
        self.ppw_spin.valueChanged.connect(self._update_calculator)

        self.lines_average_spin.valueChanged.connect(self._refresh_existing_plots)
        self.freq_start_spin.valueChanged.connect(self._refresh_existing_plots)
        self.freq_end_spin.valueChanged.connect(self._refresh_existing_plots)
        self.offset_spin.valueChanged.connect(self._refresh_existing_plots)
        self.voltage_range_combo.currentTextChanged.connect(self._on_voltage_range_changed)
        self.log_emitter.message.connect(self._append_status)

        self.time_start_spin.valueChanged.connect(self._schedule_parameter_change)
        self.time_end_spin.valueChanged.connect(self._schedule_parameter_change)
        self.trigger_spin.valueChanged.connect(self._schedule_parameter_change)
        self.calibration_name_combo.currentTextChanged.connect(self._on_calibration_selection_changed)

        self.group1_checkbox.toggled.connect(self._update_group_controls)
        self.group2_checkbox.toggled.connect(self._update_group_controls)
        self.group3_checkbox.toggled.connect(self._update_group_controls)

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
            n_alines=max(1, self.lines_average_spin.value()),
            data_type='beammap',
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
        holdoff = acq_info.get('TriggerHoldoff', 0)
        self._log_status(
            f"Configured acquisition: Fs={acq_info['SampleRate']}, Depth={acq_info['Depth']}, SegmentSize={acq_info['SegmentSize']}, Holdoff={holdoff}"
        )
        self.progress_bar.setValue(0)
        self._refresh_calibration_state(acquisition_config)
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

    def _start_beammap(self) -> None:
        if self._worker_is_active():
            self.pending_live_restart = False
            self.pending_beammap_start = True
            self._log_status('Stopping active worker before starting beammap...')
            self._stop_worker()
            return
        self._start_beammap_after_stop()

    def _start_beammap_after_stop(self) -> None:
        try:
            acquisition_config, display_config = self._prepare_acquisition()
            xps_backend = get_xps_backend()
            self._check_motion_stage(xps_backend)
            self._initialize_beammap_geometry(xps_backend)
        except Exception as exc:
            self._handle_configuration_error(exc)
            return

        reply = QMessageBox.question(
            self,
            'Confirm Beammap',
            self._build_beammap_confirmation_text(),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._log_status('Beammap cancelled.')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_path = str(self.save_folder / f'beammap_{timestamp}')
        save_path, _ = QFileDialog.getSaveFileName(self, 'Select folder for beammap', default_path, '')
        if not save_path:
            self._log_status('Beammap cancelled.')
            return
        self.save_folder = Path(save_path).parent
        self.current_beammap_save_dir = Path(save_path)
        self.current_beammap_save_dir.mkdir(parents=True, exist_ok=True)
        self._write_home_coordinates()

        worker = BeammapWorker(
            self.controller,
            acquisition_config,
            display_config,
            xps_backend,
            self.xps,
            self.scan_positions,
            self.current_beammap_save_dir,
            self.center_pos,
        )
        worker.frame_ready.connect(self._handle_beammap_frame)
        worker.beammap_point_ready.connect(self._handle_beammap_point)
        worker.progress_changed.connect(self._handle_beammap_progress)
        worker.status_changed.connect(self._log_status)
        worker.completed.connect(self._handle_beammap_completed)
        worker.error.connect(self._handle_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self.current_mode = 'beammap'
        self.beammap_start_time = time.monotonic()
        self.etr_label.setText('Est. time remaining: —')
        self._start_worker_thread(worker)
        self._log_status(f'Starting beammap with {self.scan_positions.shape[0]} scan positions.')

    def _check_motion_stage(self, xps_backend) -> None:
        try:
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            init_status, home_status = xps_backend.check_status(self.xps)
            if any(status == 0 for status in init_status) or any(status == 0 for status in home_status):
                raise RuntimeError('not connected')
            _, positions = xps_backend.get_positions(self.xps)
            _, limits = xps_backend.get_limits(self.xps)

            current = np.asarray(positions, dtype=float)
            self.center_pos = current

            home_coords = self._read_home_coordinates()
            if home_coords is not None:
                self.home_pos = home_coords
            else:
                self.home_pos = current.copy()

            self.axes_limits = np.asarray(limits, dtype=float)
            self.home_coordinates_edit.setText(
                f'[{self.home_pos[0]:.2f}, {self.home_pos[1]:.2f}, {self.home_pos[2]:.2f}]'
            )
            self.home_button.setEnabled(True)
        except Exception:
            raise RuntimeError('Unable to start. Check that the motion stage is connected and initialized.')

    def _initialize_beammap_geometry(self, xps_backend) -> None:
        del xps_backend
        step_size = np.array([self.step_1_spin.value(), self.step_2_spin.value(), self.step_3_spin.value()], dtype=float)
        scan_range = np.array([self.range_1_spin.value(), self.range_2_spin.value(), self.range_3_spin.value()], dtype=float)
        self.groups_to_move = np.array(
            [self.group1_checkbox.isChecked(), self.group2_checkbox.isChecked(), self.group3_checkbox.isChecked()],
            dtype=bool,
        )
        if not np.any(self.groups_to_move):
            raise RuntimeError('Select at least one motion group for beammapping.')
        scan_range[~self.groups_to_move] = 0.0
        self.scan_positions = calculate_beammap_scan_positions(self.center_pos, step_size, scan_range, self.axes_limits)
        self._configure_beammap_axes()
        self._log_beammap_scan_summary()

    def _read_home_coordinates(self) -> np.ndarray | None:
        """Read home coordinates from temporary file if it exists.
        
        Returns:
            numpy array of shape (3,) or None if file doesn't exist or is invalid.
        """
        try:
            if self.home_coordinates_tempfile.exists():
                with open(self.home_coordinates_tempfile, 'r') as f:
                    data = json.load(f)
                return np.asarray(data['home_coordinates'], dtype=float)
        except Exception as exc:
            LOGGER.warning('Failed to read home coordinates from temp file: %s', exc)
        return None

    def _write_home_coordinates(self) -> None:
        """Write current home coordinates to temporary file for persistence across app runs."""
        try:
            payload = {
                'home_coordinates': self.home_pos.tolist(),
                'timestamp': datetime.now().isoformat(),
            }
            with open(self.home_coordinates_tempfile, 'w') as f:
                json.dump(payload, f, indent=2)
            LOGGER.debug('Home coordinates saved to %s', self.home_coordinates_tempfile)
        except Exception as exc:
            LOGGER.warning('Failed to write home coordinates to temp file: %s', exc)

    def _delete_home_coordinates(self) -> None:
        """Delete the temporary home coordinates file on normal app close."""
        try:
            if self.home_coordinates_tempfile.exists():
                self.home_coordinates_tempfile.unlink()
                LOGGER.debug('Home coordinates temp file deleted: %s', self.home_coordinates_tempfile)
        except Exception as exc:
            LOGGER.warning('Failed to delete home coordinates temp file: %s', exc)

    def _configure_beammap_axes(self) -> None:
        group1_dims = np.sort(np.unique(self.scan_positions[:, 0]))
        group2_dims = np.sort(np.unique(self.scan_positions[:, 1]))
        group3_dims = np.sort(np.unique(self.scan_positions[:, 2]))
        self.dim1, self.dim2, self.beammap_x_label, self.beammap_y_label, self.beammap_reverse_x = resolve_beammap_axes(
            self.groups_to_move,
            group1_dims,
            group2_dims,
            group3_dims,
        )

        self.live_beammap = np.zeros((len(self.dim1), len(self.dim2)), dtype=float)
        self.beammap_canvas.update_map(
            self.live_beammap,
            self.dim1,
            self.dim2,
            self.beammap_x_label,
            self.beammap_y_label,
            self.beammap_reverse_x,
        )

    def _log_beammap_scan_summary(self) -> None:
        lines = [f'Total number of scan positions: {self.scan_positions.shape[0]}']
        if self.groups_to_move[0]:
            lines.append(f'Group1: {self.dim1.min() if self.dim1.size else 0:.2f} to {self.dim1.max() if self.dim1.size else 0:.2f}')
        if self.groups_to_move[1]:
            group2_dims = np.sort(np.unique(self.scan_positions[:, 1]))
            lines.append(f'Group2: {group2_dims.min():.2f} to {group2_dims.max():.2f}')
        if self.groups_to_move[2]:
            group3_dims = np.sort(np.unique(self.scan_positions[:, 2]))
            lines.append(f'Group3: {group3_dims.min():.2f} to {group3_dims.max():.2f}')
        if not np.allclose(self.center_pos, self.home_pos, atol=1e-3):
            lines.append('Warning: Current start position is different from the stored home position.')
        self.status_text.setPlainText('\n'.join(lines))

    def _build_beammap_confirmation_text(self) -> str:
        lines = [f'Total scan positions: {self.scan_positions.shape[0]}']
        group_names = ('Group1', 'Group2', 'Group3')
        step_spins = (self.step_1_spin, self.step_2_spin, self.step_3_spin)
        range_spins = (self.range_1_spin, self.range_2_spin, self.range_3_spin)
        for i, (name, step_spin, range_spin) in enumerate(zip(group_names, step_spins, range_spins)):
            if not self.groups_to_move[i]:
                continue
            dims = np.sort(np.unique(self.scan_positions[:, i]))
            n_steps = len(dims)
            lines.append(
                f'{name}:  range = {range_spin.value():.3g} mm,  step = {step_spin.value():.3g} mm,  steps = {n_steps}'
            )
        lines.append(f'Calibration: {self.calibration_name_combo.currentText()}')
        return '\n'.join(lines)

    def _move_stage_home(self) -> None:
        try:
            xps_backend = get_xps_backend()
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            for group_index, group_name in enumerate(MOTION_GROUPS):
                xps_backend.move_stage(self.xps, group_name, float(self.home_pos[group_index]), relative=False)
            self._log_status('Motion stage returned to home coordinates.')
        except Exception as exc:
            self._handle_configuration_error(exc)

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
        del fft_average
        frame.min_voltage = min_voltage
        self._remember_fft_frame(frame)
        self._render_frame(frame, freq_axis_mhz)

    def _handle_beammap_frame(self, frame: AcquisitionFrame) -> None:
        self._remember_fft_frame(frame)
        self._render_frame(frame, frame.freq_axis_mhz)

    def _handle_beammap_point(self, position: object, value: float, step_index: int, total_positions: int) -> None:
        pos = np.asarray(position, dtype=float)
        if np.all(self.groups_to_move):
            if (step_index - 1) % (len(self.dim1) * len(self.dim2)) == 0:
                self.live_beammap = np.zeros((len(self.dim1), len(self.dim2)), dtype=float)
            pos_to_update = np.array([pos[0], pos[2]], dtype=float)
        elif np.sum(self.groups_to_move) == 2:
            pos_to_update = pos[self.groups_to_move]
        else:
            pos_to_update = np.array([0.0, pos[self.groups_to_move][0]], dtype=float)

        row_index = int(np.argmin(np.abs(self.dim1 - pos_to_update[0])))
        col_index = int(np.argmin(np.abs(self.dim2 - pos_to_update[1])))
        self.live_beammap[row_index, col_index] = float(value)
        self.beammap_canvas.update_map(
            self.live_beammap,
            self.dim1,
            self.dim2,
            self.beammap_x_label,
            self.beammap_y_label,
            self.beammap_reverse_x,
        )
        self.progress_bar.setValue(int(round(step_index / total_positions * 100.0)))

    def _render_frame(self, frame: AcquisitionFrame, freq_axis_mhz: np.ndarray) -> None:
        display_config = self._collect_display_config()
        self.last_frame = frame
        self.last_fft_freq_axis = np.asarray(freq_axis_mhz)
        self.last_fft_values = self._build_rolling_fft_average(display_config)
        self.aline_canvas.update_frame(frame.time_axis_us, frame.volts_data, frame.min_voltage, display_config)
        self.fft_canvas.update_spectrum(self.last_fft_freq_axis, self.last_fft_values, display_config)
        if self.last_fft_values.size:
            max_index = int(np.argmax(self.last_fft_values))
            max_frequency_mhz = float(self.last_fft_freq_axis[max_index])
            self.max_frequency_edit.setText(f'{max_frequency_mhz:.6f}')
            self._update_pressure_estimate(max_frequency_mhz, frame.min_voltage)

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

    def _update_calculator(self) -> None:
        speed_of_sound = self.speed_spin.value()
        focal_length_mm = self.focal_spin.value()
        frequency_hz = self.frequency_spin.value() * 1e6
        tof_s = (focal_length_mm * 1e-3) / speed_of_sound if speed_of_sound else 0.0
        wavelength_mm = (speed_of_sound / frequency_hz) * 1e3 if frequency_hz > 0 else 0.0
        max_step_mm = wavelength_mm / self.ppw_spin.value() if self.ppw_spin.value() else 0.0
        self.tof_edit.setText(f'{tof_s * 1e6:.6g}')
        self.wavelength_edit.setText(f'{wavelength_mm:.6g}')
        self.max_step_size_edit.setText(f'{max_step_mm:.6g}')

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

    def _handle_beammap_progress(self, percent: int) -> None:
        self.progress_bar.setValue(percent)
        start = self.beammap_start_time
        if start is None or percent <= 0:
            self.etr_label.setText('Est. time remaining: —')
            return
        if percent >= 100:
            self.etr_label.setText('Est. time remaining: 00:00')
            return
        elapsed = time.monotonic() - start
        remaining = elapsed * (100.0 - percent) / percent
        self.etr_label.setText(f'Est. time remaining: {self._format_duration(remaining)}')

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f'{hours:d}:{minutes:02d}:{secs:02d}'
        return f'{minutes:02d}:{secs:02d}'

    def _handle_beammap_completed(self, save_dir: str) -> None:
        self.progress_bar.setValue(0)
        self.beammap_start_time = None
        self.etr_label.setText('Est. time remaining: —')
        self._log_status(f'Beammap completed and saved to {save_dir}')
        QMessageBox.information(self, 'Beammap Complete', f'Beammap saved to:\n{save_dir}')

    def _handle_worker_error(self, payload) -> None:
        short_message = getattr(payload, 'message', None) or str(payload)
        self.progress_bar.setValue(0)
        self.beammap_start_time = None
        self.etr_label.setText('Est. time remaining: —')
        self._log_status(f'Error: {payload}')
        QMessageBox.critical(self, 'Acquisition Error', short_message)

    def _handle_configuration_error(self, exc: Exception) -> None:
        self._log_status(f'Configuration error: {exc}')
        QMessageBox.critical(self, 'Configuration Error', str(exc))

    def _on_worker_finished(self) -> None:
        thread = self.worker_thread
        worker = self.worker
        finished_mode = self.current_mode
        self.worker_thread = None
        self.worker = None
        self.current_mode = None
        if thread is not None:
            thread.quit()
            thread.wait()
        if worker is not None:
            worker.deleteLater()
        if self.pending_beammap_start:
            self.pending_beammap_start = False
            self._start_beammap_after_stop()
            return
        if self.pending_live_restart:
            self._apply_parameter_change()
            return
        if finished_mode == 'beammap':
            self._start_live()

    def _worker_is_active(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.isRunning()

    def _update_group_controls(self) -> None:
        self.range_1_spin.setEnabled(self.group1_checkbox.isChecked())
        self.step_1_spin.setEnabled(self.group1_checkbox.isChecked())
        self.range_2_spin.setEnabled(self.group2_checkbox.isChecked())
        self.step_2_spin.setEnabled(self.group2_checkbox.isChecked())
        self.range_3_spin.setEnabled(self.group3_checkbox.isChecked())
        self.step_3_spin.setEnabled(self.group3_checkbox.isChecked())
        self._update_beammap_axes_preview()

    def _update_beammap_axes_preview(self) -> None:
        groups = np.array(
            [self.group1_checkbox.isChecked(), self.group2_checkbox.isChecked(), self.group3_checkbox.isChecked()],
            dtype=bool,
        )
        if not np.any(groups):
            return
        dummy = np.array([0.0])
        try:
            _, _, x_label, y_label, _ = resolve_beammap_axes(groups, dummy, dummy, dummy)
        except RuntimeError:
            return
        self.beammap_x_label = x_label
        self.beammap_y_label = y_label
        self.beammap_canvas.update_axis_labels(x_label, y_label)

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
        self.pending_beammap_start = False
        self._stop_worker()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
        logging.getLogger().removeHandler(self.log_handler)
        self.controller.close()
        self._delete_home_coordinates()
        super().closeEvent(event)


def resolve_beammap_axes(
    groups_to_move: np.ndarray,
    group1_dims: np.ndarray,
    group2_dims: np.ndarray,
    group3_dims: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, str, bool]:
    groups_to_move = np.asarray(groups_to_move, dtype=bool)

    if np.all(groups_to_move) or np.array_equal(groups_to_move, np.array([True, False, True])):
        # Group2 is intentionally dropped from the 2D map; scan plane is Group1 (rows) x Group3 (columns).
        return group1_dims, group3_dims, 'Group3 (mm)', 'Group1 (mm)', True

    if np.array_equal(groups_to_move, np.array([True, True, False])):
        return group1_dims, group2_dims, 'Group2 (mm)', 'Group1 (mm)', False

    if np.array_equal(groups_to_move, np.array([False, True, True])):
        return group2_dims, group3_dims, 'Group3 (mm)', 'Group2 (mm)', True

    if np.sum(groups_to_move) == 1:
        axis_index = int(np.flatnonzero(groups_to_move)[0])
        dims = [group1_dims, group2_dims, group3_dims][axis_index]
        return np.array([0.0], dtype=float), dims, f'Group{axis_index + 1} (mm)', '', axis_index == 2

    raise RuntimeError('Unsupported beammap group combination.')


def main() -> None:
    app = QApplication(sys.argv)
    window = BeammapGui()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()