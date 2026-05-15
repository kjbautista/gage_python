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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

from run_gage.acquisition_workers import BrentSpatialPeakWorker, LiveAcquisitionWorker, build_display_fft_mag
from run_gage.gui.app_motion_control import get_xps_backend, show_motion_control_window
from run_gage.controller import GageAlineController, linear_interp_extrap
from run_gage.models import AcquisitionConfig, AcquisitionFrame, DEFAULT_SAMPLE_RATE_HZ, DisplayConfig, SUPPORTED_INPUT_RANGES_V
from read_gage.python.calibration_loader import get_names, get_entry_by_name
from run_gage.gui.constants import MOTION_GROUPS, RECONFIGURE_DEBOUNCE_MS, VOLTAGE_RANGE_OPTIONS
from run_gage.gui.style_utils import apply_gui_scaling
from run_gage.gui.plot_widgets import AlinePlotCanvas, FftPlotCanvas, SpatialPeakPlotCanvas


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent

_POSITION_POLL_INTERVAL_MS = 250


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


class FindSpatialPeakBrentGui(QMainWindow):
    """PyQt front end for two-phase Brent coordinate-descent spatial peak finding."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Spatial Peak Finder (Brent)')
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
        self.pending_peak_search_start = False
        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None

        self.xps = None
        self.axes_limits = np.zeros((3, 2), dtype=float)
        self.center_pos = np.zeros(3, dtype=float)
        self.scan_range = np.zeros(3, dtype=float)
        self.groups_to_move = np.zeros(3, dtype=bool)
        self.best_peak_position: np.ndarray | None = None
        self._search_phase: str = 'coarse'
        self._best_value_so_far: float = 0.0
        self._active_indices: np.ndarray = np.array([], dtype=int)

        self.log_emitter = StatusEmitter()
        self.log_handler = GuiLogHandler(self.log_emitter)

        self._build_ui()
        self.hardware_input_range_v = float(self.voltage_range_combo.currentText())
        apply_gui_scaling(self)
        self._connect_signals()
        self._configure_logging()
        self._update_group_controls()

        self.reconfigure_timer = QTimer(self)
        self.reconfigure_timer.setSingleShot(True)
        self.reconfigure_timer.timeout.connect(self._apply_parameter_change)

        self._position_poll_timer = QTimer(self)
        self._position_poll_timer.setInterval(_POSITION_POLL_INTERVAL_MS)
        self._position_poll_timer.timeout.connect(self._poll_current_positions)

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
        controls_layout.addWidget(self._build_peak_search_group())
        controls_layout.addWidget(self._build_current_position_group())
        controls_layout.addWidget(self._build_hydrophone_group())

        motion_button = QPushButton('Motion Controller')
        motion_button.clicked.connect(self._show_motion_controller)
        controls_layout.addWidget(motion_button)
        controls_layout.addStretch(1)

        self.fft_canvas = FftPlotCanvas()
        self.aline_canvas = AlinePlotCanvas()
        self.peak_canvas = SpatialPeakPlotCanvas()
        display_layout.addWidget(self._wrap_group('Frequency Spectrum', self.fft_canvas), 2)
        display_layout.addWidget(self._wrap_group('A-line', self.aline_canvas), 2)
        display_layout.addWidget(self._wrap_group('Peak Map', self.peak_canvas), 5)

        progress_group = QGroupBox('Search Progress')
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
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
        self.max_frequency_edit.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')

        layout.addRow('Lines to average', self.lines_average_spin)
        layout.addRow('Freq start (MHz)', self.freq_start_spin)
        layout.addRow('Freq end (MHz)', self.freq_end_spin)
        layout.addRow('Max frequency (MHz)', self.max_frequency_edit)
        return group

    def _build_peak_search_group(self) -> QGroupBox:
        group = QGroupBox('Peak Search')
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

        self.n_coarse_spin = QSpinBox()
        self.n_coarse_spin.setRange(2, 20)
        self.n_coarse_spin.setValue(5)

        self.max_sweeps_spin = QSpinBox()
        self.max_sweeps_spin.setRange(1, 10)
        self.max_sweeps_spin.setValue(3)

        self.refinement_tol_spin = QDoubleSpinBox()
        self.refinement_tol_spin.setRange(0.001, 1.0)
        self.refinement_tol_spin.setDecimals(3)
        self.refinement_tol_spin.setSingleStep(0.001)
        self.refinement_tol_spin.setValue(0.01)

        self.start_peak_search_button = QPushButton('Start Peak Search')
        self.home_button = QPushButton('Home')
        self.home_button.setEnabled(False)
        self.move_to_peak_button = QPushButton('Move to Peak')
        self.move_to_peak_button.setEnabled(False)

        self.home_coordinates_edit = QLineEdit('[]')
        self.home_coordinates_edit.setReadOnly(True)
        self.home_coordinates_edit.setStyleSheet('background-color: #e6eaf2; border: 1px solid #e6eaf2;')

        _ro_style = 'background-color: #e6eaf2; border: 1px solid #e6eaf2;'
        self.phase_label = QLabel('Ready')
        self.phase_label.setStyleSheet(_ro_style)

        layout.addWidget(self.group1_checkbox, 0, 0, 1, 2)
        layout.addWidget(self.group2_checkbox, 0, 2, 1, 2)
        layout.addWidget(self.group3_checkbox, 0, 4, 1, 2)

        layout.addWidget(QLabel('Range (mm)'), 1, 0, 1, 2)
        layout.addWidget(QLabel('Range (mm)'), 1, 2, 1, 2)
        layout.addWidget(QLabel('Range (mm)'), 1, 4, 1, 2)

        layout.addWidget(self.range_1_spin, 2, 0, 1, 2)
        layout.addWidget(self.range_2_spin, 2, 2, 1, 2)
        layout.addWidget(self.range_3_spin, 2, 4, 1, 2)

        layout.addWidget(QLabel('Coarse pts / axis'), 3, 0)
        layout.addWidget(self.n_coarse_spin, 3, 1)
        layout.addWidget(QLabel('Max sweeps'), 3, 2)
        layout.addWidget(self.max_sweeps_spin, 3, 3)
        layout.addWidget(QLabel('Tol (mm)'), 3, 4)
        layout.addWidget(self.refinement_tol_spin, 3, 5)

        layout.addWidget(QLabel('Home coordinates'), 4, 0)
        layout.addWidget(self.home_coordinates_edit, 4, 1, 1, 3)
        layout.addWidget(self.home_button, 4, 4)
        layout.addWidget(self.start_peak_search_button, 4, 5)

        layout.addWidget(QLabel('Phase'), 5, 0)
        layout.addWidget(self.phase_label, 5, 1, 1, 4)
        layout.addWidget(self.move_to_peak_button, 5, 5)

        return group

    def _build_current_position_group(self) -> QGroupBox:
        group = QGroupBox('Current Position')
        layout = QFormLayout(group)

        _ro_style = 'background-color: #e6eaf2; border: 1px solid #e6eaf2;'

        self.group1_pos_edit = QLineEdit('\u2014')
        self.group1_pos_edit.setReadOnly(True)
        self.group1_pos_edit.setStyleSheet(_ro_style)

        self.group2_pos_edit = QLineEdit('\u2014')
        self.group2_pos_edit.setReadOnly(True)
        self.group2_pos_edit.setStyleSheet(_ro_style)

        self.group3_pos_edit = QLineEdit('\u2014')
        self.group3_pos_edit.setReadOnly(True)
        self.group3_pos_edit.setStyleSheet(_ro_style)

        layout.addRow('Group1 (mm)', self.group1_pos_edit)
        layout.addRow('Group2 (mm)', self.group2_pos_edit)
        layout.addRow('Group3 (mm)', self.group3_pos_edit)
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

    def _wrap_group(self, title: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(widget)
        return group

    def _connect_signals(self) -> None:
        self.run_button.clicked.connect(self._start_live)
        self.stop_button.clicked.connect(self._stop_worker)
        self.start_peak_search_button.clicked.connect(self._start_peak_search)
        self.home_button.clicked.connect(self._move_stage_home)
        self.move_to_peak_button.clicked.connect(self._move_stage_to_peak)

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
        self._position_poll_timer.stop()
        self._start_worker_thread(worker)
        self._log_status('Live acquisition started.')

    def _start_peak_search(self) -> None:
        if self._worker_is_active():
            self.pending_live_restart = False
            self.pending_peak_search_start = True
            self._log_status('Stopping active worker before starting peak search...')
            self._stop_worker()
            return
        self._start_peak_search_after_stop()

    def _start_peak_search_after_stop(self) -> None:
        try:
            acquisition_config, display_config = self._prepare_acquisition()
            xps_backend = get_xps_backend()
            self._check_motion_stage(xps_backend)
            self._initialize_search_geometry(xps_backend)
        except Exception as exc:
            self._handle_configuration_error(exc)
            return

        self._position_poll_timer.stop()
        self.phase_label.setText('Starting...')
        self.move_to_peak_button.setEnabled(False)
        self.best_peak_position = None
        self._best_value_so_far = 0.0

        worker = BrentSpatialPeakWorker(
            self.controller,
            acquisition_config,
            display_config,
            xps_backend,
            self.xps,
            self.center_pos,
            self.groups_to_move,
            self.axes_limits,
            self.scan_range,
            n_coarse=self.n_coarse_spin.value(),
            max_sweeps=self.max_sweeps_spin.value(),
            refinement_tol=self.refinement_tol_spin.value(),
        )
        worker.frame_ready.connect(self._handle_search_frame)
        worker.point_ready.connect(self._handle_search_point)
        worker.progress_changed.connect(self.progress_bar.setValue)
        worker.phase_changed.connect(self._handle_phase_changed)
        worker.peak_found.connect(self._handle_peak_found)
        worker.status_changed.connect(self._log_status)
        worker.error.connect(self._handle_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self.current_mode = 'peak_search'
        self._start_worker_thread(worker)
        n_est = int(np.sum(self.groups_to_move)) * self.n_coarse_spin.value()
        self._log_status(f'Starting Brent peak search: ~{n_est} coarse probes + Brent refinement.')

    def _check_motion_stage(self, xps_backend) -> None:
        try:
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            init_status, home_status = xps_backend.check_status(self.xps)
            if any(status == 0 for status in init_status) or any(status == 0 for status in home_status):
                raise RuntimeError('not connected')
            _, positions = xps_backend.get_positions(self.xps)
            _, limits = xps_backend.get_limits(self.xps)
            self.center_pos = np.asarray(positions, dtype=float)
            self.axes_limits = np.asarray(limits, dtype=float)
            self.home_coordinates_edit.setText(f'[{self.center_pos[0]:.2f}, {self.center_pos[1]:.2f}, {self.center_pos[2]:.2f}]')
            self.home_button.setEnabled(True)
            self._position_poll_timer.start()
        except Exception:
            raise RuntimeError('Unable to start. Check that the motion stage is connected and initialized.')

    def _initialize_search_geometry(self, xps_backend) -> None:
        del xps_backend
        self.scan_range = np.array(
            [self.range_1_spin.value(), self.range_2_spin.value(), self.range_3_spin.value()],
            dtype=float,
        )
        self.groups_to_move = np.array(
            [self.group1_checkbox.isChecked(), self.group2_checkbox.isChecked(), self.group3_checkbox.isChecked()],
            dtype=bool,
        )
        if not np.any(self.groups_to_move):
            raise RuntimeError('Select at least one motion group for peak searching.')
        self._configure_search_axes()
        self._log_search_summary()

    def _configure_search_axes(self) -> None:
        active = np.flatnonzero(self.groups_to_move)
        self._active_indices = active
        n_active = int(len(active))
        is_1d = n_active == 1

        if is_1d:
            x_label = f'Group{active[0] + 1} (mm)'
            y_label = 'Metric (V)'
        elif n_active == 2:
            x_label = f'Group{active[1] + 1} (mm)'
            y_label = f'Group{active[0] + 1} (mm)'
        else:
            x_label = 'Group3 (mm)'
            y_label = 'Group1 (mm)'

        self.peak_canvas.reset(x_label, y_label, is_1d)

    def _log_search_summary(self) -> None:
        active_indices = np.flatnonzero(self.groups_to_move)
        n_active = int(len(active_indices))
        n_coarse = self.n_coarse_spin.value()
        max_sweeps = self.max_sweeps_spin.value()
        tol = self.refinement_tol_spin.value()
        lines = [
            f'Brent peak search: {n_active} active axis/axes',
            f'Coarse probes: {n_active} \u00d7 {n_coarse} = {n_active * n_coarse}',
            f'Refinement: Brent per axis, max {max_sweeps} sweep(s), tol={tol} mm',
        ]
        for axis_idx in active_indices:
            lo = max(self.axes_limits[axis_idx, 0], self.center_pos[axis_idx] - self.scan_range[axis_idx] / 2.0)
            hi = min(self.axes_limits[axis_idx, 1], self.center_pos[axis_idx] + self.scan_range[axis_idx] / 2.0)
            lines.append(f'Group{axis_idx + 1}: {lo:.2f} to {hi:.2f} mm')
        self.status_text.setPlainText('\n'.join(lines))

    def _move_stage_home(self) -> None:
        try:
            xps_backend = get_xps_backend()
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            for group_index, group_name in enumerate(MOTION_GROUPS):
                xps_backend.move_stage(self.xps, group_name, float(self.center_pos[group_index]), relative=False)
            self._log_status('Motion stage returned to home coordinates.')
        except Exception as exc:
            self._handle_configuration_error(exc)

    def _move_stage_to_peak(self) -> None:
        if self.best_peak_position is None:
            return
        try:
            xps_backend = get_xps_backend()
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            for group_index, group_name in enumerate(MOTION_GROUPS):
                xps_backend.move_stage(self.xps, group_name, float(self.best_peak_position[group_index]), relative=False)
            self._log_status(
                f'Moved to peak: [{self.best_peak_position[0]:.4f}, '
                f'{self.best_peak_position[1]:.4f}, {self.best_peak_position[2]:.4f}] mm'
            )
        except Exception as exc:
            self._handle_configuration_error(exc)

    def _poll_current_positions(self) -> None:
        if self.xps is None:
            return
        try:
            xps_backend = get_xps_backend()
            _, positions = xps_backend.get_positions(self.xps)
            pos = np.asarray(positions, dtype=float)
            self.group1_pos_edit.setText(f'{pos[0]:.4f}')
            self.group2_pos_edit.setText(f'{pos[1]:.4f}')
            self.group3_pos_edit.setText(f'{pos[2]:.4f}')
        except Exception:
            pass

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

    def _handle_search_frame(self, frame: AcquisitionFrame) -> None:
        self._remember_fft_frame(frame)
        self._render_frame(frame, frame.freq_axis_mhz)

    def _handle_search_point(self, position: object, value: float, step_index: int, total_positions: int) -> None:
        pos = np.asarray(position, dtype=float)
        self.group1_pos_edit.setText(f'{pos[0]:.4f}')
        self.group2_pos_edit.setText(f'{pos[1]:.4f}')
        self.group3_pos_edit.setText(f'{pos[2]:.4f}')

        sx, sy = self._pos_to_scatter(pos, float(value))
        self.peak_canvas.add_point(sx, sy, float(value), self._search_phase)

        if float(value) > self._best_value_so_far:
            self._best_value_so_far = float(value)
            self.peak_canvas.update_best(sx, sy)

        self.progress_bar.setValue(int(round(step_index / max(total_positions, 1) * 100.0)))

    def _pos_to_scatter(self, pos: np.ndarray, value: float) -> tuple[float, float]:
        n_active = int(len(self._active_indices))
        if n_active == 1:
            return float(pos[self._active_indices[0]]), value
        elif n_active == 2:
            return float(pos[self._active_indices[1]]), float(pos[self._active_indices[0]])
        else:
            return float(pos[2]), float(pos[0])

    def _handle_phase_changed(self, phase: str) -> None:
        self._search_phase = phase
        if phase == 'coarse':
            self.phase_label.setText('Per-axis coarse scan...')
        elif phase == 'refine':
            self.phase_label.setText('Brent coordinate-descent...')
        else:
            self.phase_label.setText(phase)

    def _handle_peak_found(self, position: object, value: float) -> None:
        self.best_peak_position = np.asarray(position, dtype=float)
        pos = self.best_peak_position
        self.phase_label.setText(
            f'Peak found: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] mm  (metric={value:.6g})'
        )
        self.move_to_peak_button.setEnabled(True)
        self._log_status(
            f'Peak found at [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] mm  metric={value:.6g}'
        )
        # Auto-move to peak
        self._move_stage_to_peak()
        # Resume position polling
        if self.xps is not None:
            self._position_poll_timer.start()

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
        frames = list(self.fft_frame_history)[-max(1, display_config.lines_to_average):]
        fft_magnitudes = [build_display_fft_mag(frame, display_config) for frame in frames]
        return np.mean(np.vstack(fft_magnitudes), axis=0)

    def _handle_worker_error(self, payload) -> None:
        short_message = getattr(payload, 'message', None) or str(payload)
        self.progress_bar.setValue(0)
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
        if self.pending_peak_search_start:
            self.pending_peak_search_start = False
            self._start_peak_search_after_stop()
            return
        if self.pending_live_restart:
            self._apply_parameter_change()
        # Re-enable position polling if XPS is available
        if self.xps is not None and not self._position_poll_timer.isActive():
            self._position_poll_timer.start()

    def _worker_is_active(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.isRunning()

    def _update_group_controls(self) -> None:
        self.range_1_spin.setEnabled(self.group1_checkbox.isChecked())
        self.range_2_spin.setEnabled(self.group2_checkbox.isChecked())
        self.range_3_spin.setEnabled(self.group3_checkbox.isChecked())

    def _show_motion_controller(self) -> None:
        show_motion_control_window(self)

    def _log_status(self, message: str) -> None:
        LOGGER.info(message)

    def _append_status(self, message: str) -> None:
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.appendPlainText(f'[{timestamp}] {message}')

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.reconfigure_timer.stop()
        self._position_poll_timer.stop()
        self.pending_live_restart = False
        self.pending_peak_search_start = False
        self._stop_worker()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
        logging.getLogger().removeHandler(self.log_handler)
        self.controller.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = FindSpatialPeakBrentGui()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
