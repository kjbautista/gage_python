from __future__ import annotations

import importlib
import logging
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
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
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MOTION_STAGE_DIR = SCRIPT_DIR.parent

if str(MOTION_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_STAGE_DIR))

LOGGER = logging.getLogger(__name__)

_XPS_BACKEND = None
_XPS_BACKEND_LOCK = threading.Lock()


_MOTION_CONTROL_WINDOW: MotionControlWindow | None = None
_MOTION_CONTROL_WINDOW_LOCK = threading.Lock()


def get_xps_backend():
    """Import the Newport motion-control backend on demand."""

    global _XPS_BACKEND
    with _XPS_BACKEND_LOCK:
        if _XPS_BACKEND is None:
            try:
                xps_backend = importlib.import_module('xps_control_main')
            except Exception as exc:
                raise RuntimeError(f'Unable to import motion-control backend from {MOTION_STAGE_DIR}: {exc}') from exc
            _XPS_BACKEND = xps_backend
        return _XPS_BACKEND


def read_motion_position_mm(existing_xps=None):
    """Best-effort read of the current motion-stage position for save metadata.

    Returns a tuple ``(position_dict, xps_handle)``. ``position_dict`` is a
    ``{stage_name: position_mm}`` mapping, or ``None`` if the read failed
    (e.g. controller unreachable). ``xps_handle`` is the connection that was
    used — either ``existing_xps`` if provided, or a freshly opened handle —
    so the caller can cache it for subsequent reads. Failures are logged at
    WARNING but never raised: callers should omit the metadata field rather
    than fail the save.
    """

    try:
        xps_backend = get_xps_backend()
        xps = existing_xps if existing_xps is not None else xps_backend.get_xps_object()
    except Exception as exc:
        LOGGER.warning('Could not open XPS for motion position read: %s', exc)
        return None, existing_xps
    try:
        stage_names, positions = xps_backend.get_positions(xps)
    except Exception as exc:
        LOGGER.warning('Could not read motion-stage position: %s', exc)
        return None, xps
    return {str(name): float(pos) for name, pos in zip(stage_names, positions)}, xps


class LampIndicator(QLabel):
    """Small circular status indicator similar to MATLAB's lamp widget."""

    def __init__(self, color: str = '#ff4d4f') -> None:
        super().__init__()
        self.setFixedSize(18, 18)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        """Update the lamp color."""

        self.setStyleSheet(
            f'''
            border-radius: 9px;
            background-color: {color};
            border: 1px solid #667085;
            '''
        )


class MotionControlWindow(QMainWindow):
    """PyQt front end motion control app backed by xps_control_main.py."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Motion Controller')
        self.resize(520, 420)

        self.xps = None
        self._build_ui()
        self._check_status()

    def _build_ui(self) -> None:
        """Create the motion-control widgets and layout."""

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self.reconnect_button = QPushButton('Reconnect')
        self.reconnect_button.clicked.connect(self._check_status)
        self.initialize_button = QPushButton('Initialize')
        self.initialize_button.clicked.connect(self._initialize_stage)
        top_row.addWidget(self.reconnect_button)
        top_row.addWidget(self.initialize_button)
        top_row.addStretch(1)
        root_layout.addLayout(top_row)

        status_group = QGroupBox('Status')
        status_layout = QGridLayout(status_group)
        self.connected_lamp = LampIndicator()
        self.initialized_lamp = LampIndicator()
        self.homed_lamp = LampIndicator()
        status_layout.addWidget(self.connected_lamp, 0, 0)
        status_layout.addWidget(QLabel('Connected'), 0, 1)
        status_layout.addWidget(self.initialized_lamp, 0, 2)
        status_layout.addWidget(QLabel('Initialized'), 0, 3)
        status_layout.addWidget(self.homed_lamp, 0, 4)
        status_layout.addWidget(QLabel('Homed'), 0, 5)
        root_layout.addWidget(status_group)

        positions_group = QGroupBox('Current Positions')
        positions_layout = QFormLayout(positions_group)
        self.group1_pos = QLineEdit('0.0')
        self.group2_pos = QLineEdit('0.0')
        self.group3_pos = QLineEdit('0.0')
        for widget in (self.group1_pos, self.group2_pos, self.group3_pos):
            widget.setReadOnly(True)
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            widget.setStyleSheet(
                '''
                background-color: #d9e3ee;
                border: 1px solid #d9e3ee;
                '''
            )
        positions_layout.addRow('Group 1', self.group1_pos)
        positions_layout.addRow('Group 2', self.group2_pos)
        positions_layout.addRow('Group 3', self.group3_pos)
        root_layout.addWidget(positions_group)

        move_group = QGroupBox('Relative Move')
        move_layout = QFormLayout(move_group)
        self.group_dropdown = QComboBox()
        self.group_dropdown.addItems(['Group1', 'Group2', 'Group3'])
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(-10.0, 10.0)
        self.step_size_spin.setDecimals(3)
        self.relative_move_button = QPushButton('Relative Move')
        self.relative_move_button.setEnabled(False)
        self.relative_move_button.clicked.connect(self._relative_move)
        move_layout.addRow('Group to move', self.group_dropdown)
        move_layout.addRow('Step size (mm)', self.step_size_spin)
        move_layout.addRow(self.relative_move_button)
        root_layout.addWidget(move_group)

        report_group = QGroupBox('Status Report')
        report_layout = QVBoxLayout(report_group)
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet(
            '''
            background-color: #f2f4f7;
            border: 1px solid #d0d5dd;
            '''
        )
        report_layout.addWidget(self.status_text)
        root_layout.addWidget(report_group, 1)

    def _set_status_message(self, message: str) -> None:
        """Replace the visible status report text."""

        self.status_text.setPlainText(message)

    def _update_status_indicators(self, init_status: list[int], home_status: list[int]) -> None:
        """Update lamps and button enablement from backend status arrays."""

        self.connected_lamp.set_color('#12b76a')
        all_initialized = bool(init_status) and all(init_status)
        all_homed = bool(home_status) and all(home_status)

        self.initialized_lamp.set_color('#12b76a' if all_initialized else '#ff4d4f')
        self.homed_lamp.set_color('#12b76a' if all_homed else '#ff4d4f')
        self.relative_move_button.setEnabled(all_initialized and all_homed)
        self.initialize_button.setEnabled(not (all_initialized and all_homed))

        if all_initialized and all_homed:
            self._set_status_message('Connection successful. All axes have been initialized and homed.')

    def _update_positions(self) -> None:
        """Read current stage positions and refresh the UI."""

        if self.xps is None:
            return
        xps_backend = get_xps_backend()
        stage_names, positions = xps_backend.get_positions(self.xps)
        stage_map = dict(zip(stage_names, positions))
        self.group1_pos.setText(f"{float(stage_map.get('Group1.Pos', 0.0)):.6g}")
        self.group2_pos.setText(f"{float(stage_map.get('Group2.Pos', 0.0)):.6g}")
        self.group3_pos.setText(f"{float(stage_map.get('Group3.Pos', 0.0)):.6g}")

    def _check_status(self) -> None:
        """Connect if needed, then query motion-controller status and positions."""

        try:
            xps_backend = get_xps_backend()
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            init_status, home_status = xps_backend.check_status(self.xps)
            self._set_status_message('Connection successful.')
            self._update_status_indicators(init_status, home_status)
            self._update_positions()
        except Exception as exc:
            self.connected_lamp.set_color('#ff4d4f')
            self.initialized_lamp.set_color('#ff4d4f')
            self.homed_lamp.set_color('#ff4d4f')
            self.relative_move_button.setEnabled(False)
            self._set_status_message(str(exc))
            if 'Login failed' in str(exc):
                QMessageBox.warning(
                    self,
                    'Login failed',
                    "Turn on the motion controller, verify the network connection, and click 'Reconnect' to try again.",
                )

    def _initialize_stage(self) -> None:
        """Initialize and home all groups after explicit user confirmation."""

        selection = QMessageBox.warning(
            self,
            'Confirm initialization and homing',
            'Motion stage will initialize and home. Confirm that the hydrophone is not attached and the motion stage will not run into anything.',
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if selection != QMessageBox.StandardButton.Ok:
            self._set_status_message('Initialization and homing canceled')
            return

        try:
            xps_backend = get_xps_backend()
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            init_status, home_status = xps_backend.initialize_stage(self.xps)
            self._update_status_indicators(init_status, home_status)
            self._update_positions()
        except Exception as exc:
            self._set_status_message(str(exc))

    def _relative_move(self) -> None:
        """Execute a relative move for the selected group and refresh positions."""

        try:
            xps_backend = get_xps_backend()
            if self.xps is None:
                self.xps = xps_backend.get_xps_object()
            group_selection = self.group_dropdown.currentText()
            distance = float(self.step_size_spin.value())
            position = xps_backend.move_stage(self.xps, group_selection, distance, True)
            self._set_status_message(f'{group_selection} moved by {distance:.6g} mm. New position: {position:.6g} mm')
            self._update_positions()
        except Exception as exc:
            self._set_status_message(str(exc))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Disconnect the controller on window close."""

        global _MOTION_CONTROL_WINDOW
        if self.xps is not None:
            try:
                self.xps.disconnect()
            except Exception:
                LOGGER.warning('XPS disconnect failed during motion-control close', exc_info=True)
            self.xps = None
        with _MOTION_CONTROL_WINDOW_LOCK:
            if _MOTION_CONTROL_WINDOW is self:
                _MOTION_CONTROL_WINDOW = None
        LOGGER.info('Motion-control window closed (caller=%s)', __name__)
        super().closeEvent(event)


def show_motion_control_window(parent: QWidget | None = None) -> MotionControlWindow:
    """Show the singleton motion-control window and bring it to the front."""

    global _MOTION_CONTROL_WINDOW
    with _MOTION_CONTROL_WINDOW_LOCK:
        if _MOTION_CONTROL_WINDOW is None:
            _MOTION_CONTROL_WINDOW = MotionControlWindow()
            LOGGER.info('Motion-control window created')
        window = _MOTION_CONTROL_WINDOW
    if parent is not None and window.parent() is None:
        window.setParent(parent, window.windowFlags())
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def close_motion_control_window() -> None:
    """Close and dispose of the singleton motion-control window, if any."""

    global _MOTION_CONTROL_WINDOW
    with _MOTION_CONTROL_WINDOW_LOCK:
        window = _MOTION_CONTROL_WINDOW
        _MOTION_CONTROL_WINDOW = None
    if window is not None:
        LOGGER.info('Closing motion-control window (caller=%s)', __name__)
        window.close()


def main() -> None:
    """Launch the motion-control window as a standalone Python app."""

    app = QApplication(sys.argv)
    window = show_motion_control_window()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()