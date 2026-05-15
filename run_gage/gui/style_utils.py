from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMainWindow


FONT_POINT_SIZE = 11
BUTTON_MIN_HEIGHT_PX = 44
INPUT_MIN_HEIGHT_PX = 34


def apply_gui_scaling(window: QMainWindow) -> None:
    font = QFont(window.font())
    font.setPointSize(FONT_POINT_SIZE)
    window.setFont(font)

    scaling_stylesheet = f"""
    QWidget {{
        font-size: {FONT_POINT_SIZE}pt;
    }}
    QPushButton {{
        min-height: {BUTTON_MIN_HEIGHT_PX}px;
        padding: 8px 18px;
    }}
    QLineEdit,
    QComboBox,
    QAbstractSpinBox,
    QPlainTextEdit {{
        min-height: {INPUT_MIN_HEIGHT_PX}px;
    }}
    QGroupBox::title {{
        padding: 0 4px;
    }}
    """
    existing_stylesheet = window.styleSheet().strip()
    window.setStyleSheet(f'{existing_stylesheet}\n{scaling_stylesheet}' if existing_stylesheet else scaling_stylesheet)