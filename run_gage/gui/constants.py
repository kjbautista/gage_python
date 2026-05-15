from __future__ import annotations

from enum import Enum

RECONFIGURE_DEBOUNCE_MS = 250


class VoltageRange(float, Enum):
    """Digitizer input half-range options exposed by the GUI combo boxes.

    Values below 0.1 V are kept for legacy pickers but are rejected by
    ``AcquisitionConfig.validate()`` when committed — see
    ``SUPPORTED_INPUT_RANGES_V`` in ``run_gage.models``.
    """

    V_0_025 = 0.025
    V_0_05 = 0.05
    V_0_1 = 0.1
    V_0_2 = 0.2
    V_0_5 = 0.5
    V_1 = 1.0
    V_2 = 2.0
    V_5 = 5.0

    @property
    def display_label(self) -> str:
        """Return the label shown in GUI combo boxes (trailing zeros trimmed)."""

        return ('%g' % float(self.value))


VOLTAGE_RANGE_OPTIONS: tuple[str, ...] = tuple(r.display_label for r in VoltageRange)

# Motion stage group names in axis order (Group1=axis 0, Group2=axis 1, Group3=axis 2).
# All three apps and workers iterate this tuple when they need to address every axis.
MOTION_GROUPS = ('Group1', 'Group2', 'Group3')
