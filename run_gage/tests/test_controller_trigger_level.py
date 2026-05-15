"""Validate the trigger-level conversion math exposed by controller constants (P0-3)."""

from __future__ import annotations

import unittest

import numpy as np

from run_gage.controller import TRIGGER_LEVEL_FULL_SCALE_V, TRIGGER_LEVEL_PERCENT_MAX


def _volts_to_percent(volts: float) -> int:
    return int(np.floor(volts / TRIGGER_LEVEL_FULL_SCALE_V * TRIGGER_LEVEL_PERCENT_MAX))


class TriggerLevelConversionTests(unittest.TestCase):
    def test_zero_volts_maps_to_zero_percent(self) -> None:
        self.assertEqual(_volts_to_percent(0.0), 0)

    def test_full_scale_maps_to_hundred_percent(self) -> None:
        self.assertEqual(_volts_to_percent(TRIGGER_LEVEL_FULL_SCALE_V), TRIGGER_LEVEL_PERCENT_MAX)

    def test_half_scale_maps_to_fifty_percent(self) -> None:
        self.assertEqual(_volts_to_percent(TRIGGER_LEVEL_FULL_SCALE_V / 2.0), TRIGGER_LEVEL_PERCENT_MAX // 2)

    def test_intermediate_floor_behaviour(self) -> None:
        # 2.4 V → floor(48) = 48
        self.assertEqual(_volts_to_percent(2.4), 48)
        # 2.499 V → floor(49.98) = 49
        self.assertEqual(_volts_to_percent(2.499), 49)


if __name__ == '__main__':
    unittest.main()
