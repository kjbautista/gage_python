"""Verify SaveAcquisitionWorker refuses to save frames whose time axes disagree (P0-1)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_gage.acquisition_workers import SaveAcquisitionWorker
from run_gage.models import (
    AcquisitionConfig,
    AcquisitionFrame,
    AcquisitionInconsistencyError,
    DisplayConfig,
)


def _build_worker(destination: Path) -> SaveAcquisitionWorker:
    return SaveAcquisitionWorker(
        controller=None,
        acquisition_config=AcquisitionConfig(
            t_start_us=0.0,
            t_end_us=10.0,
            sample_rate_hz=200e6,
            trigger_level_v=2.5,
            input_range_v=0.1,
            dc_offset_mv=0.0,
            n_alines=2,
        ),
        display_config=DisplayConfig(
            displayed_voltage_range_v=0.1,
            plot_offset_v=0.0,
            freq_start_mhz=0.0,
            freq_end_mhz=10.0,
            lines_to_average=2,
        ),
        destination=destination,
    )


def _frame(time_axis_us: np.ndarray, volts_data: np.ndarray) -> AcquisitionFrame:
    return AcquisitionFrame(
        time_axis_us=time_axis_us,
        volts_data=volts_data,
        freq_axis_mhz=np.array([], dtype=float),
        fft_mag=np.array([], dtype=float),
        min_voltage=float(np.min(volts_data)),
    )


class SaveWorkerTimeAxisConsistencyTests(unittest.TestCase):
    def test_mismatched_time_axis_raises_inconsistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = _build_worker(Path(tmp_dir) / 'aline.txt')
            frames = [
                _frame(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
                _frame(np.array([0.0, 1.0, 5.0]), np.array([4.0, 5.0, 6.0])),
            ]
            with self.assertRaises(AcquisitionInconsistencyError):
                worker._write_text_matrix(frames)

    def test_mismatched_length_raises_inconsistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = _build_worker(Path(tmp_dir) / 'aline.txt')
            frames = [
                _frame(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
                _frame(np.array([0.0, 1.0]), np.array([4.0, 5.0])),
            ]
            with self.assertRaises(AcquisitionInconsistencyError):
                worker._write_text_matrix(frames)

    def test_matching_time_axes_save_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / 'aline.txt'
            worker = _build_worker(destination)
            frames = [
                _frame(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
                _frame(np.array([0.0, 1.0, 2.0]), np.array([4.0, 5.0, 6.0])),
            ]
            worker._write_text_matrix(frames)
            data_path = destination.with_suffix('.npy')
            self.assertTrue(data_path.exists())


if __name__ == '__main__':
    unittest.main()
