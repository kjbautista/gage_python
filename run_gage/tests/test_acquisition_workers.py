import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_gage.acquisition_workers import SaveAcquisitionWorker, build_display_fft_mag
from run_gage.models import AcquisitionConfig, AcquisitionFrame, DisplayConfig


class AcquisitionWorkerDisplayTests(unittest.TestCase):
    def test_fft_uses_plot_offset_adjusted_waveform(self) -> None:
        frame = AcquisitionFrame(
            time_axis_us=np.array([0.0, 1.0, 2.0, 3.0]),
            volts_data=np.array([0.25, -0.5, 0.75, -1.0]),
            freq_axis_mhz=np.array([], dtype=float),
            fft_mag=np.array([], dtype=float),
            min_voltage=-1.0,
        )
        no_offset = DisplayConfig(
            displayed_voltage_range_v=1.0,
            plot_offset_v=0.0,
            freq_start_mhz=0.0,
            freq_end_mhz=10.0,
            lines_to_average=1,
        )
        large_offset = DisplayConfig(
            displayed_voltage_range_v=1.0,
            plot_offset_v=2.0,
            freq_start_mhz=0.0,
            freq_end_mhz=10.0,
            lines_to_average=1,
        )

        fft_without_offset = build_display_fft_mag(frame, no_offset)
        fft_with_offset = build_display_fft_mag(frame, large_offset)

        self.assertFalse(np.allclose(fft_without_offset, fft_with_offset))

    def test_fft_matches_offset_adjusted_input(self) -> None:
        frame = AcquisitionFrame(
            time_axis_us=np.array([0.0, 1.0, 2.0, 3.0]),
            volts_data=np.array([0.25, -0.5, 0.75, -1.0]),
            freq_axis_mhz=np.array([], dtype=float),
            fft_mag=np.array([], dtype=float),
            min_voltage=-1.0,
        )
        display_config = DisplayConfig(
            displayed_voltage_range_v=1.0,
            plot_offset_v=0.5,
            freq_start_mhz=0.0,
            freq_end_mhz=10.0,
            lines_to_average=1,
        )

        adjusted_waveform = display_config.apply_plot_offset(frame.volts_data)
        fft_linear = np.abs(np.fft.rfft(adjusted_waveform, n=4))
        expected_fft = 20 * np.log10(fft_linear / np.max(fft_linear))  # dB normalized

        np.testing.assert_allclose(build_display_fft_mag(frame, display_config), expected_fft)

    def test_aline_save_writes_json_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / 'aline_test.txt'
            worker = SaveAcquisitionWorker(
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
                    plot_offset_v=0.001,
                    freq_start_mhz=0.0,
                    freq_end_mhz=10.0,
                    lines_to_average=2,
                ),
                destination=destination,
            )
            frames = [
                AcquisitionFrame(
                    time_axis_us=np.array([0.0, 1.0]),
                    volts_data=np.array([1.0, 2.0]),
                    freq_axis_mhz=np.array([], dtype=float),
                    fft_mag=np.array([], dtype=float),
                    min_voltage=1.0,
                ),
                AcquisitionFrame(
                    time_axis_us=np.array([0.0, 1.0]),
                    volts_data=np.array([3.0, 4.0]),
                    freq_axis_mhz=np.array([], dtype=float),
                    fft_mag=np.array([], dtype=float),
                    min_voltage=3.0,
                ),
            ]

            worker._write_text_matrix(frames)

            data_path = destination.with_suffix('.npy')
            metadata_path = Path(tmp_dir) / 'aline_test_parameters.json'
            self.assertTrue(data_path.exists())
            self.assertTrue(metadata_path.exists())
            payload = json.loads(metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(payload['schema_version'], 2)
            self.assertEqual(payload['data_file'], 'aline_test.npy')
            self.assertEqual(payload['frame_count'], 2)
            self.assertEqual(payload['acquisition_config']['n_alines'], 2)
            self.assertEqual(payload['display_config']['lines_to_average'], 2)


if __name__ == '__main__':
    unittest.main()