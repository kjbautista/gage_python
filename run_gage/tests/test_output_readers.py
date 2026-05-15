import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from run_gage.acquisition_workers import (
    BeammapWorker,
    SaveAcquisitionWorker,
    build_metadata_payload,
    write_calibration_sidecar,
    write_json_sidecar,
)
from run_gage.models import AcquisitionConfig, AcquisitionFrame, DisplayConfig
from read_gage.python import read_aline_output, read_beammap_output, read_mmode_output
from read_gage.python.calibration_loader import load_calibration_data


def _build_acquisition_config(*, n_alines: int, data_type: str = 'aline') -> AcquisitionConfig:
    return AcquisitionConfig(
        t_start_us=0.0,
        t_end_us=10.0,
        sample_rate_hz=200e6,
        trigger_level_v=2.5,
        input_range_v=0.1,
        dc_offset_mv=0.0,
        n_alines=n_alines,
        data_type=data_type,
    )


def _build_display_config(*, plot_offset_v: float = 0.001, lines_to_average: int = 2) -> DisplayConfig:
    return DisplayConfig(
        displayed_voltage_range_v=0.1,
        plot_offset_v=plot_offset_v,
        freq_start_mhz=0.0,
        freq_end_mhz=10.0,
        lines_to_average=lines_to_average,
    )


class OutputReaderTests(unittest.TestCase):
    def test_read_aline_output_loads_sidecar_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / 'aline_test.txt'
            worker = SaveAcquisitionWorker(
                controller=None,
                acquisition_config=_build_acquisition_config(n_alines=2),
                display_config=_build_display_config(plot_offset_v=0.001, lines_to_average=2),
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

            result = read_aline_output(destination.with_suffix('.npy'))

            np.testing.assert_allclose(result['time_axis_us'], np.array([0.0, 1.0]))
            np.testing.assert_allclose(
                result['voltage_data_v'],
                np.array([[0.999, 2.999], [1.999, 3.999]], dtype=float),
            )
            self.assertEqual(result['metadata']['data_file'], 'aline_test.npy')
            self.assertEqual(result['metadata']['acquisition_config']['n_alines'], 2)

    def test_read_mmode_output_loads_sidecar_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / 'mmode_test.txt'
            data_path = destination.with_suffix('.npy')
            save_matrix = np.array(
                [
                    [0.0, 1.1, 2.1, 3.1],
                    [1.0, 1.2, 2.2, 3.2],
                    [2.0, 1.3, 2.3, 3.3],
                ],
                dtype=float,
            )
            np.save(data_path, save_matrix)
            write_json_sidecar(
                destination,
                build_metadata_payload(
                    _build_acquisition_config(n_alines=10, data_type='mmode'),
                    _build_display_config(plot_offset_v=0.002, lines_to_average=3),
                    extra={
                        'data_file': data_path.name,
                        'valid_alines': 3,
                        'next_write_index': 4,
                        'last_write_index': 3,
                    },
                ),
            )

            result = read_mmode_output(data_path)

            np.testing.assert_allclose(result['time_axis_us'], np.array([0.0, 1.0, 2.0]))
            np.testing.assert_allclose(
                result['voltage_data_v'],
                np.array([[1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3]], dtype=float),
            )
            self.assertEqual(result['metadata']['data_file'], 'mmode_test.npy')
            self.assertEqual(result['metadata']['valid_alines'], 3)

    def test_read_beammap_output_loads_scan_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            worker = BeammapWorker(
                controller=None,
                acquisition_config=_build_acquisition_config(n_alines=2, data_type='beammap'),
                display_config=_build_display_config(plot_offset_v=0.001, lines_to_average=2),
                xps_backend=None,
                xps=None,
                scan_positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
                save_dir=save_dir,
                home_position=np.array([7.0, 8.0, 9.0], dtype=float),
            )
            worker.output_timestamp = '20260402_120000'

            worker._write_metadata_outputs(time_axis_us=np.array([0.0, 0.5, 1.0], dtype=float))
            # pos1 rows=[1,2,3] cols=[4,5,6] -> mean per sample = [2.5, 3.5, 4.5]
            worker._save_position_lines(
                position_index=1,
                displayed_voltage_lines=np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=float),
            )
            # pos2 rows=[7,8,9] cols=[10,11,12] -> mean per sample = [8.5, 9.5, 10.5]
            worker._save_position_lines(
                position_index=2,
                displayed_voltage_lines=np.array([[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]], dtype=float),
            )

            result = read_beammap_output(save_dir)

            np.testing.assert_allclose(result['coordinates_mm'], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            self.assertIsNone(result['pressure_peak_pos'])
            self.assertIsNone(result['pressure_peak_neg'])
            # dim axes: unique([1,4]) -> center -> [-1.5, 1.5]; same for dim2, dim3
            np.testing.assert_allclose(result['dim1'], np.array([-1.5, 1.5]))
            np.testing.assert_allclose(result['dim2'], np.array([-1.5, 1.5]))
            np.testing.assert_allclose(result['dim3'], np.array([-1.5, 1.5]))
            # peak maps 2x2x2: pos0->[0,0,0], pos1->[1,1,1]
            expected_peak_pos = np.zeros((2, 2, 2))
            expected_peak_pos[0, 0, 0] = 4.5
            expected_peak_pos[1, 1, 1] = 10.5
            expected_peak_neg = np.zeros((2, 2, 2))
            expected_peak_neg[0, 0, 0] = 2.5
            expected_peak_neg[1, 1, 1] = 8.5
            np.testing.assert_allclose(result['voltage_peak_pos'], expected_peak_pos)
            np.testing.assert_allclose(result['voltage_peak_neg'], expected_peak_neg)
            self.assertEqual(result['save_dir'], save_dir)

    def test_read_beammap_output_uses_first_scan_in_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            (save_dir / 'acquisition_parameters_20260402_120000.json').write_text(
                '{"timestamp": "20260402_120000"}', encoding='utf-8',
            )
            (save_dir / 'acquisition_parameters_20260402_120500.json').write_text(
                '{"timestamp": "20260402_120500"}', encoding='utf-8',
            )
            np.save(save_dir / 'coordinates_mm_20260402_120000.npy', np.array([[1.0, 2.0, 3.0]], dtype=float))
            np.save(save_dir / 'time_axis_us_20260402_120000.npy', np.array([0.0, 0.5], dtype=float))
            np.save(save_dir / 'voltage_data_pos0001_20260402_120000.npy', np.array([[1.0], [2.0]], dtype=float))

            result = read_beammap_output(save_dir)

            # Single position, single row -> peak arrays are shape (1,1,1)
            np.testing.assert_allclose(result['voltage_peak_pos'], np.array([[[2.0]]]))
            np.testing.assert_allclose(result['voltage_peak_neg'], np.array([[[1.0]]]))
            np.testing.assert_allclose(result['dim1'], np.array([0.0]))


class CalibrationSidecarTests(unittest.TestCase):
    CAL_STEM = 'h1344_p2040'

    def test_write_calibration_sidecar_dumps_freq_and_sens_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sidecar_path = Path(tmp_dir) / 'aline_test.cal'
            written = write_calibration_sidecar(sidecar_path, self.CAL_STEM)
            self.assertEqual(written, sidecar_path)
            self.assertTrue(sidecar_path.exists())
            # np.save would normally auto-append .npy; confirm the .cal path is the actual file.
            self.assertFalse(sidecar_path.with_suffix('.cal.npy').exists())
            saved = np.load(sidecar_path)
            expected = load_calibration_data(self.CAL_STEM)
            self.assertEqual(saved.shape, expected.shape)
            self.assertEqual(saved.shape[1], 2)
            np.testing.assert_allclose(saved, expected)

    def test_write_calibration_sidecar_skips_when_no_calibration_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sidecar_path = Path(tmp_dir) / 'aline_test.cal'
            self.assertIsNone(write_calibration_sidecar(sidecar_path, ''))
            self.assertFalse(sidecar_path.exists())

    def test_read_aline_output_uses_sidecar_for_pressure_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / 'aline_test.npy'
            sample_rate_hz = 200e6
            t = np.arange(256) / sample_rate_hz
            wave = np.sin(2 * np.pi * 1e6 * t).astype(float)
            save_matrix = np.column_stack([t * 1e6, wave])
            np.save(destination, save_matrix)
            write_json_sidecar(
                destination,
                build_metadata_payload(
                    AcquisitionConfig(
                        t_start_us=0.0, t_end_us=10.0, sample_rate_hz=sample_rate_hz,
                        trigger_level_v=2.5, input_range_v=0.1, dc_offset_mv=0.0,
                        n_alines=1, data_type='aline',
                        calibration_file=self.CAL_STEM,
                    ),
                    DisplayConfig(
                        displayed_voltage_range_v=0.1, plot_offset_v=0.0,
                        freq_start_mhz=0.0, freq_end_mhz=10.0, lines_to_average=1,
                    ),
                    extra={'data_file': destination.name, 'frame_count': 1},
                ),
            )
            write_calibration_sidecar(
                destination.with_suffix('.cal'),
                self.CAL_STEM,
            )

            result = read_aline_output(destination)

            self.assertIsNotNone(result['pressure_data_pa'])
            self.assertEqual(result['pressure_data_pa'].shape, result['voltage_data_v'].shape)

    def test_read_aline_output_warns_when_sidecar_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / 'aline_test.npy'
            np.save(destination, np.array([[0.0, 0.1], [1.0, 0.2]], dtype=float))
            write_json_sidecar(
                destination,
                build_metadata_payload(
                    _build_acquisition_config(n_alines=1),
                    _build_display_config(),
                    extra={'data_file': destination.name, 'frame_count': 1},
                ),
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = read_aline_output(destination)

            self.assertIsNone(result['pressure_data_pa'])
            self.assertTrue(any('Calibration sidecar not found' in str(w.message) for w in caught))
            self.assertTrue(any('aline_test.cal' in str(w.message) for w in caught))


if __name__ == '__main__':
    unittest.main()
