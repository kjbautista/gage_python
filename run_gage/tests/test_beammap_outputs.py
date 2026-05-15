import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from run_gage.acquisition_workers import BeammapWorker
from run_gage.models import AcquisitionConfig, DisplayConfig


class BeammapOutputTests(unittest.TestCase):
    def test_beammap_outputs_use_shared_metadata_and_per_position_voltage_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir)
            worker = BeammapWorker(
                controller=None,
                acquisition_config=AcquisitionConfig(
                    t_start_us=0.0,
                    t_end_us=10.0,
                    sample_rate_hz=200e6,
                    trigger_level_v=2.5,
                    input_range_v=0.1,
                    dc_offset_mv=0.0,
                    n_alines=2,
                    data_type='beammap',
                ),
                display_config=DisplayConfig(
                    displayed_voltage_range_v=0.1,
                    plot_offset_v=0.001,
                    freq_start_mhz=0.0,
                    freq_end_mhz=10.0,
                    lines_to_average=2,
                ),
                xps_backend=None,
                xps=None,
                scan_positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
                save_dir=save_dir,
                home_position=np.array([7.0, 8.0, 9.0], dtype=float),
            )
            worker.output_timestamp = '20260330_120000'

            worker._write_metadata_outputs(time_axis_us=np.array([0.0, 0.5, 1.0], dtype=float))
            worker._save_position_lines(
                position_index=1,
                displayed_voltage_lines=np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=float),
            )
            worker._save_position_lines(
                position_index=2,
                displayed_voltage_lines=np.array([[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]], dtype=float),
            )

            parameters_path = save_dir / 'acquisition_parameters_20260330_120000.json'
            coordinates_path = save_dir / 'coordinates_mm_20260330_120000.npy'
            time_axis_path = save_dir / 'time_axis_us_20260330_120000.npy'
            pos1_path = save_dir / 'voltage_data_pos0001_20260330_120000.npy'
            pos2_path = save_dir / 'voltage_data_pos0002_20260330_120000.npy'

            self.assertTrue(parameters_path.exists())
            self.assertTrue(coordinates_path.exists())
            self.assertTrue(time_axis_path.exists())
            self.assertTrue(pos1_path.exists())
            self.assertTrue(pos2_path.exists())

            self.assertFalse((save_dir / 'beammap_values.txt').exists())
            self.assertFalse((save_dir / 'voltage_data_V.txt').exists())

            np.testing.assert_allclose(
                np.load(coordinates_path),
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
            )
            np.testing.assert_allclose(
                np.load(time_axis_path),
                np.array([0.0, 0.5, 1.0], dtype=float),
            )
            np.testing.assert_allclose(
                np.load(pos1_path),
                np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=float),
            )
            np.testing.assert_allclose(
                np.load(pos2_path),
                np.array([[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]], dtype=float),
            )
            parameters_payload = json.loads(parameters_path.read_text(encoding='utf-8'))
            self.assertEqual(parameters_payload['schema_version'], 1)
            self.assertEqual(parameters_payload['timestamp'], '20260330_120000')
            self.assertEqual(parameters_payload['home_position_mm'], [7.0, 8.0, 9.0])
            self.assertEqual(parameters_payload['acquisition_config']['data_type'], 'beammap')
            self.assertEqual(parameters_payload['display_config']['lines_to_average'], 2)


if __name__ == '__main__':
    unittest.main()