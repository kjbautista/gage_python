import unittest

import numpy as np

from run_gage.beammap_utils import calculate_beammap_scan_positions


class BeammapScanPositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = np.array([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]])

    def test_group1_group3_snake_matches_validated_order(self) -> None:
        positions = calculate_beammap_scan_positions(
            center_pos=np.array([0.0, 0.0, 0.0]),
            step_size=np.array([1.0, 0.0, 1.0]),
            scan_range=np.array([2.0, 0.0, 2.0]),
            limits=self.limits,
        )

        expected = np.array(
            [
                [-1.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(positions, expected)

    def test_all_three_groups_follow_nested_snake_pattern(self) -> None:
        positions = calculate_beammap_scan_positions(
            center_pos=np.array([0.0, 0.0, 0.0]),
            step_size=np.array([1.0, 1.0, 1.0]),
            scan_range=np.array([1.0, 1.0, 1.0]),
            limits=self.limits,
        )

        expected = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ]
        )

        np.testing.assert_allclose(positions, expected)


if __name__ == '__main__':
    unittest.main()