import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication

from run_gage.gui.app_beammap import resolve_beammap_axes
from run_gage.gui.plot_widgets import BeammapPlotCanvas


class BeammapAxisResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.group1 = np.array([-1.0, 0.0, 1.0])
        self.group2 = np.array([10.0, 20.0])
        self.group3 = np.array([5.0, 6.0, 7.0])

    def test_group1_group3_axes(self) -> None:
        dim1, dim2, x_label, y_label, reverse_x = resolve_beammap_axes(
            np.array([True, False, True]),
            self.group1,
            self.group2,
            self.group3,
        )

        np.testing.assert_allclose(dim1, self.group1)
        np.testing.assert_allclose(dim2, self.group3)
        self.assertEqual(x_label, 'Group3 (mm)')
        self.assertEqual(y_label, 'Group1 (mm)')
        self.assertTrue(reverse_x)

    def test_group1_group2_axes(self) -> None:
        dim1, dim2, x_label, y_label, reverse_x = resolve_beammap_axes(
            np.array([True, True, False]),
            self.group1,
            self.group2,
            self.group3,
        )

        np.testing.assert_allclose(dim1, self.group1)
        np.testing.assert_allclose(dim2, self.group2)
        self.assertEqual(x_label, 'Group2 (mm)')
        self.assertEqual(y_label, 'Group1 (mm)')
        self.assertFalse(reverse_x)

    def test_group2_group3_axes(self) -> None:
        dim1, dim2, x_label, y_label, reverse_x = resolve_beammap_axes(
            np.array([False, True, True]),
            self.group1,
            self.group2,
            self.group3,
        )

        np.testing.assert_allclose(dim1, self.group2)
        np.testing.assert_allclose(dim2, self.group3)
        self.assertEqual(x_label, 'Group3 (mm)')
        self.assertEqual(y_label, 'Group2 (mm)')
        self.assertTrue(reverse_x)

    def test_single_axis_group3_reverses_x(self) -> None:
        dim1, dim2, x_label, y_label, reverse_x = resolve_beammap_axes(
            np.array([False, False, True]),
            self.group1,
            self.group2,
            self.group3,
        )

        np.testing.assert_allclose(dim1, np.array([0.0]))
        np.testing.assert_allclose(dim2, self.group3)
        self.assertEqual(x_label, 'Group3 (mm)')
        self.assertEqual(y_label, '')
        self.assertTrue(reverse_x)


class BeammapCanvasMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_update_map_keeps_dim1_on_y_and_dim2_on_x(self) -> None:
        canvas = BeammapPlotCanvas()
        beammap = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dim1 = np.array([100.0, 200.0])
        dim2 = np.array([10.0, 20.0, 30.0])

        canvas.update_map(beammap, dim1, dim2, 'Group3 (mm)', 'Group1 (mm)', reverse_x=True)

        np.testing.assert_allclose(canvas.image.get_array(), beammap)
        self.assertEqual(canvas.axes.get_xlabel(), 'Group3 (mm)')
        self.assertEqual(canvas.axes.get_ylabel(), 'Group1 (mm)')
        self.assertEqual(tuple(canvas.image.get_extent()), (-10.0, 10.0, -50.0, 50.0))
        self.assertEqual(tuple(canvas.axes.get_xlim()), (10.0, -10.0))
        self.assertEqual(tuple(canvas.axes.get_ylim()), (50.0, -50.0))


if __name__ == '__main__':
    unittest.main()