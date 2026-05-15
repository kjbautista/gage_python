from __future__ import annotations

import numpy as np
from matplotlib import style
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy

from run_gage.models import DisplayConfig


style.use('seaborn-v0_8-whitegrid')  

PLOT_FONT_SIZE_PT = 12
PLOT_TITLE_FONT_SIZE_PT = 12


class BasePlotCanvas(FigureCanvasQTAgg):
    """Shared matplotlib canvas setup for Qt-embedded plots."""

    def __init__(self, xlabel: str, ylabel: str, title: str = '') -> None:
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel(xlabel, fontsize=PLOT_FONT_SIZE_PT)
        self.axes.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE_PT)
        self.axes.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE_PT)
        self.axes.tick_params(axis='both', labelsize=PLOT_FONT_SIZE_PT)
        self.axes.grid(True)
        self._apply_figure_margins()
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _apply_figure_margins(self) -> None:
        self.figure.subplots_adjust(left=0.13, right=0.97, bottom=0.25, top=0.97)


class AlinePlotCanvas(BasePlotCanvas):
    """Plot canvas dedicated to the time-domain A-line view."""

    def __init__(self) -> None:
        super().__init__('Time (us)', 'Voltage (V)', '')
        (self.trace_line,) = self.axes.plot([], [])
        self.min_line = self.axes.axhline(0.0, color='red', linestyle='--', linewidth=1)

    def update_frame(self, time_axis_us, volts_data, min_voltage: float, display_config: DisplayConfig) -> None:
        """Update the plotted A-line trace without rebuilding the entire axes."""

        plotted_volts = display_config.apply_plot_offset(volts_data)
        plotted_min_voltage = float(display_config.apply_plot_offset([min_voltage])[0])
        self.trace_line.set_data(time_axis_us, plotted_volts)
        self.min_line.set_ydata([plotted_min_voltage, plotted_min_voltage])
        self.axes.set_xlim(time_axis_us[0], time_axis_us[-1])
        self.axes.set_ylim(
            -display_config.displayed_voltage_range_v + display_config.plot_offset_v,
            display_config.displayed_voltage_range_v + display_config.plot_offset_v,
        )
        self.axes.relim()
        self.axes.autoscale_view(scalex=False, scaley=False)
        self._apply_figure_margins()
        self.draw_idle()


class FftPlotCanvas(BasePlotCanvas):
    """Plot canvas dedicated to the frequency-domain spectrum view."""

    def __init__(self) -> None:
        super().__init__('Frequency (MHz)', 'Magnitude (dB)', '')
        (self.trace_line,) = self.axes.plot([], [])
        self._last_xlim: tuple[float, float] | None = None

    def update_spectrum(self, freq_axis_mhz, fft_mag, display_config: DisplayConfig) -> None:
        """Update the FFT trace and enforce the selected frequency display window.

        X and Y limits are computed from the filtered slice and applied together
        before ``draw_idle()`` to avoid a first-frame autoscale flicker where Y
        would briefly reflect the unfiltered full spectrum.
        """

        freq_start, freq_end = display_config.normalized_frequency_limits()
        freq_axis_mhz = np.asarray(freq_axis_mhz)
        fft_mag = np.asarray(fft_mag)
        visible_mask = (freq_axis_mhz >= freq_start) & (freq_axis_mhz <= freq_end)

        if visible_mask.any():
            visible_fft_mag = fft_mag[visible_mask]
            y_min = float(visible_fft_mag.min())
            y_max = float(visible_fft_mag.max())
            if y_min == y_max:
                padding = max(abs(y_min) * 0.05, 1.0)
            else:
                padding = (y_max - y_min) * 0.05
            ylim = (y_min - padding, y_max + padding)
        else:
            ylim = None

        self.trace_line.set_data(freq_axis_mhz, fft_mag)
        xlim = (freq_start, freq_end)
        if xlim != self._last_xlim:
            self.axes.set_xlim(*xlim)
            self._last_xlim = xlim
        if ylim is not None:
            self.axes.set_ylim(*ylim)
        else:
            self.axes.relim()
            self.axes.autoscale_view(scalex=False, scaley=True)
        self._apply_figure_margins()
        self.draw_idle()


class MmodePlotCanvas(BasePlotCanvas):
    """Plot canvas dedicated to the rolling M-mode image view."""

    def __init__(self) -> None:
        super().__init__('Buffer index', 'Time (us)', '')
        self.axes.grid(False)
        self.image = self.axes.imshow(
            np.zeros((2, 2), dtype=float),
            aspect='auto',
            cmap='gray',
            origin='lower',
            interpolation='nearest',
        )
        self.position_line = self.axes.axvline(0.0, color='#ff6b6b', linestyle='--', linewidth=1.5)

    def update_image(self, mmode_data: np.ndarray, time_axis_us: np.ndarray, buffer_size: int, write_index: int) -> None:
        """Update the M-mode image using the provided ring-buffer data."""

        if mmode_data.size == 0 or time_axis_us.size == 0:
            return

        self.image.set_data(mmode_data)
        self.image.set_extent((0, buffer_size, float(time_axis_us[0]), float(time_axis_us[-1])))
        self.axes.set_xlim(0, buffer_size)
        self.axes.set_ylim(float(time_axis_us[0]), float(time_axis_us[-1]))
        self.image.set_clim(float(np.min(mmode_data)), float(np.max(mmode_data)))
        self.position_line.set_xdata([float(write_index), float(write_index)])
        self._apply_figure_margins()
        self.draw_idle()


class BeammapPlotCanvas(BasePlotCanvas):
    """Plot canvas dedicated to 2D beammap intensity images."""

    def _apply_figure_margins(self) -> None:
        self.figure.subplots_adjust(left=0.10, right=0.85, bottom=0.10, top=0.97)

    def __init__(self) -> None:
        super().__init__('Dimension 2 (mm)', 'Dimension 1 (mm)', '')
        self.axes.grid(False)
        self.image = self.axes.imshow(
            np.zeros((2, 2), dtype=float),
            aspect='equal',
            cmap='viridis',
            origin='lower',
            interpolation='nearest',
        )
        self.colorbar = self.figure.colorbar(self.image, ax=self.axes)
        self.colorbar.set_label('Min voltage magnitude (V)', fontsize=PLOT_FONT_SIZE_PT)
        self.colorbar.ax.tick_params(labelsize=PLOT_FONT_SIZE_PT)

    def update_map(
        self,
        beammap_data: np.ndarray,
        dim1: np.ndarray,
        dim2: np.ndarray,
        x_label: str,
        y_label: str,
        reverse_x: bool,
    ) -> None:
        """Update the beammap image and axis metadata."""

        if beammap_data.size == 0 or dim1.size == 0 or dim2.size == 0:
            return

        x_center = float((np.min(dim2) + np.max(dim2)) / 2.0)
        y_center = float((np.min(dim1) + np.max(dim1)) / 2.0)
        x_min, x_max = float(np.min(dim2) - x_center), float(np.max(dim2) - x_center)
        y_min, y_max = float(np.min(dim1) - y_center), float(np.max(dim1) - y_center)
        self.image.set_data(beammap_data)
        self.image.set_extent((x_min, x_max, y_min, y_max))
        self.image.set_clim(float(np.min(beammap_data)), float(np.max(beammap_data)))
        self.axes.set_xlabel(x_label, fontsize=PLOT_FONT_SIZE_PT)
        self.axes.set_ylabel(y_label, fontsize=PLOT_FONT_SIZE_PT)
        self.axes.set_xlim(x_max, x_min) if reverse_x else self.axes.set_xlim(x_min, x_max)
        self.axes.set_ylim(y_max, y_min)
        self._apply_figure_margins()
        self.draw_idle()

    def update_axis_labels(self, x_label: str, y_label: str) -> None:
        """Update axis labels without redrawing image data."""
        self.axes.set_xlabel(x_label, fontsize=PLOT_FONT_SIZE_PT)
        self.axes.set_ylabel(y_label, fontsize=PLOT_FONT_SIZE_PT)
        self.draw_idle()


class SpatialPeakPlotCanvas(BasePlotCanvas):
    """Scatter-plot canvas for live spatial peak search feedback.

    Supports 1D (position vs. metric), 2D, and 3D (projected to 2D) searches.
    Coarse-phase probes are drawn as circles, refinement probes as diamonds.
    The current best position is marked with a gold star (2D/3D) or vertical
    dashed line (1D).
    """

    def __init__(self) -> None:
        self._is_1d: bool = False
        super().__init__('', '', '')
        self.axes.grid(True)
        self._best_pos: tuple[float, float] = (float('nan'), float('nan'))
        self._coarse_pts: list[tuple[float, float, float]] = []
        self._refine_pts: list[tuple[float, float, float]] = []
        self._colorbar = None

    def _apply_figure_margins(self) -> None:
        right = 0.97 if self._is_1d else 0.85
        self.figure.subplots_adjust(left=0.13, right=right, bottom=0.25, top=0.97)

    def reset(self, x_label: str, y_label: str, is_1d: bool) -> None:
        self._is_1d = is_1d
        self._best_pos = (float('nan'), float('nan'))
        self._coarse_pts.clear()
        self._refine_pts.clear()
        self._redraw(x_label, y_label)

    def add_point(self, x: float, y: float, value: float, phase: str) -> None:
        if phase == 'coarse':
            self._coarse_pts.append((x, y, value))
        else:
            self._refine_pts.append((x, y, value))
        self._redraw(self.axes.get_xlabel(), self.axes.get_ylabel())

    def update_best(self, x: float, y: float) -> None:
        self._best_pos = (x, y)
        self._redraw(self.axes.get_xlabel(), self.axes.get_ylabel())

    def _redraw(self, x_label: str, y_label: str) -> None:
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        self.axes.cla()
        self.axes.set_xlabel(x_label, fontsize=PLOT_FONT_SIZE_PT)
        self.axes.set_ylabel(y_label, fontsize=PLOT_FONT_SIZE_PT)
        self.axes.tick_params(axis='both', labelsize=PLOT_FONT_SIZE_PT)
        self.axes.grid(True)

        all_pts = self._coarse_pts + self._refine_pts
        if not all_pts:
            self._apply_figure_margins()
            self.draw_idle()
            return

        all_vals = [v for _, _, v in all_pts]
        vmin, vmax = min(all_vals), max(all_vals)
        if vmin == vmax:
            vmin -= 1e-9
            vmax += 1e-9

        if self._is_1d:
            if self._coarse_pts:
                self.axes.scatter(
                    [p[0] for p in self._coarse_pts],
                    [p[2] for p in self._coarse_pts],
                    color='steelblue', s=30, label='Coarse', zorder=3,
                )
            if self._refine_pts:
                self.axes.scatter(
                    [p[0] for p in self._refine_pts],
                    [p[2] for p in self._refine_pts],
                    color='coral', s=30, label='Refine', zorder=4,
                )
            if not np.isnan(self._best_pos[0]):
                self.axes.axvline(
                    self._best_pos[0], color='gold',
                    linestyle='--', linewidth=1.5, label='Best', zorder=5,
                )
            self.axes.legend(fontsize=PLOT_FONT_SIZE_PT - 2)
        else:
            sc = None
            if self._coarse_pts:
                sc = self.axes.scatter(
                    [p[0] for p in self._coarse_pts],
                    [p[1] for p in self._coarse_pts],
                    c=[p[2] for p in self._coarse_pts],
                    cmap='viridis', vmin=vmin, vmax=vmax,
                    s=40, marker='o', alpha=0.7, zorder=3,
                )
            if self._refine_pts:
                sc = self.axes.scatter(
                    [p[0] for p in self._refine_pts],
                    [p[1] for p in self._refine_pts],
                    c=[p[2] for p in self._refine_pts],
                    cmap='viridis', vmin=vmin, vmax=vmax,
                    s=50, marker='D', alpha=0.9, zorder=4,
                )
            if sc is not None:
                self._colorbar = self.figure.colorbar(sc, ax=self.axes)
                self._colorbar.set_label('Metric (V)', fontsize=PLOT_FONT_SIZE_PT)
                self._colorbar.ax.tick_params(labelsize=PLOT_FONT_SIZE_PT)
            if not np.isnan(self._best_pos[0]):
                self.axes.plot(
                    self._best_pos[0], self._best_pos[1],
                    '*', color='gold', markersize=14, zorder=5, label='Best',
                )
                self.axes.legend(fontsize=PLOT_FONT_SIZE_PT - 2)

        self._apply_figure_margins()
        self.draw_idle()
