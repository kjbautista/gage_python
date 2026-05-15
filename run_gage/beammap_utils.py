"""Beammap scan-position generator, MATLAB-compatible.

The output order matches the original MATLAB implementation so saved scans can
be read by both Python and MATLAB tools without re-sorting. Two behaviours
surprise new readers:

- ``reshape(..., order='F')`` is used so consecutive positions differ only in
  the fastest-varying axis, matching MATLAB's column-major memory layout.
- Alternate rows/columns are reversed to produce a *snake* path — minimising
  large backlash moves. This only runs when ≥2 axes move; a single-axis scan
  stays monotonic.

Worked example (axes 1 and 2 move, axis 3 fixed)::

    center_pos = (0, 0, 0)   step_size = (1, 1, 0)   scan_range = (2, 2, 0)
    limits     = [[-10,10], [-10,10], [-10,10]]

    group1 = [-1, 0, 1]   group2 = [-1, 0, 1]   group3 = [0]

    Raw grid (Fortran order, group1 varies fastest):
        (-1,-1, 0), ( 0,-1, 0), ( 1,-1, 0),
        (-1, 0, 0), ( 0, 0, 0), ( 1, 0, 0),
        (-1, 1, 0), ( 0, 1, 0), ( 1, 1, 0)

    After snake reversal on every second group1 column:
        (-1,-1, 0), ( 0,-1, 0), ( 1,-1, 0),     ← row 0: left→right
        ( 1, 0, 0), ( 0, 0, 0), (-1, 0, 0),     ← row 1: right→left (snake)
        (-1, 1, 0), ( 0, 1, 0), ( 1, 1, 0)      ← row 2: left→right
"""

from __future__ import annotations

import numpy as np


def calculate_beammap_scan_positions(
    center_pos: np.ndarray,
    step_size: np.ndarray,
    scan_range: np.ndarray,
    limits: np.ndarray,
) -> np.ndarray:
    """Return absolute beammap scan positions using the MATLAB snake-pattern logic.

    Args:
        center_pos: 3-vector — centre of the scan in absolute mm for each axis.
        step_size:  3-vector — spacing along each axis; ``0`` disables that axis.
        scan_range: 3-vector — full width around the centre for each axis.
        limits:     ``(3, 2)`` — absolute min/max per axis; positions outside
                    these bounds are dropped.

    Returns:
        ``(N, 3)`` array of absolute stage positions (mm) in execution order.
        Disabled axes hold ``center_pos`` for every returned row. Snaking is
        applied only when two or more axes move; a 1-axis scan remains
        monotonic because no backlash savings apply.
    """

    center_pos = np.asarray(center_pos, dtype=float).reshape(3)
    step_size = np.asarray(step_size, dtype=float).reshape(3)
    scan_range = np.asarray(scan_range, dtype=float).reshape(3)
    limits = np.asarray(limits, dtype=float).reshape(3, 2)

    scan_range = scan_range.copy()
    step_size = step_size.copy()
    scan_range[step_size == 0] = 0
    step_size[step_size == 0] = 1
    groups_to_move = scan_range != 0

    def build_axis(center: float, step: float, width: float, axis_limits: np.ndarray) -> np.ndarray:
        values = np.arange(center - width / 2.0, center + width / 2.0 + step * 0.5, step, dtype=float)
        if values.size == 0:
            values = np.array([center], dtype=float)
        values = values - np.mean(values) + center
        mask = (values > axis_limits[0]) & (values < axis_limits[1])
        values = values[mask]
        return values if values.size else np.array([center], dtype=float)

    group1 = build_axis(center_pos[0], step_size[0], scan_range[0], limits[0])
    group2 = build_axis(center_pos[1], step_size[1], scan_range[1], limits[1])
    group3 = build_axis(center_pos[2], step_size[2], scan_range[2], limits[2])

    grid1, grid3, grid2 = np.meshgrid(group1, group3, group2, indexing='ij')

    def reshape_matlab(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(array, shape, order='F')

    scan_positions = np.column_stack((grid1.ravel(order='F'), grid2.ravel(order='F'), grid3.ravel(order='F')))

    if int(np.sum(groups_to_move)) > 1:
        if groups_to_move[0]:
            scan_positions = reshape_matlab(scan_positions, (len(group1), -1, 3))
            for column in range(1, scan_positions.shape[1], 2):
                scan_positions[:, column, 0] = scan_positions[::-1, column, 0]
            scan_positions = reshape_matlab(scan_positions, (-1, 3))
            if groups_to_move[2] and groups_to_move[1]:
                scan_positions = reshape_matlab(scan_positions, (len(group3) * len(group1), -1, 3))
                for column in range(1, scan_positions.shape[1], 2):
                    scan_positions[:, column, 2] = scan_positions[::-1, column, 2]
                scan_positions = reshape_matlab(scan_positions, (-1, 3))
        elif groups_to_move[2]:
            scan_positions = reshape_matlab(scan_positions, (len(group3), -1, 3))
            for column in range(1, scan_positions.shape[1], 2):
                scan_positions[:, column, 2] = scan_positions[::-1, column, 2]
            scan_positions = reshape_matlab(scan_positions, (-1, 3))

    return scan_positions