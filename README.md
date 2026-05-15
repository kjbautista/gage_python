# gage_python

Python GUI and data-reading pipeline for the Gage CompuScope digitizer with optional Newport XPS motion-stage control for acoustic measurements using a hydrophone.

---

## Repository layout

```
examples/                   User-facing example scripts
  example_read_gage_python.ipynb  Python notebook: read and plot saved data
  example_read_gage.m             MATLAB example: read and plot saved data
read_gage/                  Everything needed to read saved data files
  calibrations/             Onda .txt calibration files + calibration_lookup.csv catalogue
  matlab/
    read_gage_output.m      Single MATLAB entry point for all data types
  python/
    calibration_loader.py   Hydrophone calibration catalogue and loader
    output_readers.py       Python functions for reading saved data files
run_gage/                   Core acquisition package
  models.py                 Shared dataclasses and exceptions
  controller.py             Low-level Gage SDK wrapper
  acquisition_workers.py    Background acquisition and save workers (Qt)
  beammap_utils.py          Snake-pattern scan position calculator
  xps_control_main.py       Newport XPS back-end functions (connect, status, move …)
  from_gage/                Gage SDK runtime files (GageConstants, PyGage3_64.pyd …)
  newportxps/               Newport XPS Python driver package (NewportXPS class …)
  gui/
    app_aline.py            A-line acquisition GUI
    app_mmode.py            M-mode acquisition GUI
    app_beammap.py          Beammap acquisition GUI
    app_find_spatial_peak.py          Spatial peak finder GUI (Nelder-Mead simplex)
    app_find_spatial_peak_brent.py    Spatial peak finder GUI (Brent 1-D method)
    app_motion_control.py   Newport XPS motion stage GUI (standalone or embedded)
    plot_widgets.py         Reusable matplotlib-in-Qt canvas widgets
    constants.py            GUI-wide option lists and timing constants
    style_utils.py          Font and widget-size stylesheet helper
  tests/                    Automated tests (pytest)
```

---

## Setup

### Hardware Requirements

- Gage PCIe Digitizer (software was developed using model # CSE1222)
- Newport Motion Controller (software was developed using model # XPS-Q8)

### Software Requirements

- Python 3.11 or later
- Windows (required for the Gage SDK `.pyd` extension)

### Install

```bash
# 1. Create and activate a virtual environment
py -3.11 -m venv gage_py_env
.\gage_py_env\Scripts\activate          # Windows PowerShell

# 2. Install the package and all dependencies from pyproject.toml
pip install -e .
```

The `-e` (editable) flag lets you edit source files without reinstalling.

### Run the tests

```bash
python -m pytest run_gage/tests/
```

---

## Running the GUIs

Each GUI can be launched as a module from the repo root. A connected Gage
digitizer is required at startup.

```bash
# A-line: single triggered waveform with live FFT display and file saving
python -m launch_aline

# M-mode: time-scrolling 2-D display saved as a single matrix file
python -m launch_mmode

# Beammap: 2-D spatial scan with motion-stage control and live preview
python -m launch_beammap

# Spatial peak finder — Nelder-Mead simplex optimisation
python -m launch_find_spatial_peak

# Spatial peak finder — Brent 1-D method per axis
python -m launch_find_spatial_peak_brent

# Motion stage only (Newport XPS) — also opens automatically from the above GUIs
python -m launch_motion_control
```

### Calibration / pressure conversion

Select the hydrophone, pre-amplifier, and attenuator from the dropdowns in any
GUI.  Entry names come from `read_gage/calibrations/calibration_lookup.csv`.  When a
matching Onda `.txt` calibration file is found, a pressure trace is computed
and displayed alongside the voltage trace.

---

## Hardware configuration

XPS connection details, motion-verification windows, and scan velocities are
read once per process from (in descending priority):

1. Environment variables (table below).
2. `run_gage/config.local.json` — gitignored per-deployment overrides.
3. Defaults baked into [`run_gage/config.py`](run_gage/config.py).

Do not edit `xps_control_main.py` or `acquisition_workers.py` to change these
values — override them here so every launcher picks up the same config. The
merged configuration (with the password redacted) is logged at INFO on the
first `load_gage_config()` call, so every debug log records the exact values
that ran.

| Environment variable | Field | Default |
|---|---|---|
| `GAGE_XPS_IP` | `xps.ip` | `192.168.0.254` |
| `GAGE_XPS_USER` | `xps.username` | `Administrator` |
| `GAGE_XPS_PASSWORD` | `xps.password` | `Administrator` |
| `GAGE_XPS_SOCKET_TIMEOUT_S` | `xps.socket_timeout_s` | `10.0` |
| `GAGE_XPS_INIT_TIMEOUT_S` | `xps.initialize_timeout_s` | `60.0` |

Example `run_gage/config.local.json`:

```json
{
  "xps": {
    "ip": "10.0.0.42",
    "socket_timeout_s": 5.0,
    "motion_tolerance_mm": 0.001,
    "motion_epsilon_mm": 0.0001,
    "motion_settle_timeout_s": 1.0,
    "initialize_timeout_s": 60.0,
    "initialize_poll_s": 0.5
  },
  "motion": {"scan_velocity_mm_s": 5.0},
  "gui":    {"reconfigure_debounce_ms": 500}
}
```

---

## Exceptions and what they mean

All acquisition workers emit a `WorkerError` payload on the `error` signal
(see [`run_gage/models.py`](run_gage/models.py)). The `category` field lets
GUI handlers dispatch without isinstance checks.

| Exception | Category | Typical cause | Next step |
|---|---|---|---|
| `ConfigurationError` | `config` | Invalid GUI settings (time window, trigger level, unsupported input range) or commit rejected by the SDK. | Fix the setting reported in the message, then retry. |
| `TriggerTimeoutError` | `transient` | No trigger pulse arrived within `AcquisitionConfig.effective_timeout_s()`. | Check the external trigger cable/signal source. Retry — the run may succeed next time. |
| `AcquisitionStoppedError` | `cancelled` | User pressed Stop mid-capture. | No action required. |
| `TransferError` | `hardware` | `PyGage.StartCapture` / `TransferData` returned an error code. | Reconnect the digitizer; check the SDK error message for detail. |
| `AcquisitionInconsistencyError` | `hardware` | Saved frames disagreed on time axis (trigger delay changed mid-capture). | Re-capture; the matrix was not written to avoid silent misalignment. |
| `MotionTimeoutError` | `motion` | Stage reported position disagreed with the commanded target by more than `xps.motion_tolerance_mm` after a (blocking) move returned. | Check for physical obstructions or a limit trip; widen `motion_tolerance_mm` in `config.local.json` only if your servo precision genuinely exceeds the default. |
| `StageInitializationError` | `motion` | A group failed to initialize or home within `xps.initialize_timeout_s`. | Power-cycle the XPS controller; confirm no physical obstruction. |

---

## Troubleshooting

- **`Could not import GageConstants…`** at GUI launch — the Gage SDK is
  missing. Confirm `run_gage/from_gage/` contains `GageConstants.py` and the
  matching `PyGage3_64.pyd`. Both are copied from the Gage installer.
- **`Login failed` from the motion control window** — the XPS is unreachable
  or the credentials are wrong. Verify the IP / username / password via the
  env vars or `config.local.json` (see above), then click **Reconnect**.
- **Trigger timeouts that did not happen before** — the acquisition timeout
  now scales with the capture duration (`effective_timeout_s`). A recurring
  timeout on short captures usually means the trigger signal is absent or the
  trigger level is outside 0.1–5.0 V. Set `AcquisitionConfig.acquisition_timeout_s`
  to override the computed value if the default is too tight for your setup.
- **`.failed` marker in a beammap folder** — a scan crashed or was stopped.
  Open the file for the JSON error report; existing position files are kept.
- **Calibration dropdown is empty** — check
  `read_gage/calibrations/calibration_lookup.csv` is present and matches the
  filename convention documented below.

---

## Reading saved data

Example scripts and example data are provided in `examples`.

### Python

A-line and M-mode waveforms are saved as NumPy binary files (`.npy`); Beammap
scans save the per-position arrays under a single output folder. A JSON
sidecar (`<stem>_parameters.json` or `acquisition_parameters_<timestamp>.json`)
records the acquisition and display configuration used for the capture.

```python
from read_gage.python import read_aline_output, read_mmode_output, read_beammap_output

# A-line — pass the .npy data path
# If a calibration was selected during acquisition, a .cal sidecar is saved
# alongside the .npy file and loaded automatically for pressure conversion.
result = read_aline_output('path/to/example_aline_with_cal_file.npy')
# result['time_axis_us']    — 1-D time vector
# result['voltage_data_v']  — samples × channels array
# result['pressure_data_pa']— same shape, or None if no calibration

# M-mode
result = read_mmode_output('path/to/mmode_20260330_161627.npy')
# result['time_axis_us']    — 1-D time vector
# result['voltage_data_v']  — samples × alines array
# result['pressure_data_pa']— same shape, or None if no calibration

# Beammap (pass the scan folder)
result = read_beammap_output('path/to/beammap_20260330_160251/')
# result['coordinates_mm']       — positions × 3 raw scan coordinates (mm)
# result['voltage_peak_pos/neg'] — n1 × n2 × n3 peak maps (V)
# result['pressure_peak_pos/neg']— n1 × n2 × n3 peak maps (Pa) or None
# result['dim1/2/3']             — centred spatial axes (mm)
# result['save_dir']             — folder the scan was loaded from

# Override the calibration file (full path to an Onda .txt file)
result = read_aline_output('aline.npy',
    calibration_file_path='C:/path/to/h1393_p1322_rightangle_highgain.txt')
```

### MATLAB

Add `read_gage/matlab/` to your MATLAB path, then:

```matlab
addpath('C:\...\gage_python\read_gage\matlab');

% A-line (hydrophone selected during acquisition — .cal sidecar loaded automatically)
[data, info] = read_gage_output('path\to\example_aline_with_cal_file.npy');
plot(data.timeScale, data.pressureData * 1e-6);   % MPa

% A-line (no hydrophone selected — pass calibration file explicitly)
[data, info] = read_gage_output('path\to\example_aline_without_cal_file.npy', ...
    'C:\path\to\h1344_p2040_rightangle_atten.txt');
plot(data.timeScale, data.voltageData);

% M-mode
[data, info] = read_gage_output('path\to\mmode_20260330_161627.npy');
imagesc(1:size(data.voltageData,2), data.timeScale, data.voltageData);
set(gca,'YDir','Normal');

% Beammap (pass the folder)
[data, info] = read_gage_output('path\to\beammap_20260330_160251\');
imagesc(data.dim1, data.dim2, data.voltageData_peakPos(:,:,1));
axis image; colorbar;
```

A warning is printed if the peak signal frequency falls outside the calibration
frequency range.

---

## Calibration files

Pressure conversion works by applying an FFT-based sensitivity calibration to the recorded voltage traces.  The calibration data lives in `read_gage/calibrations/`.

### File types

| File | Purpose |
|---|---|
| `calibration_lookup.csv` | Catalogue that maps a human-readable name to a calibration file and its hardware description. Used by the GUIs to populate the hydrophone/preamp dropdowns. |
| `*.txt` | Onda-format calibration files. Each file has a plain-text header followed by `HEADER_END`, then rows of `FREQ_MHZ  SENS_DB  SENS_VPERPA  SENS_V2CM2PERW`. The reader uses `FREQ_MHZ` and `SENS_VPERPA` columns only. |
| `*.cal` | Binary sidecar saved next to each `.npy` data file when a calibration is selected during acquisition. Contains the `[freq_mhz, sens_v_per_pa]` array extracted at save time. The Python and MATLAB readers load this automatically — no `calibration_file_path` argument needed when a `.cal` is present. Run `_migrate_cal_sidecars.py` to generate `.cal` files for older datasets that pre-date this format. |

### Adding a new calibration

1. **Get the Onda `.txt` file** from your hydrophone calibration report.  
   The file must contain a `HEADER_END` line followed by data rows in the format:  
   `FREQ_MHZ  SENS_DB  SENS_VPERPA  SENS_V2CM2PERW`

2. **Copy the `.txt` file** into `read_gage/calibrations/`:
   ```
   read_gage/calibrations/h<hyd_sn>_p<amp_sn>_<connector>_<gain>.txt
   ```
   Suggested naming convention — `h` = hydrophone serial, `p` = preamplifier serial, connector type, gain setting.  
   Example: `h1393_p1322_rightangle_highgain.txt`

3. **Add a row to `calibration_lookup.csv`**:
   ```
   name,calibrationFile,hydrophone_preamplifier,connector,attenuator,gain
   HNA400-1393_rightangle_highgain,h1393_p1322_rightangle_highgain.txt,HNA400-1393_AH2020-1322,rightangle,None,high
   ```
   - `name` — display label shown in the GUI dropdowns.
   - `calibrationFile` — filename (not path) of the `.txt` file in `read_gage/calibrations/`.
   - `hydrophone_preamplifier` — `<hydrophone_model>-<hyd_sn>_<amp_model>-<amp_sn>`.
   - `connector`, `attenuator`, `gain` — use `None` if not applicable.

4. **Verify** the entry is visible in the GUI dropdown or via Python:
   ```python
   from read_gage.python.calibration_loader import get_names
   print(get_names())
   ```

### Updating an existing calibration

Replace the `.txt` file in `read_gage/calibrations/` with the new version.  
If the filename changes (e.g. after a re-calibration with a new serial number), also update the `calibrationFile` column in `calibration_lookup.csv`.

### Using a calibration outside the catalogue

Pass the full path directly to any reader — no CSV entry is needed:

```python
result = read_aline_output('aline.npy',
    calibration_file_path='C:/path/to/my_custom_calibration.txt')
```

---

## Key scripts

| File | Purpose |
|---|---|
| `run_gage/models.py` | Frozen dataclasses (`AcquisitionConfig`, `DisplayConfig`, `AcquisitionFrame`) and custom exceptions used across the entire package |
| `run_gage/controller.py` | Wraps the Gage CompuScope SDK: board configuration, trigger arming, and raw data transfer |
| `run_gage/acquisition_workers.py` | Qt `QObject` workers that run acquisition loops in background threads; handles live preview, file saving, and beammap position sequencing |
| `run_gage/beammap_utils.py` | Pure-NumPy function that computes the full ordered snake-pattern grid of scan positions from centre, step size, and range inputs |
| `read_gage/python/calibration_loader.py` | Reads `calibration_lookup.csv` to match hydrophone/preamp/attenuator selections to Onda `.txt` sensitivity files; exposes `load_calibration_data()` |
| `run_gage/gui/app_aline.py` | A-line GUI: live waveform + FFT, configurable acquisition settings, triggered single-shot save |
| `run_gage/gui/app_mmode.py` | M-mode GUI: time-scrolling 2-D image built from repeated A-line acquisitions, circular buffer save |
| `run_gage/gui/app_beammap.py` | Beammap GUI: drives the XPS stage through a grid of positions while collecting and displaying a 2-D peak-pressure map |
| `run_gage/gui/app_motion_control.py` | Standalone Newport XPS motion-stage controller; opens as a child window from the beammap and A-line GUIs |
| `run_gage/xps_control_main.py` | Newport XPS back-end functions (connect, status, move, get positions) used by `app_motion_control.py` |
| `run_gage/newportxps/newportxps.py` | Low-level Newport XPS Python driver (`NewportXPS` class) |
| `run_gage/gui/app_find_spatial_peak.py` | Spatial peak finder GUI using Nelder-Mead simplex optimisation to locate the acoustic focus |
| `run_gage/gui/app_find_spatial_peak_brent.py` | Spatial peak finder GUI using Brent 1-D bracket method per axis to locate the acoustic focus |
| `run_gage/gui/plot_widgets.py` | `AlinePlotCanvas`, `FftPlotCanvas`, `MmodePlotCanvas`, `BeammapPlotCanvas` — thin matplotlib-in-Qt canvases |
| `read_gage/python/output_readers.py` | `read_aline_output`, `read_mmode_output`, `read_beammap_output` — load saved `.npy` data files and their JSON sidecars; optionally convert to pressure via FFT calibration |
| `read_gage/matlab/read_gage_output.m` | Single MATLAB function; detects data type from the JSON sidecar and dispatches to the appropriate reader; supports optional calibration path override |
| `_migrate_cal_sidecars.py` | One-time utility that generates `.cal` sidecar files for existing datasets saved before the automatic sidecar format was introduced |

---

### References

`run_gage/newportxps/` was adapted from the following repository [pyepics/newportxps](https://github.com/pyepics/newportxps) (Copyright (c) 2025 Matthew Newville, The University of Chicago).


