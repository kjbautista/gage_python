# CLAUDE.md

Guidance for agents working in this repo. Read the repo-root `README.md`
first — this file adds agent-specific notes that aren't user-facing.

## Layout at a glance

```
launch_<name>.py                # thin launchers; do not add logic here
run_gage/
  gui/app_<name>.py             # Qt front ends (one per acquisition mode)
  acquisition_workers.py        # QObject workers running off the GUI thread
  controller.py                 # Gage CompuScope SDK wrapper
  xps_control_main.py           # Newport XPS back-end helpers
  models.py                     # frozen dataclasses + exception hierarchy
  config.py                     # runtime config (env > JSON > defaults)
  from_gage/                    # vendor SDK — do NOT edit
  newportxps/                   # vendored driver — edits to XPS_C8_drivers.py are off-limits
  tests/                        # pytest tests (no hardware required)
read_gage/                      # read-back utilities for saved data
```

## Conventions

- **Frozen dataclasses** for configuration (`AcquisitionConfig`, `DisplayConfig`,
  `GageConfig`). Use `dataclasses.replace(...)` to derive variants.
- **Errors** extend `GageGuiError` in `models.py`. Workers emit `WorkerError`
  payloads (via `error = pyqtSignal(object)`), never raw strings. The
  `category` field drives GUI dispatch.
- **Thread safety**: the digitizer handle is guarded by `GageAlineController.lock`.
  Do not call `PyGage.*` outside a method that holds it.
- **`app_*.py` files** follow a common shape (log handler install, status
  emitter, worker lifecycle in `closeEvent`). Keep new ones consistent.
- **`.npy`** is the on-disk format for acquired data. Metadata sidecars live
  next to the data file as `<stem>_parameters.json` (or
  `acquisition_parameters_<timestamp>.json` for beammap folders).

## Do-not-edit list

- `run_gage/from_gage/` — Gage SDK runtime, replace as a whole when updating.
- `run_gage/newportxps/XPS_C8_drivers.py` — vendor driver; API surface is
  relied on everywhere. Wrap or extend in `xps_control_main.py` instead.
- Hardcoded credentials — override via env vars or `run_gage/config.local.json`
  (see README). Never check a real IP or password into source.

## Running tests

```bash
python -m pytest run_gage/tests/
```

No hardware is required; tests use mocks / in-memory fixtures.

## Making changes

- **Hardware magic numbers** (`CHANNEL_NUM`, `SAMPLE_ALIGNMENT`,
  `CS_SAMPLE_OFFSET_DEFAULT`, `TRIGGER_LEVEL_FULL_SCALE_V`,
  `TRIGGER_LEVEL_PERCENT_MAX`) are hoisted to `run_gage/controller.py`
  with SDK references. If you change one, cite the SDK page/version.
- **Timeouts** flow through `AcquisitionConfig.effective_timeout_s()` and
  `XpsConnectionConfig.initialize_timeout_s`. Do not hard-code sleep loops
  in workers. Note that Newport XPS move calls are already blocking — the
  post-move check in `_MotionMixin._verify_motion_reached` is a tolerance
  comparison, not a settling wait.
- **New exceptions** belong in `models.py` and must also be mapped in
  `_classify_exception`. Add a row in the README's exceptions table.
- **New data-file formats** need a matching reader in
  `read_gage/python/output_readers.py` and a MATLAB dispatch in
  `read_gage/matlab/read_gage_output.m`.

## Known rough edges

- `SpatialPeakWorker` and `BrentSpatialPeakWorker` share most motion/capture
  code via the `_MotionMixin`, but the optimization phases are still
  duplicated. A shared `BaseSpatialPeakWorker` is planned.
- The five `app_*.py` GUIs duplicate startup, logging, and teardown.
  A `BaseAcquisitionApp` base class is planned — if you are adding a new
  app, either factor first or keep the new file tightly in sync with an
  existing one.
