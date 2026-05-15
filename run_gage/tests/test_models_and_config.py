"""Tests for AcquisitionConfig validation, effective timeout, WorkerError payloads, and config loader."""

from __future__ import annotations

import os
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from run_gage import config as config_module
from run_gage.config import GageConfig, XpsConnectionConfig, load_gage_config
from run_gage.models import (
    ACQUISITION_TIMEOUT_SAFETY_FACTOR,
    ACQUISITION_TIMEOUT_TRIGGER_SLACK_S,
    AcquisitionConfig,
    AcquisitionInconsistencyError,
    ConfigurationError,
    MIN_ACQUISITION_TIMEOUT_S,
    MotionTimeoutError,
    StageInitializationError,
    TransferError,
    TriggerTimeoutError,
    WorkerError,
    _classify_exception,
)


def _build_acq_config(**overrides) -> AcquisitionConfig:
    defaults = dict(
        t_start_us=0.0,
        t_end_us=10.0,
        sample_rate_hz=200e6,
        trigger_level_v=2.5,
        input_range_v=0.1,
        dc_offset_mv=0.0,
        n_alines=1,
    )
    defaults.update(overrides)
    return AcquisitionConfig(**defaults)


class AcquisitionConfigValidationTests(unittest.TestCase):
    def test_rejects_inverted_time_window(self) -> None:
        cfg = _build_acq_config(t_start_us=10.0, t_end_us=0.0)
        with self.assertRaises(ConfigurationError):
            cfg.validate()

    def test_rejects_trigger_level_out_of_range(self) -> None:
        with self.assertRaises(ConfigurationError):
            _build_acq_config(trigger_level_v=10.0).validate()
        with self.assertRaises(ConfigurationError):
            _build_acq_config(trigger_level_v=-0.5).validate()

    def test_rejects_unsupported_input_range(self) -> None:
        with self.assertRaises(ConfigurationError):
            _build_acq_config(input_range_v=0.3).validate()

    def test_rejects_zero_alines(self) -> None:
        with self.assertRaises(ConfigurationError):
            _build_acq_config(n_alines=0).validate()

    def test_rejects_nonpositive_timeout_when_set(self) -> None:
        with self.assertRaises(ConfigurationError):
            _build_acq_config(acquisition_timeout_s=0.0).validate()


class EffectiveTimeoutTests(unittest.TestCase):
    def test_uses_explicit_value_when_set(self) -> None:
        cfg = _build_acq_config(acquisition_timeout_s=42.0)
        self.assertEqual(cfg.effective_timeout_s(), 42.0)

    def test_short_frame_is_near_minimum_floor(self) -> None:
        cfg = _build_acq_config(t_start_us=0.0, t_end_us=1.0)
        # Frame duration ≈ 1 us → scaled term is negligible; result is dominated by slack+floor.
        self.assertGreaterEqual(cfg.effective_timeout_s(), MIN_ACQUISITION_TIMEOUT_S)
        self.assertLess(cfg.effective_timeout_s(), MIN_ACQUISITION_TIMEOUT_S + 0.001)

    def test_long_frame_scales_with_duration(self) -> None:
        cfg = _build_acq_config(t_start_us=0.0, t_end_us=2_000_000.0)
        frame_s = 2.0
        expected = ACQUISITION_TIMEOUT_SAFETY_FACTOR * frame_s + ACQUISITION_TIMEOUT_TRIGGER_SLACK_S
        self.assertEqual(cfg.effective_timeout_s(), expected)


class WorkerErrorTests(unittest.TestCase):
    def test_classify_each_category(self) -> None:
        self.assertEqual(_classify_exception(ConfigurationError('x')), 'config')
        self.assertEqual(_classify_exception(MotionTimeoutError('x')), 'motion')
        self.assertEqual(_classify_exception(StageInitializationError('x')), 'motion')
        self.assertEqual(_classify_exception(TriggerTimeoutError('x')), 'transient')
        self.assertEqual(_classify_exception(TransferError('x')), 'hardware')
        self.assertEqual(_classify_exception(AcquisitionInconsistencyError('x')), 'hardware')
        self.assertEqual(_classify_exception(ValueError('x')), 'internal')

    def test_from_exception_captures_traceback_and_recovery(self) -> None:
        try:
            raise TriggerTimeoutError('timed out')
        except TriggerTimeoutError as exc:
            payload = WorkerError.from_exception(exc, recovery_errors=['homing failed'])
        self.assertEqual(payload.category, 'transient')
        self.assertEqual(payload.message, 'timed out')
        self.assertEqual(payload.cause_qualname, 'TriggerTimeoutError')
        self.assertIn('TriggerTimeoutError', payload.traceback_text)
        self.assertEqual(payload.recovery_errors, ('homing failed',))

    def test_str_renders_recovery_block(self) -> None:
        payload = WorkerError(
            category='motion',
            message='stage stuck',
            traceback_text='Traceback...',
            cause_qualname='MotionTimeoutError',
            recovery_errors=('disconnect failed',),
        )
        rendered = str(payload)
        self.assertIn('MotionTimeoutError: stage stuck', rendered)
        self.assertIn('Recovery warnings:', rendered)
        self.assertIn('- disconnect failed', rendered)


class GageConfigLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_cache = config_module._CACHED_CONFIG
        config_module._CACHED_CONFIG = None
        self._env_keys = (
            'GAGE_XPS_IP', 'GAGE_XPS_USER', 'GAGE_XPS_PASSWORD',
            'GAGE_XPS_SOCKET_TIMEOUT_S', 'GAGE_XPS_INIT_TIMEOUT_S',
        )
        self._saved_env = {k: os.environ.pop(k, None) for k in self._env_keys}

    def tearDown(self) -> None:
        config_module._CACHED_CONFIG = self._saved_cache
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_defaults_when_no_overrides(self) -> None:
        with TemporaryDirectory() as tmp:
            merged = load_gage_config(config_path=Path(tmp) / 'missing.json', refresh=True)
        self.assertIsInstance(merged, GageConfig)
        self.assertEqual(merged.xps, XpsConnectionConfig())

    def test_env_overrides_take_effect(self) -> None:
        os.environ['GAGE_XPS_IP'] = '10.0.0.99'
        os.environ['GAGE_XPS_USER'] = 'labuser'
        os.environ['GAGE_XPS_SOCKET_TIMEOUT_S'] = '7.5'
        with TemporaryDirectory() as tmp:
            merged = load_gage_config(config_path=Path(tmp) / 'missing.json', refresh=True)
        self.assertEqual(merged.xps.ip, '10.0.0.99')
        self.assertEqual(merged.xps.username, 'labuser')
        self.assertEqual(merged.xps.socket_timeout_s, 7.5)

    def test_file_overrides_apply_and_env_wins_over_file(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'config.local.json'
            config_path.write_text(
                '{"xps": {"ip": "1.2.3.4", "socket_timeout_s": 3.0},'
                ' "motion": {"scan_velocity_mm_s": 25.0}}',
                encoding='utf-8',
            )
            # env sets ip but file sets socket_timeout_s; motion only comes from file.
            os.environ['GAGE_XPS_IP'] = '5.6.7.8'
            merged = load_gage_config(config_path=config_path, refresh=True)

        self.assertEqual(merged.xps.ip, '5.6.7.8')  # env > file
        self.assertEqual(merged.xps.socket_timeout_s, 3.0)  # file > default
        self.assertEqual(merged.motion.scan_velocity_mm_s, 25.0)

    def test_unknown_keys_in_file_are_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'config.local.json'
            config_path.write_text('{"xps": {"unknown_field": "x"}}', encoding='utf-8')
            merged = load_gage_config(config_path=config_path, refresh=True)
        self.assertEqual(asdict(merged.xps), asdict(XpsConnectionConfig()))


if __name__ == '__main__':
    unittest.main()
