"""Local checks for analysis and upload integrity; no GPU, SUMO or network needed."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("sweep_analysis", "analysis/analyze_new_sweeps.py")
sync = load("sweep_sync", "analysis/sync_new_sweeps.py")
archive = load("sweep_archive", "analysis/archive_failed_sweeps.py")
preflight = load("sweep_preflight", "scripts/qmix_memory_preflight.py")


class SweepIntegrityTests(unittest.TestCase):
    def test_archive_targets_are_exact_failed_set(self):
        import json
        manifest = json.loads(archive.MANIFEST.read_text(encoding="utf-8"))
        pairs = list(archive.archive_pairs(manifest))
        self.assertEqual(len(pairs), 15)
        self.assertEqual(len({source for source, _ in pairs}), 15)
        self.assertTrue(all(dest.name == source.name + "_failed_oom" for source, dest in pairs))

    def test_original_mixer_memory_grows_quadratically(self):
        first = preflight.minimum_mixer_bytes(414, 15, 192)
        doubled = preflight.minimum_mixer_bytes(828, 15, 192)
        self.assertEqual(doubled, 4 * first)
        self.assertGreater(first, 9 * 1024 ** 3)

    def test_cohort_comparison_uses_cav_baseline(self):
        # Two future CAVs took 8 min, three remaining humans 3 min: system baseline 5.
        values = analysis.derive_metrics({"t_pre": 5, "t_test": 4.8, "t_HDV_pre": 3,
                                          "t_CAV": 6, "t_HDV_test": 4},
                                         {"num_agents": 5, "num_machines": 2})
        self.assertEqual(values["cav_baseline_inferred_min"], 8)
        self.assertEqual(values["cav_change_pct"], -25)
        self.assertAlmostEqual(values["system_change_pct"], -4)
        self.assertAlmostEqual(values["test_population_reconciliation_min"], 0)

    def test_failed_run_cannot_be_verified_as_success(self):
        source = {"exit_code": 1, "experiment": "qmix", "benchmark": {}, "training_progress": 0, "media": []}
        remote = SimpleNamespace(state="finished", name="qmix", summary={}, files=lambda: [])
        self.assertTrue(sync.verify_remote(remote, source))
        remote.state = "failed"
        self.assertEqual(sync.verify_remote(remote, source), [])
        remote.state = "crashed"
        self.assertEqual(sync.verify_remote(remote, source), [])

    def test_missing_metric_progress_and_plot_are_detected(self):
        source = {"exit_code": 0, "experiment": "iql_test", "benchmark": {"t_test": 4.8},
                  "training_progress": 4000, "media": [{"file": "run/files/media/images/plot.png"}]}
        remote = SimpleNamespace(state="finished", name="iql_test", summary={}, files=lambda: [])
        self.assertEqual(len(sync.verify_remote(remote, source)), 3)
        remote.summary = {"benchmark/t_test": 4.8, "episode": 4000}
        remote.files = lambda: [SimpleNamespace(name="media/images/plot.png")]
        self.assertEqual(sync.verify_remote(remote, source), [])

    def test_nonfinite_remote_metric_is_rejected(self):
        source = {"exit_code": 0, "experiment": "qmix", "benchmark": {"t_test": 4.8}, "training_progress": 0, "media": []}
        remote = SimpleNamespace(state="finished", name="qmix", summary={"benchmark/t_test": float("nan")}, files=lambda: [])
        self.assertTrue(sync.verify_remote(remote, source))


class WandbPolicyTests(unittest.TestCase):
    def setUp(self):
        # Execute the real tracking function alone, avoiding unrelated Torch/SUMO imports.
        tree = ast.parse((ROOT / "scripts/utils.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_wandb")
        self.sdk = Mock()
        self.sdk.init.return_value = SimpleNamespace(mode="online", offline=False, disabled=False)
        context = {"os": os, "wandb": self.sdk, "HAS_WANDB": True, "Optional": Optional, "Any": Any}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "scripts/utils.py", "exec"), context)
        self.init = context["init_wandb"]

    def call(self, **kwargs):
        return self.init("test", "qmix", "ingolstadt_custom2", {}, **kwargs)

    def test_destination_can_be_overridden_by_environment(self):
        with patch.dict(os.environ, {"WANDB_ENTITY": "personal", "WANDB_PROJECT": "experiment"}, clear=True):
            self.call()
        self.assertEqual(self.sdk.init.call_args.kwargs["entity"], "personal")
        self.assertEqual(self.sdk.init.call_args.kwargs["project"], "experiment")

    def test_online_requirement_does_not_silently_fall_back(self):
        self.sdk.init.side_effect = RuntimeError("No credentials")
        with patch.dict(os.environ, {"WANDB_REQUIRE_ONLINE": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "online initialization failed"):
                self.call()
        self.assertEqual(self.sdk.init.call_count, 1)

    def test_explicit_offline_mode_remains_available(self):
        with patch.dict(os.environ, {"WANDB_MODE": "offline", "WANDB_REQUIRE_ONLINE": "0"}, clear=True):
            self.call()
        self.assertEqual(self.sdk.init.call_args.kwargs["mode"], "offline")

    def test_offline_result_rejected_when_online_required(self):
        self.sdk.init.return_value.offline = True
        with patch.dict(os.environ, {"WANDB_REQUIRE_ONLINE": "1"}, clear=True):
            with self.assertRaises(RuntimeError):
                self.call()


if __name__ == "__main__":
    unittest.main()
