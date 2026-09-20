"""Model-free publication/reproducibility regressions."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import benchmark
import analyze


class PublicationTests(unittest.TestCase):
    def test_install_uses_exact_pins(self):
        pins = benchmark.pinned_packages(["datasets", "matplotlib", "python-chess"])
        self.assertEqual(len(pins), 3)
        self.assertTrue(all("==" in pin for pin in pins))

    def test_unknown_dependency_is_not_silently_unpinned(self):
        with self.assertRaises(KeyError):
            benchmark.pinned_packages(["unknown-package"])

    def test_provenance_is_optional_and_explicit(self):
        args = benchmark.build_parser().parse_args(["server", "--provenance", "run.json"])
        self.assertEqual(args.provenance, Path("run.json"))
        self.assertEqual(args.concurrency, 10)

    def test_missing_git_is_supported(self):
        with patch.object(benchmark.subprocess, "check_output", side_effect=OSError):
            result = benchmark.client_provenance()
        self.assertIsNone(result["git_commit"])
        self.assertIn("python", result)
        self.assertNotIn("server", result)

    def test_suspect_runs_are_explicitly_listed(self):
        self.assertEqual(analyze.UNVERIFIED_RUNS,
                         {"results_granite4.2_3b", "results_k2_horizon"})

    def test_preparation_reuses_recorded_revision(self):
        loader = Mock(return_value=[])
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="commit123")
        modules = {"datasets": SimpleNamespace(load_dataset=loader),
                   "chess": SimpleNamespace(),
                   "huggingface_hub": SimpleNamespace(HfApi=lambda: api)}
        def build(task, load, limit, seed):
            load("dataset/repo", split="test")
            return []
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, modules), \
             patch.object(benchmark, "build_task_rows", side_effect=build), \
             patch.object(benchmark, "client_provenance", return_value={}):
            output = Path(directory) / "fixture.jsonl"
            benchmark.prepare_dataset("mmlu", output, 1, 0)
            benchmark.prepare_dataset("mmlu", output, 1, 0)
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["fixture_sha256"], benchmark.file_sha256(output))
        api.dataset_info.assert_called_once_with("dataset/repo")
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(loader.call_args.kwargs["revision"], "commit123")


if __name__ == "__main__":
    unittest.main()
