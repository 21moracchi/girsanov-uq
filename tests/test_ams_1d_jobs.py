import importlib.util
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AMS_1D_ROOT = REPO_ROOT / "scripts" / "AMS_1D"
SENSITIVE_MARKERS = (
    "#SBATCH",
    "#MSUB",
    "wckey",
    "xca06001",
    "irsrvhome",
    "issrvshare",
    "/soft/irsrv",
    "/ifpengpfs/",
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ams_1d_run_pipeline_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AMS1DJobSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.ams_root = self.tmp_path / "AMS_1D"
        shutil.copytree(AMS_1D_ROOT / "ams_fit", self.ams_root / "ams_fit")
        shutil.copy2(AMS_1D_ROOT / "weights.npy", self.ams_root / "weights.npy")
        self.pipeline = load_module(self.ams_root / "ams_fit" / "run_pipeline.py")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_prepares_minimal_inputs_with_sanitized_jobs(self):
        ini_dirs = self.pipeline.build_ini_conds_inputs(n_walkers=1, force=True, n_reps=1)
        ams_dirs = self.pipeline.build_ams_inputs(n_ams=1, n_reps=1, force=True)

        generated_dirs = ini_dirs + ams_dirs
        self.assertEqual(len(generated_dirs), 2)

        ini_dir = self.ams_root / "ams_fit" / "ini_conds" / "FV_replica_0"
        ams_dir = self.ams_root / "ams_fit" / "ams" / "ams_0"
        expected_files = [
            ini_dir / "job",
            ini_dir / "run_ini_conds.py",
            ams_dir / "job",
            ams_dir / "job_reweight",
            ams_dir / "run_ams.py",
            ams_dir / "run_reweight.py",
        ]
        for path in expected_files:
            self.assertTrue(path.exists(), f"{path} should exist")

        self.assertIn("python run_ini_conds.py", (ini_dir / "job").read_text())
        self.assertIn("python run_ams.py", (ams_dir / "job").read_text())
        self.assertIn("python run_reweight.py", (ams_dir / "job_reweight").read_text())

        for path in expected_files:
            text = path.read_text()
            for marker in SENSITIVE_MARKERS:
                self.assertNotIn(marker, text, f"{marker!r} leaked in {path}")

        generated_ams = (ams_dir / "run_ams.py").read_text()
        self.assertIn("ams_id = 0", generated_ams)
        self.assertIn("n_reps = 1", generated_ams)
        self.assertIn('parent / "weights.npy"', generated_ams)
        self.assertNotIn("# >>> COMMON", generated_ams)

        for script in [ini_dir / "run_ini_conds.py", ams_dir / "run_ams.py", ams_dir / "run_reweight.py"]:
            py_compile.compile(str(script), doraise=True)

    def test_generated_ams_job_reaches_dependency_imports_when_available(self):
        try:
            import aseams  # noqa: F401
        except Exception as exc:
            self.skipTest(f"aseams is not importable in this environment: {exc}")

        ams_dir = self.pipeline.build_ams_inputs(n_ams=1, n_reps=1, force=True)[0]
        result = subprocess.run(
            [sys.executable, "run_ams.py"],
            cwd=ams_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertNotIn("No such file or directory", result.stderr)
        self.assertNotIn("/ifpengpfs/", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
