import unittest
from pathlib import Path
from tools.edge_export.offline_guard import OfflineViolation, audit
from tools.edge_export.smoke import ROOT, cleaned, sandbox_path


class EdgeExportGuardTests(unittest.TestCase):
    def test_telemetry_and_autoinstall_actions_blocked(self):
        for event, args in (("socket.connect", ()), ("socket.getaddrinfo", ()), ("os.system", ("pip install anything",)), ("subprocess.Popen", ("pip",)), ("import", ("MNN.tools.mnnconvert",))):
            with self.subTest(event=event), self.assertRaises(OfflineViolation):
                audit(event, args)

    def test_native_converter_and_runtime_imports_allowed(self):
        audit("import", ("_tools",))
        audit("import", ("MNN.nn",))
        audit("open", (str(Path("prepared.npy")),))

    def test_output_cannot_escape_the_ignored_sandbox(self):
        self.assertEqual(sandbox_path(ROOT / "runs/phase2-edge/export-smoke"), (ROOT / "runs/phase2-edge/export-smoke").resolve())
        for path in (ROOT, ROOT / "configs", ROOT / "runs/phase2-edge/../../docs"):
            with self.assertRaises(ValueError):
                sandbox_path(path)

    def test_evidence_redacts_windows_case_and_separator_variants(self):
        for path in (str(ROOT).lower(), str(ROOT).upper(), ROOT.as_posix()):
            self.assertEqual(cleaned(path + '/runs'), '<workspace>/runs')


if __name__ == "__main__":
    unittest.main()
