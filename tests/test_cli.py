import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DATA = Path(__file__).parent / "data"


@unittest.skipUnless((DATA / "mixed_code__tahoma.png").exists(), "tests/data not present")
class CliTest(unittest.TestCase):
    def test_output_file_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            subprocess.check_call([sys.executable, "-m", "arabic_ocr.cli", str(DATA / "mixed_code__tahoma.png"), "-o", str(out)])
            text = out.read_text(encoding="utf-8")
            self.assertIn("الكود", text)
            self.assertIn("٢٠١٢/٠٦/٢٥", text)

    def test_json_stdout(self):
        res = subprocess.run([sys.executable, "-m", "arabic_ocr.cli", "--json", str(DATA / "mixed_code__tahoma.png")], capture_output=True, check=True)
        import json

        data = json.loads(res.stdout.decode("utf-8"))
        self.assertTrue(data["results"])
        self.assertIn("confidence", data["results"][0])


if __name__ == "__main__":
    unittest.main()
