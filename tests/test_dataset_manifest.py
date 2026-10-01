import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.dataset_manifest import append_sample_record


class DatasetManifestTests(unittest.TestCase):
    def test_writes_auditable_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample.npy"
            np.save(sample, np.zeros(109, dtype=np.float32))
            manifest = root / "manifest.jsonl"
            append_sample_record(
                manifest, sample, label="HOLA", signer_id="s001",
                session_id="s001-01", device_id="pixel-7", lighting="led",
                camera_index=0, sharpness=70.0,
                extractor_version="test", consent=True,
            )
            item = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(item["label"], "HOLA")
            self.assertEqual(item["signer_id"], "s001")
            self.assertEqual(len(item["sha256"]), 64)

    def test_rejects_missing_consent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample.npy"
            np.save(sample, np.zeros(109, dtype=np.float32))
            with self.assertRaises(ValueError):
                append_sample_record(
                    root / "manifest.jsonl", sample, label="HOLA", signer_id="s001",
                    session_id="s001-01", device_id="pixel-7", lighting="led",
                    camera_index=0, sharpness=70.0,
                    extractor_version="test", consent=False,
                )


if __name__ == "__main__":
    unittest.main()
