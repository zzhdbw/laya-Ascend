"""Real ONNX export/parity checks; no checkpoint download or Ascend required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from laya import Agent
from laya.aisbench import INPUT_NAMES
from aisbench_export import compile_bundle, export_bundle, exportable_model
from test_aisbench import tiny_checkpoint


class ExportTests(unittest.TestCase):
    def test_both_rope_families_export_and_preserve_padding_masks(self):
        import onnxruntime as ort
        torch.set_num_threads(1)
        for theta in (10000, 160000):
            with self.subTest(local_rope_theta=theta), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "checkpoint"
                tiny_checkpoint(source, rope_theta=theta)
                bundle = export_bundle(source, root / "bundle", batch_size=2, max_options=8)
                metadata = json.loads((bundle / "aisbench_config.json").read_text())
                self.assertEqual(metadata["seq_len"], 64)
                self.assertFalse((bundle / "model.safetensors").exists())
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                session = ort.InferenceSession(str(bundle / "model.onnx"), sess_options=options,
                                               providers=["CPUExecutionProvider"])
                agent = Agent(str(source), device="cpu")
                for qtype in range(3):
                    ids = torch.randint(1, 8, (2, 64))
                    attention = torch.ones_like(ids)
                    attention[0, 23:] = 0
                    attention[1, 17:] = 0
                    markers = torch.zeros((2, 8), dtype=torch.long)
                    markers[:, :3] = torch.tensor([1, 7, 12])
                    marker_mask = torch.zeros_like(markers, dtype=torch.bool)
                    marker_mask[0, :1] = True
                    marker_mask[1, :3] = True
                    inputs = (ids, attention, markers, marker_mask, torch.full((2,), qtype))
                    with torch.no_grad():
                        expected = agent.model(*inputs)
                    outputs = session.run(None, {n: t.numpy() for n, t in zip(INPUT_NAMES, inputs)})
                    for actual, reference in zip(outputs, expected):
                        np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-4, atol=1e-4)
                with self.assertRaises(FileExistsError):
                    export_bundle(source, bundle)

    def test_export_context_restores_original_modules_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            tiny_checkpoint(temp)
            model = Agent(temp, device="cpu").model
            encoder, layers = model.encoder, model.head.layers
            attention = encoder.config._attn_implementation
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with exportable_model(model, 64):
                    raise RuntimeError("test failure")
            self.assertIs(model.encoder, encoder)
            self.assertIs(model.head.layers, layers)
            self.assertEqual(encoder.config._attn_implementation, attention)

    def test_atc_requires_explicit_target_and_checks_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.onnx").touch()
            (root / "aisbench_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                compile_bundle(root, "")
            with patch("shutil.which", return_value="/toolkit/atc"), patch("subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "did not produce"):
                    compile_bundle(root, "Ascend310P3")
                self.assertTrue(run.call_args.kwargs["check"])
                self.assertIn("--soc_version=Ascend310P3", run.call_args.args[0])
                def compile_fake(*args, **kwargs):
                    (root / "model.om").touch()
                run.side_effect = compile_fake
                self.assertEqual(compile_bundle(root, "Ascend310P3"), root.resolve() / "model.om")
                with self.assertRaises(FileExistsError):
                    compile_bundle(root, "Ascend310P3")


if __name__ == "__main__":
    unittest.main()
