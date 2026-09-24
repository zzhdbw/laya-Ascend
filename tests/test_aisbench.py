"""Offline AISBench adapter tests: real tokenization/model, fake ACL transport only."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from laya import Agent, AisBenchAgent, Router, load
from laya.aisbench import INPUT_NAMES, device_index, input_shapes
from laya.common import DecisionModel


def tiny_checkpoint(root, rope_theta=160000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = ModernBertConfig(
        vocab_size=8, hidden_size=64, num_hidden_layers=3, num_attention_heads=2,
        intermediate_size=96, max_position_embeddings=256, local_attention=8,
        pad_token_id=0, bos_token_id=2, eos_token_id=3, cls_token_id=2, sep_token_id=3,
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 160000},
                         "sliding_attention": {"rope_type": "default", "rope_theta": rope_theta}},
    )
    config.save_pretrained(root / "encoder")
    tok = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                              "[MASK]": 4, "hello": 5, "yes": 6, "no": 7}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                       cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tokenizer.save_pretrained(root / "tokenizer")
    torch.manual_seed(17)
    model = DecisionModel(ModernBertModel(config), head_layers=1).eval()
    save_file(model.state_dict(), root / "model.safetensors")
    cfg = {"encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
           "max_len": 64, "head_max_len": 32, "temperature": [1.1, 1.2, 1.3],
           "temperature_by_options": {"choice:2": 1.4}}
    (root / "rl_agent_config.json").write_text(json.dumps(cfg))
    return cfg


QUESTIONS = {
    "choice": {"type": "choice", "instructions": "Pick one", "criteria": ["yes", "no"]},
    "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "medium", "high"]},
    "noul": {"type": "noul", "instructions": "Is this hello?"},
    "single": {"type": "choice", "instructions": "Pick", "criteria": ["only"]},
    "other": {"type": "choice", "instructions": "Pick again", "criteria": {"yes": "true", "no": "false"}},
}


class AisBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cfg = tiny_checkpoint(cls.root)
        cls.reference = Agent(str(cls.root), device="cpu")
        cls.meta = {"format_version": 1, "batch_size": 2, "seq_len": 64, "max_options": 8,
                    "n_act": 2, "agent_config": cls.cfg}
        (cls.root / "aisbench_config.json").write_text(json.dumps(cls.meta))
        (cls.root / "model.om").touch()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        reference = self.reference
        self.sessions = []
        sessions = self.sessions

        class Session:
            def __init__(self, device_id, path):
                self.device_id, self.path = device_id, path
                self.calls, self.freed = [], 0
                sessions.append(self)

            def get_inputs(self):
                # Deliberately reordered to verify descriptor-based input binding.
                return [SimpleNamespace(name=name, shape=shape) for name, shape in
                        reversed(list(input_shapes(2, 64, 8).items()))]

            def infer(self, feeds, mode):
                assert mode == "static"
                values = dict(zip((d.name for d in self.get_inputs()), feeds))
                self.calls.append(values)
                with torch.no_grad():
                    return [v.numpy() for v in reference.model(*(torch.from_numpy(values[n]) for n in INPUT_NAMES))]

            def free_resource(self):
                self.freed += 1

            @staticmethod
            def finalize():
                pass

        self.modules = patch.dict(sys.modules, {
            "ais_bench": SimpleNamespace(), "ais_bench.infer": SimpleNamespace(),
            "ais_bench.infer.interface": SimpleNamespace(InferSession=Session),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_all_question_types_calibration_chunking_and_padding(self):
        with AisBenchAgent(self.root, device="npu:7") as agent:
            expected = self.reference.predict("hello", QUESTIONS)
            actual = agent.predict("hello", QUESTIONS)
            self.assertEqual(actual, expected)
            session = self.sessions[-1]
            self.assertEqual(session.device_id, 7)
            self.assertEqual(len(session.calls), 3)
            last = session.calls[-1]
            for name, shape in input_shapes(2, 64, 8).items():
                self.assertEqual(last[name].shape, shape)
                self.assertTrue(last[name].flags.c_contiguous)
                np.testing.assert_array_equal(last[name][0], last[name][1])
            self.assertEqual(last["marker_mask"].dtype, np.bool_)
            self.assertEqual(last["input_ids"].dtype, np.int64)
            self.assertLess(actual["usage"]["input_tokens"], 5 * 64)
        self.assertEqual(session.freed, 1)
        agent.close()
        self.assertEqual(session.freed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            agent.predict("hello", QUESTIONS)

    def test_empty_questions_do_not_infer(self):
        with AisBenchAgent(self.root) as agent:
            self.assertEqual(agent.predict("hello", {})["answers"], {})
            self.assertEqual(self.sessions[-1].calls, [])

    def test_shape_overflow_is_not_silently_truncated(self):
        with AisBenchAgent(self.root) as agent:
            for length, options in ((65, 2), (8, 9)):
                batch = {name: torch.zeros(shape, dtype=torch.bool if name == "marker_mask" else torch.long)
                         for name, shape in input_shapes(1, length, options).items()}
                with self.assertRaisesRegex(ValueError, "Re-export"):
                    agent._forward_batch(batch)
            self.assertEqual(self.sessions[-1].calls, [])

    def test_router_and_factory_cover_all_three_models(self):
        router = Router(models={n: str(self.root) for n in ("english", "multilingual", "typed-decisions")},
                        backend="aisbench", device="npu:7", preload=True)
        for state, model, expected in (("hello", None, "english"), ("你好", None, "multilingual"),
                                        ("hello", "typed-decisions", "typed-decisions")):
            result = router.predict(state, QUESTIONS, model=model)
            self.assertEqual(result["routing"]["model"], expected)
        self.assertEqual(len(self.sessions), 3)
        self.assertTrue(all(s.device_id == 7 for s in self.sessions))
        router.unload()
        self.assertTrue(all(s.freed == 1 for s in self.sessions))
        with load(str(self.root), backend="aisbench", device="7") as agent:
            self.assertIsInstance(agent, AisBenchAgent)
        with self.assertRaises(ValueError):
            load(str(self.root), backend="invalid")

    def test_runtime_does_not_load_torch_weights(self):
        with patch("safetensors.torch.load_file", side_effect=AssertionError("weights must not be loaded")):
            with AisBenchAgent(self.root):
                pass

    def test_dependency_missing_is_actionable(self):
        with patch.dict(sys.modules, {"ais_bench.infer.interface": None}):
            with self.assertRaisesRegex(ImportError, "aclruntime"):
                AisBenchAgent(self.root)

    def test_invalid_bundle_and_output(self):
        with self.assertRaises(FileNotFoundError):
            AisBenchAgent(self.root / "missing")
        with AisBenchAgent(self.root) as agent:
            with patch.object(agent._session, "infer", return_value=[np.full((2, 8), np.nan), np.zeros((2, 2))]):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    agent.predict("hello", QUESTIONS)
            with patch.object(agent._session, "infer", return_value=[np.zeros((1, 8)), np.zeros((2, 2))]):
                with self.assertRaisesRegex(RuntimeError, "output shapes"):
                    agent.predict("hello", QUESTIONS)

    def test_shutdown_releases_models_before_global_finalize(self):
        import laya.aisbench as runtime
        agents = [AisBenchAgent(self.root, device=7) for _ in range(2)]
        session_class = type(self.sessions[0])
        with patch.object(session_class, "finalize") as finalize:
            runtime._shutdown()
            self.assertTrue(all(s.freed == 1 for s in self.sessions))
            finalize.assert_called_once()
            runtime._shutdown()
            finalize.assert_called_once()
        self.assertTrue(all(a._session is None for a in agents))

    def test_games_bypass_torch_npu_device_setup(self):
        root = Path(__file__).resolve().parents[1]
        for game in ("snake", "tetris"):
            spec = importlib.util.spec_from_file_location(game + "_runtime", root / "examples" / game / "runtime.py")
            runtime = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runtime)
            with patch.object(runtime, "choose_device", side_effect=AssertionError("must not import torch_npu")):
                with runtime.load_agent(self.root, "npu:7", "aisbench") as agent:
                    self.assertEqual(agent.device_id, 7)

    def test_device_validation(self):
        for value, expected in ((None, 0), (7, 7), ("npu:7", 7), ("0", 0)):
            self.assertEqual(device_index(value), expected)
        for value in ("cpu", "cuda:0", -1, "auto", True):
            with self.assertRaises(ValueError):
                device_index(value)


if __name__ == "__main__":
    unittest.main()
