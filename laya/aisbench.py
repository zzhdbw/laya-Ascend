"""AISBench OM runtime. Importing Laya does not require ais_bench or torch_npu."""
import atexit
import json
import threading
import weakref
from pathlib import Path

import numpy as np
import torch

from .agent import Agent, _fix_tokenizer_config

INPUT_NAMES = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
_LIVE_AGENTS = weakref.WeakSet()
_SESSION_CLASS = None


def _shutdown():
    # CANN must be finalized before native libraries tear down. Doing this in
    # close()/LRU eviction would invalidate other models in the same process.
    global _SESSION_CLASS
    for agent in list(_LIVE_AGENTS):
        agent.close()
    session_class, _SESSION_CLASS = _SESSION_CLASS, None
    if session_class is not None:
        session_class.finalize()


atexit.register(_shutdown)

MODEL_DIRS = {
    "english": "laya",
    "multilingual": "laya-multilingual",
    "typed-decisions": "laya-typed-decisions",
}


def input_shapes(batch_size, seq_len, max_options):
    return {
        "input_ids": (batch_size, seq_len),
        "attention_mask": (batch_size, seq_len),
        "marker_pos": (batch_size, max_options),
        "marker_mask": (batch_size, max_options),
        "qtype": (batch_size,),
    }


def device_index(device):
    """Accept an ACL index or npu:<index>; never silently select a CPU backend."""
    if device is None:
        return 0
    value = str(device)
    if value.startswith("npu:"):
        value = value[4:]
    if not value.isdecimal():
        raise ValueError("AISBench device must be a non-negative index or 'npu:<index>'")
    return int(value)


class AisBenchAgent(Agent):
    """Load an exported bundle (model.om, aisbench_config.json and tokenizer/).

    Tokenization, calibration and answer formatting are inherited from Agent. Only
    the neural forward pass uses AISBench. Static OM batches are padded/chunked, so
    callers can ask any number of choice, score and noul questions. No torch model
    weights are loaded and there is no silent CPU fallback.
    """

    def __init__(self, model_id_or_path, device=None):
        from transformers import AutoTokenizer

        self._lock = threading.RLock()
        self._session = None
        root = Path(model_id_or_path).expanduser()
        manifest = root / "aisbench_config.json"
        if not manifest.is_file() or not (root / "model.om").is_file():
            raise FileNotFoundError(
                "AISBench requires an exported bundle with aisbench_config.json and model.om: %s. "
                "Run export-aisbench.py with --compile first." % root
            )
        with manifest.open(encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format_version") != 1:
            raise ValueError("Unsupported AISBench bundle format")
        for field in ("batch_size", "seq_len", "max_options", "n_act"):
            value = meta.get(field)
            if type(value) is not int or value < 1:
                raise ValueError("Invalid AISBench bundle dimension: %s" % field)
        if meta["max_options"] < 2:
            raise ValueError("AISBench max_options must be at least 2")
        self.batch_size = meta["batch_size"]
        self.seq_len = meta["seq_len"]
        self.max_options = meta["max_options"]
        self.n_act = meta["n_act"]
        self.cfg = meta["agent_config"]
        self._init_temperatures()
        self.device_id = device_index(device)
        self.device = "aisbench:%d" % self.device_id
        self.model_name = str(root)
        _fix_tokenizer_config(str(root))
        self.tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)

        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as exc:
            raise ImportError(
                "Install the Ascend ais_bench inference and aclruntime wheels matching your "
                "Python/CANN environment, then source the CANN set_env.sh. See README-AISBench.md."
            ) from exc
        if not hasattr(InferSession, "free_resource"):
            raise ImportError("Outdated ais_bench: install the current Ascend/tools Gitee inference package")
        global _SESSION_CLASS
        _SESSION_CLASS = InferSession
        self._session = InferSession(self.device_id, str(root / "model.om"))
        _LIVE_AGENTS.add(self)
        try:
            expected = input_shapes(self.batch_size, self.seq_len, self.max_options)
            inputs = self._session.get_inputs()
            self._input_names = [desc.name for desc in inputs]
            if len(inputs) != len(expected) or set(self._input_names) != set(expected):
                raise ValueError("OM inputs do not match the Laya export contract")
            for desc in inputs:
                if tuple(desc.shape) != expected[desc.name]:
                    raise ValueError("OM shape does not match aisbench_config.json: %s" % desc.name)
        except Exception:
            self.close()
            raise

    def _forward_batch(self, batch):
        rows, length = batch["input_ids"].shape
        options = batch["marker_pos"].shape[1]
        if length > self.seq_len or options > self.max_options:
            raise ValueError(
                "Request needs seq_len=%d, max_options=%d; OM supports %d, %d. "
                "Re-export a larger bundle (input is not silently truncated)."
                % (length, options, self.seq_len, self.max_options)
            )
        shapes = input_shapes(self.batch_size, self.seq_len, self.max_options)
        logits_parts, act_parts = [], []
        # ACL sessions are not safe for simultaneous infer/free_resource calls.
        with self._lock:
            if self._session is None:
                raise RuntimeError("AISBench agent is closed")
            for start in range(0, rows, self.batch_size):
                count = min(self.batch_size, rows - start)
                feeds = {}
                for name, shape in shapes.items():
                    dtype = np.bool_ if name == "marker_mask" else np.int64
                    fill = self.tok.pad_token_id if name == "input_ids" else 0
                    array = np.full(shape, fill, dtype=dtype)
                    source = batch[name][start:start + count].cpu().numpy()
                    slices = tuple(slice(0, dim) for dim in source.shape)
                    array[slices] = source
                    # Repeat a real row, rather than all-masked attention (NaN).
                    if count < self.batch_size:
                        array[count:] = array[count - 1]
                    feeds[name] = array
                outputs = self._session.infer([feeds[name] for name in self._input_names], mode="static")
                if len(outputs) != 2:
                    raise RuntimeError("Expected OM outputs: logits and act_logits")
                logits, act = (np.asarray(out, dtype=np.float32) for out in outputs)
                if logits.shape != (self.batch_size, self.max_options) or act.shape != (self.batch_size, self.n_act):
                    raise RuntimeError("OM output shapes do not match aisbench_config.json")
                if not np.isfinite(logits[:count]).all() or not np.isfinite(act[:count]).all():
                    raise RuntimeError("OM returned non-finite logits; check compilation precision")
                logits_parts.append(logits[:count, :options].copy())
                act_parts.append(act[:count].copy())
        return torch.from_numpy(np.concatenate(logits_parts)), torch.from_numpy(np.concatenate(act_parts))

    def close(self):
        """Release this model only; never finalize ACL globally (other agents may use it)."""
        with self._lock:
            session, self._session = self._session, None
            if session is not None:
                session.free_resource()
            _LIVE_AGENTS.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            try:
                self.close()
            except Exception:
                pass  # Interpreter shutdown may already have torn down ACL.
