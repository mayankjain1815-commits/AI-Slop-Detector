import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _load_utils():
    """Load utils.py without needing the openai package installed."""
    if "openai" not in sys.modules:
        stub = types.ModuleType("openai")
        stub.OpenAI = object
        stub.types = types.SimpleNamespace(FileObject=object, Batch=object)
        sys.modules["openai"] = stub

    spec = importlib.util.spec_from_file_location("dg_utils", SRC / "utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dg_utils"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("split_file_count", [1, 2])
def test_prepare_batch_file_is_idempotent(tmp_path, split_file_count):
    utils = _load_utils()

    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text": "alpha"}\n{"text": "beta"}\n', encoding="utf-8")
    output_base = str(tmp_path / "requests.jsonl")

    def prepare():
        return utils.prepare_batch_file(
            str(input_path),
            output_base,
            lambda: "system",
            lambda datum: datum["text"],
            "test-model",
            split_file_count=split_file_count,
        )

    prepare()
    first = {
        path.name: path.read_text(encoding="utf-8")
        for path in tmp_path.glob("requests*.jsonl")
    }

    prepare()
    second = {
        path.name: path.read_text(encoding="utf-8")
        for path in tmp_path.glob("requests*.jsonl")
    }

    assert second == first
