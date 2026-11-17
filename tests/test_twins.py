"""Clean twins must be the same agent: same model and the same memory store."""

from pathlib import Path

from recover import RunInfo


def _info(memory_mode, pipeline=None, model="gpt-5.4-mini"):
    config = {"agent": {"memory_mode": memory_mode}}
    if pipeline:
        config["mem0"] = {"pipeline": pipeline}
    return RunInfo(Path("x"), "r", "v", "clean", model, config, {})


def test_memory_label_separates_mem0_pipelines():
    assert _info("evolving").memory_label == "evolving"
    assert _info("mem0", "update").memory_label == "mem0"
    assert _info("mem0").memory_label == "mem0"
    assert _info("mem0", "additive").memory_label == "mem0-additive"
    assert _info("mem0", "additive").memory_label != _info("mem0", "update").memory_label
