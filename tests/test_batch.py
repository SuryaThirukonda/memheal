"""batch.py: job-file expansion, round-robin order, part merging and failure counting.

recover.main and run_demo.run_once are replaced by fakes that only record what they were
asked and write a line where recover.py would, so no run.db is opened, no model is
called and nothing outside tmp_path is written. Run folders are names only: the run
lookups recover.py would do on disk are faked too.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeEmbedder

import batch
import recover
import run_demo
from memheal import memory
from memheal.llm import DEFAULT_MODEL
from memheal.scenarios import LONGMEMEVAL, MEMBENCH

ADAPTIVE = "office_benefits_enrolment_dense_adaptive_evade"     # in the dense-adaptive suite
DIRECT = "office_expense_receipts_dense_direct"                 # not in it
# name -> (variant, condition)
RUNS = {
    "0001_poisoned_a": (DIRECT, "poisoned"),
    "0002_clean_a": (DIRECT, "clean"),
    "0003_poisoned_b": (ADAPTIVE, "poisoned"),
    "0004_poisoned_c": (DIRECT, "poisoned"),
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Selectors are space-free run-name lists joined by ','; run_info answers from RUNS.
    The fakes record (kind, what) in call order, and recover's writes its --runs name
    to --out the way recover.py appends a record."""
    runs_dir = tmp_path / "runs"
    calls: list[tuple] = []
    lock = threading.Lock()
    hooks: dict[str, object] = {}       # run name -> callable run before the write

    def resolve_runs(selectors):
        return [runs_dir / name for sel in selectors for name in sel.split(",")]

    def run_info(path):
        if Path(path).name == "0009_broken":
            raise sqlite3.DatabaseError("file is not a database")
        variant, condition = RUNS[Path(path).name]
        return SimpleNamespace(variant=variant, condition=condition)

    def fake_recover_main(argv, llm_factory=None, embedder=None):
        run = Path(argv[argv.index("--runs") + 1]).name
        out = Path(argv[argv.index("--out") + 1])
        with lock:
            calls.append(("recover", run, tuple(argv), embedder))
        if run in hooks:
            hooks[run]("before")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps({"run": run, "argv": argv}) + "\n")
        if run in hooks:
            hooks[run]("after")
        return []

    def fake_run_once(scenario, variant, condition, i, args, embedder):
        with lock:
            calls.append(("history", variant, condition, args, embedder))
        if variant in hooks:
            hooks[variant]("before")
        return {}

    monkeypatch.setattr(recover, "resolve_runs", resolve_runs)
    monkeypatch.setattr(recover, "run_info", run_info)
    monkeypatch.setattr(recover, "main", fake_recover_main)
    monkeypatch.setattr(run_demo, "run_once", fake_run_once)

    def fail(*_a, **_k):
        raise AssertionError("the real embedder must not load in tests")

    monkeypatch.setattr(memory, "get_embedder", fail)
    return SimpleNamespace(tmp=tmp_path, calls=calls, hooks=hooks)


def _jobfile(tmp_path: Path, *lines: str) -> Path:
    p = tmp_path / "jobs.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def _records(out: Path) -> list[str]:
    return [json.loads(line)["run"] for line in out.read_text().splitlines()]


def _recover_calls(world) -> list[tuple]:
    return [c for c in world.calls if c[0] == "recover"]


# --- job expansion --------------------------------------------------------------------

def test_a_recover_line_becomes_one_job_per_selected_run(world, capsys):
    out = world.tmp / "res" / "k.jsonl"
    jobs = batch.load_jobs(_jobfile(
        world.tmp, f"recover --runs 0001_poisoned_a,0002_clean_a 0004_poisoned_c,0009_broken "
                   f"--mode none incident --k 3 --work-subdir k --out {out}"), "b1")
    # The clean run is skipped (no --include-clean), the unreadable one too, with a word.
    assert [j.label for j in jobs] == ["recover:res/k:0001_poisoned_a",
                                       "recover:res/k:0004_poisoned_c"]
    assert "skip 0009_broken: unreadable" in capsys.readouterr().out
    assert [j.order for j in jobs] == [(0, 0), (0, 1)]
    assert {j.experiment for j in jobs} == {f"recover:{out}"}
    for job in jobs:
        assert job.out == out and job.part.parent == out.parent / ".parts"
        assert job.part.name.startswith("k.b1.000.")
    assert len({j.part for j in jobs}) == 2

    failed = batch.main([str(_jobfile(
        world.tmp, f"recover --runs 0001_poisoned_a,0002_clean_a 0004_poisoned_c "
                   f"--mode none incident --k 3 --work-subdir k --out {out}")),
        "--workers", "1"], embedder=FakeEmbedder())
    assert failed == []
    calls = _recover_calls(world)
    # Each job got its own run (no late-bound argument) and every other argument as given.
    assert [c[1] for c in calls] == ["0001_poisoned_a", "0004_poisoned_c"]
    for _, run, argv, embedder in calls:
        argv = list(argv)
        assert argv[:2] == ["--runs", str(world.tmp / "runs" / run)]
        assert argv[2:8] == ["--mode", "none", "incident", "--k", "3", "--work-subdir"]
        assert argv.count("--runs") == argv.count("--out") == 1
        assert Path(argv[argv.index("--out") + 1]).parent == out.parent / ".parts"
        assert isinstance(embedder, batch.LockedEmbedder)
        assert embedder.name == FakeEmbedder.name             # attributes pass through
    assert _records(out) == ["0001_poisoned_a", "0004_poisoned_c"]


def test_suite_and_include_clean_filter_the_runs_like_recover(world):
    out = world.tmp / "res" / "a.jsonl"
    every = "0001_poisoned_a,0002_clean_a,0003_poisoned_b,0004_poisoned_c"
    jobs = batch.recover_jobs(["--runs", every, "--suite", "dense-adaptive", "--mode", "none",
                               "--out", str(out)], 0, "b")
    assert [j.label.rsplit(":", 1)[1] for j in jobs] == ["0003_poisoned_b"]
    batch.run_all(jobs, FakeEmbedder(), 1)
    (argv,) = [c[2] for c in _recover_calls(world)]
    assert "--suite" not in argv and "dense-adaptive" not in argv   # applied here, once
    with_clean = batch.recover_jobs(["--runs", every, "--include-clean", "--mode", "none",
                                     "--out", str(out)], 0, "b")
    assert [j.label.rsplit(":", 1)[1] for j in with_clean] == [
        "0001_poisoned_a", "0002_clean_a", "0003_poisoned_b", "0004_poisoned_c"]
    assert all("--include-clean" in j.run.args[0] for j in with_clean)


# --- order ---------------------------------------------------------------------------

@pytest.fixture
def external(monkeypatch):
    """One LongMemEval and one MemBench variant, whether or not this checkout has the
    case files: names registered in the suites (a stand-in scenario is enough, as
    run_once is faked)."""
    lme, mb = "lme_test_case_direct", "mb_test_case_direct"
    for name, suite in ((lme, LONGMEMEVAL), (mb, MEMBENCH)):
        monkeypatch.setitem(suite, name, run_demo.VARIANTS[DIRECT])
        monkeypatch.setitem(run_demo.VARIANTS, name, run_demo.VARIANTS[DIRECT])
    return SimpleNamespace(lme=lme, mb=mb)


def test_jobs_run_round_robin_across_experiments(world, external):
    """Two records files with the same name in different folders are two experiments,
    and two lines into one file are one, in file order. History runs are split by suite
    (catalog, adaptive, LongMemEval, MemBench) as well as memory and model, so the
    suites lower in the file do not wait for the ones above."""
    a, b = world.tmp / "a" / "records.jsonl", world.tmp / "b" / "records.jsonl"
    path = _jobfile(
        world.tmp,
        "# a comment line, then a blank one",
        "",
        f"recover --runs 0001_poisoned_a,0003_poisoned_b --mode none --out {a}",
        f"recover --runs 0004_poisoned_c --mode none --out {b}",
        f"recover --runs 0004_poisoned_c --mode incident --out {a}   # same file as line 3",
        f"history {DIRECT} clean evolving",
        f"history {DIRECT} poisoned evolving",
        f"history {ADAPTIVE} poisoned mem0-additive",
        f"history {ADAPTIVE} clean evolving",
        f"history {external.lme} poisoned evolving",
        f"history {external.mb} poisoned evolving",
    )
    history = [j.experiment for j in batch.load_jobs(path, "b") if j.part is None]
    assert history == ["history:catalog:evolving:", "history:catalog:evolving:",
                       "history:adaptive:mem0-additive:", "history:adaptive:evolving:",
                       "history:longmemeval:evolving:", "history:membench:evolving:"]
    assert batch.main([str(path), "--workers", "1"], embedder=FakeEmbedder()) == []
    order = [c[1] if c[0] == "recover" else f"{c[1]}:{c[2]}:{c[3].memory}" for c in world.calls]
    assert order == [
        "0001_poisoned_a",                               # a: line 3, run 0
        "0004_poisoned_c",                               # b
        f"{DIRECT}:clean:evolving",                      # catalog, evolving
        f"{ADAPTIVE}:poisoned:mem0",                     # adaptive, mem0-additive
        f"{ADAPTIVE}:clean:evolving",                    # adaptive, evolving
        f"{external.lme}:poisoned:evolving",             # LongMemEval
        f"{external.mb}:poisoned:evolving",              # MemBench
        "0003_poisoned_b",                               # a: line 3, run 1
        f"{DIRECT}:poisoned:evolving",                   # catalog, evolving: second
        "0004_poisoned_c",                               # a: line 5
    ]
    assert _records(a) == ["0001_poisoned_a", "0003_poisoned_b", "0004_poisoned_c"]
    assert _records(b) == ["0004_poisoned_c"]


def test_parts_are_merged_in_job_file_order_whatever_finished_first(world):
    out = world.tmp / "res" / "records.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text('{"run": "earlier sweep"}\n')
    foreign = out.parent / ".parts" / "records.000.0000.jsonl"       # another batch's part
    foreign.parent.mkdir()
    foreign.write_text('{"run": "foreign"}\n')
    # The first job finishes last: it waits until the three others have written.
    others_done = threading.Semaphore(0)
    finished: list[str] = []

    def hook(run):
        def at(when):
            if when == "before" and run == "0001_poisoned_a":
                for _ in range(3):
                    assert others_done.acquire(timeout=10)
            if when == "after":
                finished.append(run)
                others_done.release()
        return at

    world.hooks.update({r: hook(r) for r in ("0001_poisoned_a", "0003_poisoned_b",
                                             "0004_poisoned_c")})
    path = _jobfile(world.tmp,
                    f"recover --runs 0001_poisoned_a,0003_poisoned_b --mode none --out {out}",
                    f"recover --runs 0004_poisoned_c,0003_poisoned_b --mode root --out {out}")
    assert batch.main([str(path), "--workers", "4"], embedder=FakeEmbedder()) == []
    assert finished[-1] == "0001_poisoned_a" and len(finished) == 4
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["run"] for r in lines] == ["earlier sweep", "0001_poisoned_a", "0003_poisoned_b",
                                         "0004_poisoned_c", "0003_poisoned_b"]
    assert [r["argv"][r["argv"].index("--mode") + 1] for r in lines[1:]] == [
        "none", "none", "root", "root"]
    # This batch's parts are gone; the other batch's part is neither merged nor touched.
    assert sorted(p.name for p in foreign.parent.iterdir()) == [foreign.name]
    assert foreign.read_text() == '{"run": "foreign"}\n'


# --- failures ------------------------------------------------------------------------

def test_failures_are_counted_and_the_rest_still_merges(world, capsys):
    out = world.tmp / "res" / "records.jsonl"

    def raise_before(exc):
        def hook(when):
            if when == "before":
                raise exc
        return hook

    def raise_after(when):
        if when == "after":
            raise RuntimeError("died after writing one configuration")

    world.hooks.update({
        "0002_clean_a": raise_before(RuntimeError("no snapshot")),
        "0003_poisoned_b": raise_before(SystemExit(2)),          # what argparse raises
        "0004_poisoned_c": raise_after,
        DIRECT: raise_before(ValueError("history run failed")),
    })
    path = _jobfile(world.tmp,
                    f"recover --runs 0001_poisoned_a,0002_clean_a,0003_poisoned_b,0004_poisoned_c "
                    f"--include-clean --mode none --out {out}",
                    f"history {DIRECT} poisoned evolving gpt-5.4")
    failed = batch.main([str(path), "--workers", "3"], embedder=FakeEmbedder())
    assert sorted(failed) == sorted([
        "recover:res/records:0002_clean_a", "recover:res/records:0003_poisoned_b",
        "recover:res/records:0004_poisoned_c",
        f"history:{DIRECT}:poisoned:evolving:gpt-5.4"])
    text = capsys.readouterr().out
    assert "finished: 1 ok, 4 failed" in text
    assert text.count("FAIL ") == 4 and "SystemExit" in text
    # The job that wrote before failing keeps its record; the ones that failed first
    # wrote nothing and leave no part behind.
    assert _records(out) == ["0001_poisoned_a", "0004_poisoned_c"]
    assert not any((out.parent / ".parts").iterdir())


def test_a_torn_part_does_not_swallow_the_next_one(tmp_path):
    out = tmp_path / "records.jsonl"
    parts = [tmp_path / ".parts" / f"p{i}.jsonl" for i in range(2)]
    parts[0].parent.mkdir()
    parts[0].write_text('{"run": "a"}\n{"run": "b", "cut')
    parts[1].write_text('{"run": "c"}\n')
    jobs = [batch.Job(f"j{i}", "e", lambda e: None, (0, i), p, out) for i, p in enumerate(parts)]
    assert batch.merge_parts(list(reversed(jobs))) == 2
    assert out.read_text().splitlines() == ['{"run": "a"}', '{"run": "b", "cut', '{"run": "c"}']


# --- history lines --------------------------------------------------------------------

def test_history_lines_parse_memory_pipeline_and_model(world):
    path = _jobfile(world.tmp,
                    f"history {DIRECT} poisoned mem0-additive",
                    f"history {DIRECT} clean mem0",
                    f"history {ADAPTIVE} poisoned evolving gpt-4.1-mini",
                    f"history {DIRECT} clean flat")
    emb = FakeEmbedder()
    assert batch.main([str(path), "--workers", "1"], embedder=emb) == []
    got = {(c[1], c[2], c[3].memory, c[3].mem0_pipeline, c[3].model) for c in world.calls}
    assert got == {(DIRECT, "poisoned", "mem0", "additive", DEFAULT_MODEL),
                   (DIRECT, "clean", "mem0", None, DEFAULT_MODEL),
                   (ADAPTIVE, "poisoned", "evolving", None, "gpt-4.1-mini"),
                   (DIRECT, "clean", "flat", None, DEFAULT_MODEL)}
    for c in world.calls:
        args = c[3]
        assert (args.n, args.task_index, args.k, args.no_record) == (1, 0, 5, True)
        assert run_demo.mem0_pipeline(args) in (None, "update", "additive")
        assert c[4].encode(["x"]).shape == emb.encode(["x"]).shape


@pytest.mark.parametrize("line", [
    f"history {DIRECT} clean flat-additive",          # a pipeline belongs to mem0 only
    f"history {DIRECT} clean mem0-turbo",
    f"history {DIRECT} clean mem0-",
    f"history {DIRECT} clean",
    f"history {DIRECT} sideways evolving",
    "history no_such_variant clean evolving",
    "recover --runs 0001_poisoned_a --mode none",     # no --out: would hit the canonical file
    "recover --runs 0001_poisoned_a --out a.jsonl b.jsonl",
    "recover --runs 0001_poisoned_a --suite --out a.jsonl",
    "recover --mode none --out a.jsonl",              # neither --runs nor --suite
    # Refused by recover.py's own checks, run on the whole line at load time:
    "recover --runs 0001_poisoned_a --mode sideways --out a.jsonl",
    "recover --runs 0001_poisoned_a --mode incident --k 3 --out a.jsonl",   # default folder
    "frobnicate 0001_poisoned_a",
])
def test_a_bad_line_stops_the_batch_before_anything_runs(world, line):
    path = _jobfile(world.tmp, f"history {DIRECT} clean evolving", line)
    with pytest.raises(SystemExit) as exc:
        batch.main([str(path)])        # the real embedder would load next: get_embedder fails
    assert "line 2" in str(exc.value)
    assert world.calls == []


@pytest.mark.parametrize("out", [recover.DEFAULT_OUT,
                                 recover.DEFAULT_OUT.relative_to(recover.ROOT)])
def test_a_knob_line_aimed_at_the_canonical_records_is_refused(world, out, monkeypatch):
    """Each job writes a part file, so recover.py would never see the canonical --out
    and the merge would append k=3 records to the file measure.py reads as the sweep.
    The whole batch stops before anything runs, and the canonical file is untouched."""
    monkeypatch.chdir(recover.ROOT)    # a relative --out names the canonical file only here
    before = recover.DEFAULT_OUT.stat().st_mtime_ns if recover.DEFAULT_OUT.exists() else None
    path = _jobfile(world.tmp,
                    f"recover --runs 0001_poisoned_a --mode none --out {world.tmp / 'ok.jsonl'}",
                    f"recover --runs 0004_poisoned_c --mode incident --k 3 --work-subdir k "
                    f"--out {out}")
    with pytest.raises(SystemExit) as exc:
        batch.main([str(path)])
    assert str(exc.value).startswith("line 2: recover.py refuses this line: ")
    assert "needs its own --out" in str(exc.value)
    assert world.calls == []
    after = recover.DEFAULT_OUT.stat().st_mtime_ns if recover.DEFAULT_OUT.exists() else None
    assert after == before
    # The same knob line with an --out of its own loads.
    assert len(batch.recover_jobs(["--runs", "0004_poisoned_c", "--mode", "incident", "--k", "3",
                                   "--work-subdir", "k", "--out", str(world.tmp / "k.jsonl")],
                                  0, "b")) == 1


def test_parse_memory():
    assert batch.parse_memory("evolving") == ("evolving", None)
    assert batch.parse_memory("mem0") == ("mem0", None)
    assert batch.parse_memory("mem0-update") == ("mem0", "update")
    assert batch.parse_memory("mem0-additive") == ("mem0", "additive")
    for bad in ("Mem0", "flat-update", "mem0-", "evolving-additive", ""):
        with pytest.raises(ValueError):
            batch.parse_memory(bad)
