"""Run many recover.py sweeps and run_demo history runs at once, in one process.

Each sweep is IO-bound (it waits on the model), and each extra process costs ~0.5 GB
for its own copy of torch and the embedding model. So one process runs every job on a
thread pool and shares one embedder (behind a lock). A job is one run of one recover.py
command line, or one history run; jobs from different experiments are interleaved so
they all advance together.

Job file, one per line ('#' comments allowed):
    recover <recover.py arguments, including --runs/--suite, --work-subdir and --out>
    history <variant> <clean|poisoned> <flat|evolving|mem0|mem0-<pipeline>> [model]

A recover line becomes one job per run it selects (a clean run only with
--include-clean: recover.py would skip it anyway). Each recover job writes its records
to a part file of its own next to its --out, named after this batch, and once every job
has finished the parts are appended to --out in job-file order (line, then run), so
concurrent jobs never interleave lines, and a part another batch left behind is never
merged or overwritten by this one.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import os
import shlex
import sqlite3
import threading
import time
import traceback
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

ROOT = Path(__file__).parent
MEMORY_MODES = ("flat", "evolving", "mem0")
CONDITIONS = ("clean", "poisoned")


class LockedEmbedder:
    """One embedder shared by every thread; encode is serialised (it is fast)."""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def encode(self, texts):
        with self._lock:
            return self._inner.encode(texts)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass(frozen=True)
class Job:
    """One unit of work. `run` takes the shared embedder, so the job file is parsed (and
    a bad line refused) before the embedding model is loaded."""
    label: str
    experiment: str                      # jobs are interleaved across these
    run: Callable[[Any], None]
    order: tuple[int, int]               # (line, run): the merge order
    part: Path | None = None             # recover: where its records wait for the merge
    out: Path | None = None              # recover: the records file they are merged into


# --- recover lines ----------------------------------------------------------------

def _flag(argv: list[str], name: str) -> list[str] | None:
    """Values following `name` up to the next flag; None when the flag is absent."""
    if name not in argv:
        return None
    i = argv.index(name) + 1
    out = []
    while i < len(argv) and not argv[i].startswith("--"):
        out.append(argv[i])
        i += 1
    return out


def _without(argv: list[str], name: str) -> list[str]:
    if name not in argv:
        return argv
    i = argv.index(name)
    return argv[:i] + argv[i + 1 + len(_flag(argv, name)):]


def _single(argv: list[str], name: str, where: str) -> str | None:
    """The one value of a one-value flag (--out, --suite); None when absent. A flag given
    with no value, or twice, would otherwise be dropped or half-read without a word."""
    values = _flag(argv, name)
    if values is None:
        return None
    if len(values) != 1 or argv.count(name) != 1:
        raise SystemExit(f"{where}: {name} takes exactly one value, once")
    return values[0]


def check_recover_line(argv: list[str], where: str) -> None:
    """Run recover.py's own argument checks on the whole line, with the --out it names.

    Each job is handed a part file in place of --out, so recover.py's guard that keeps
    an experiment's records (a knob, an opt-in mode) out of the canonical
    results/recovery/records.jsonl never sees that file, and merge_parts would append
    them to it without a word. The other checks (a bad --mode, a knob left in the
    default work folder) would only fail job by job, after the batch had started.
    Nothing runs in parallel yet, so borrowing stderr for argparse's message is safe."""
    import recover

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            recover.parse_args(argv)
    except SystemExit:
        said = err.getvalue().strip().splitlines()
        reason = said[-1].partition("error: ")[2] if said else ""
        raise SystemExit(f"{where}: recover.py refuses this line: {reason or 'no reason given'}"
                         ) from None


def run_recover(argv: list[str], embedder) -> None:
    import recover
    recover.main(argv, embedder=embedder)


def recover_jobs(argv: list[str], line_no: int, batch_id: str) -> list[Job]:
    """Split one recover.py command line into one job per run folder it selects.

    The --suite filter and the clean-run skip are applied here, so the job count is
    the work there is; every other argument goes to each job unchanged."""
    import recover

    where = f"line {line_no + 1}"
    out = _single(argv, "--out", where)
    if out is None:
        raise SystemExit(f"{where}: a recover job needs its own --out records file")
    out = Path(out)
    suite = _single(argv, "--suite", where)
    selectors = _flag(argv, "--runs")
    if not selectors and suite is None:
        raise SystemExit(f"{where}: give --runs, --suite, or both")
    check_recover_line(argv, where)
    keep =recover.suite_variants(suite) if suite is not None else None
    with_clean = "--include-clean" in argv
    rest = _without(_without(_without(argv, "--runs"), "--suite"), "--out")
    jobs: list[Job] = []
    for run in recover.resolve_runs(selectors or ["*"]):
        try:
            info = recover.run_info(run)
        except (sqlite3.Error, TypeError, KeyError, ValueError) as exc:
            print(f"{where}: skip {run.name}: unreadable ({exc})", flush=True)
            continue
        if keep is not None and info.variant not in keep:
            continue
        if info.condition == "clean" and not with_clean:
            continue
        i = len(jobs)
        part = out.parent / ".parts" / f"{out.stem}.{batch_id}.{line_no:03d}.{i:04d}.jsonl"
        jobs.append(Job(
            label=f"recover:{out.parent.name}/{out.stem}:{run.name}",
            experiment=f"recover:{out}",
            run=partial(run_recover, ["--runs", str(run), *rest, "--out", str(part)]),
            order=(line_no, i), part=part, out=out))
    return jobs


# --- history lines ----------------------------------------------------------------

def parse_memory(spec: str) -> tuple[str, str | None]:
    """'mem0-additive' -> ('mem0', 'additive'), 'evolving' -> ('evolving', None). A
    pipeline is only valid on mem0 and only one Mem0 has, so a typo is refused before a
    run folder is claimed (run_demo makes the folder before it builds the backend)."""
    from memheal.backends import mem0_backend
    mode, sep, pipeline = spec.partition("-")
    if mode not in MEMORY_MODES or (sep and (mode != "mem0"
                                             or pipeline not in mem0_backend.PIPELINES)):
        raise ValueError(f"unknown memory {spec!r}: one of {', '.join(MEMORY_MODES)} or "
                         f"mem0-<{'|'.join(mem0_backend.PIPELINES)}>")
    return mode, pipeline or None


def run_history(variant: str, condition: str, memory_mode: str, pipeline: str | None,
                model: str | None, embedder) -> None:
    """One run_demo run (n=1, task 0, k=5), not appended to results/variants.jsonl."""
    import run_demo
    from memheal.llm import DEFAULT_MODEL

    args = Namespace(condition=[condition], variant=[variant], suite=None, n=1, task_index=0,
                     k=5, source_labels=False, memory=memory_mode, mem0_pipeline=pipeline,
                     model=model or DEFAULT_MODEL, no_record=True, table=False)
    run_demo.run_once(run_demo.VARIANTS[variant], variant, condition, 0, args, embedder)


def history_suite(variant: str, scenario) -> str:
    """The experiment a history run belongs to, beyond its memory and model: the
    external suites, the adaptive attacks, or the catalog. Without it X10 adaptive, X12
    LongMemEval and MemBench all run evolving with the default model, pool into one
    round-robin experiment and run in file order, so a batch stopped early has barely
    started the suites lower in the file."""
    from memheal.scenarios import ADAPTIVE_CATEGORIES, LONGMEMEVAL, MEMBENCH
    if variant in LONGMEMEVAL:
        return "longmemeval"
    if variant in MEMBENCH:
        return "membench"
    if scenario.attack_category in ADAPTIVE_CATEGORIES:
        return "adaptive"
    return "catalog"


def history_job(argv: list[str], line_no: int) -> Job:
    import run_demo

    where = f"line {line_no + 1}"
    if len(argv) not in (3, 4):
        raise SystemExit(f"{where}: history <variant> <clean|poisoned> <memory> [model]")
    variant, condition, spec, *model = argv
    scenario = run_demo.VARIANTS.get(variant)
    if scenario is None:
        raise SystemExit(f"{where}: unknown variant {variant!r}")
    if condition not in CONDITIONS or condition not in scenario.conditions:
        raise SystemExit(f"{where}: {variant} has no condition {condition!r}")
    try:
        mode, pipeline = parse_memory(spec)
    except ValueError as exc:
        raise SystemExit(f"{where}: {exc}") from None
    model_name = model[0] if model else None
    return Job(label=f"history:{variant}:{condition}:{spec}" + (f":{model_name}" if model else ""),
               experiment=f"history:{history_suite(variant, scenario)}:{spec}:{model_name or ''}",
               run=partial(run_history, variant, condition, mode, pipeline, model_name),
               order=(line_no, 0))


# --- the batch --------------------------------------------------------------------

def load_jobs(path: Path, batch_id: str) -> list[Job]:
    jobs: list[Job] = []
    for n, line in enumerate(path.read_text().splitlines()):
        words = shlex.split(line, comments=True)
        if not words:
            continue
        kind, *rest = words
        if kind == "recover":
            jobs += recover_jobs(rest, n, batch_id)
        elif kind == "history":
            jobs.append(history_job(rest, n))
        else:
            raise SystemExit(f"line {n + 1}: unknown job kind {kind!r}")
    return jobs


def round_robin(jobs: list[Job]) -> list[Job]:
    """The first job of every experiment, then the second of each, ...: every experiment
    makes progress, and within one the job-file order is kept."""
    by: dict[str, list[Job]] = {}
    for job in jobs:
        by.setdefault(job.experiment, []).append(job)
    return [j for group in itertools.zip_longest(*by.values()) for j in group if j is not None]


def run_all(jobs: list[Job], embedder, workers: int) -> list[str]:
    """Run the jobs in this order; the labels of those that raised. SystemExit counts
    too: recover.py's argument checks exit, and the pool hands that back as the job's
    exception, which would otherwise end the batch before any part is merged."""
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job.run, embedder): job for job in jobs}
        for done, fut in enumerate(as_completed(futures), start=1):
            label = futures[fut].label
            try:
                fut.result()
            except (Exception, SystemExit):
                failed.append(label)
                print(f"[{done}/{len(jobs)}] FAIL {label}\n{traceback.format_exc()}", flush=True)
            else:
                print(f"[{done}/{len(jobs)}] ok   {label}", flush=True)
    return failed


def merge_parts(jobs: list[Job]) -> int:
    """Append each recover job's part to its --out in job-file order, then remove it;
    the number of parts merged. A job that failed after some configurations finished
    still has those records (each a complete line) and they are merged; one that failed
    before writing has no part."""
    merged = 0
    for job in sorted((j for j in jobs if j.part is not None), key=lambda j: j.order):
        if not job.part.exists():
            continue
        text = job.part.read_text()
        if text and not text.endswith("\n"):
            text += "\n"              # a torn last line must not swallow the next part's first
        job.out.parent.mkdir(parents=True, exist_ok=True)
        with job.out.open("a") as f:
            f.write(text)
        job.part.unlink()
        merged += 1
    return merged


def main(argv=None, embedder=None) -> list[str]:
    """Run a job file; the labels of the jobs that failed. `embedder` is for tests: the
    default loads .env and the real embedding model, after the job file has parsed."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be at least 1")

    batch_id = f"{args.jobs.stem}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    jobs = load_jobs(args.jobs, batch_id)
    if embedder is None:
        load_dotenv(ROOT / ".env")
        from memheal import memory
        embedder = memory.get_embedder()
    order = round_robin(jobs)
    print(f"{len(jobs)} jobs over {len({j.experiment for j in jobs})} experiments, "
          f"{args.workers} workers", flush=True)
    failed = run_all(order, LockedEmbedder(embedder), args.workers)
    merge_parts(jobs)
    print(f"finished: {len(jobs) - len(failed)} ok, {len(failed)} failed", flush=True)
    for label in failed:
        print(f"  failed: {label}", flush=True)
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
