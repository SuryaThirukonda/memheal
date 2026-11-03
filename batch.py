"""Run many recover.py sweeps and run_demo history runs at once, in one process.

Each sweep is IO-bound (it waits on the model), and each extra process costs ~0.5 GB
for its own copy of torch and the embedding model. So one process runs every job on a
thread pool and shares one embedder (behind a lock). A job is one run of one recover.py
command line, or one history run; jobs from different experiments are interleaved so
they all advance together.

Job file, one per line ('#' comments allowed):
    recover <recover.py arguments, including --runs/--suite, --work-subdir and --out>
    history <variant> <clean|poisoned> <flat|evolving|mem0|mem0-additive> [model]

Each recover job writes its records to a part file next to its --out and the parts are
appended to --out in job order at the end, so concurrent jobs never interleave lines.
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import threading
import traceback
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent


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


def _flag(argv: list[str], name: str) -> list[str]:
    """Values following `name` up to the next flag."""
    if name not in argv:
        return []
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
    j = i + 1 + len(_flag(argv, name))
    return argv[:i] + argv[j:]


def recover_jobs(argv: list[str], line_no: int) -> list[tuple[str, list[str], Path, Path]]:
    """Split one recover.py command line into one job per run folder."""
    import recover

    runs = recover.resolve_runs(_flag(argv, "--runs") or ["*"])
    suite = _flag(argv, "--suite")
    if suite:
        keep = recover.suite_variants(suite[0])
        runs = [p for p in runs if recover.run_info(p).variant in keep]
    out = Path(_flag(argv, "--out")[0])
    rest = _without(_without(_without(argv, "--runs"), "--suite"), "--out")
    jobs = []
    for i, run in enumerate(runs):
        part = out.parent / ".parts" / f"{out.stem}.{line_no:03d}.{i:04d}.jsonl"
        jobs.append((f"recover:{out.stem}:{run.name}",
                     ["--runs", str(run), *rest, "--out", str(part)], part, out))
    return jobs


def run_recover(argv: list[str], embedder) -> None:
    import recover
    recover.main(argv, embedder=embedder)


def run_history(variant: str, condition: str, memory_mode: str, model: str | None,
                embedder) -> None:
    import run_demo
    from memheal.llm import DEFAULT_MODEL

    memory_mode, _, pipeline = memory_mode.partition("-")   # mem0-additive -> mem0, additive
    args = Namespace(condition=[condition], variant=[variant], suite=None, n=1, task_index=0,
                     k=5, source_labels=False, memory=memory_mode,
                     model=model or DEFAULT_MODEL, no_record=True, table=False,
                     mem0_pipeline=pipeline or None)
    run_demo.run_once(run_demo.VARIANTS[variant], variant, condition, 0, args, embedder)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args(argv)
    load_dotenv(ROOT / ".env")

    from memheal import memory
    embedder = LockedEmbedder(memory.get_embedder())

    by_experiment: dict[str, list] = {}
    for n, line in enumerate(args.jobs.read_text().splitlines()):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        kind, *rest = shlex.split(line)
        if kind == "recover":
            for label, jargv, part, out in recover_jobs(rest, n):
                by_experiment.setdefault(out.stem, []).append(
                    (label, lambda a=jargv: run_recover(a, embedder), part, out))
        elif kind == "history":
            variant, condition, memory_mode, *model = rest
            by_experiment.setdefault(f"history:{memory_mode}:{model[0] if model else ''}", []).append(
                (f"history:{variant}:{condition}:{memory_mode}",
                 lambda v=variant, c=condition, m=memory_mode, mo=(model[0] if model else None):
                 run_history(v, c, m, mo, embedder), None, None))
        else:
            raise SystemExit(f"line {n + 1}: unknown job kind {kind!r}")

    # Round-robin across experiments so every one of them makes progress.
    jobs = [j for group in itertools.zip_longest(*by_experiment.values()) for j in group if j]
    print(f"{len(jobs)} jobs over {len(by_experiment)} experiments, {args.workers} workers",
          flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn): label for label, fn, _p, _o in jobs}
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                fut.result()
                print(f"[{done}/{len(jobs)}] ok   {futures[fut]}", flush=True)
            except Exception:
                failed += 1
                print(f"[{done}/{len(jobs)}] FAIL {futures[fut]}\n{traceback.format_exc()}",
                      flush=True)

    for _label, _fn, part, out in jobs:
        if part is not None and part.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a") as f:
                f.write(part.read_text())
            part.unlink()
    print(f"finished: {len(jobs) - failed} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
