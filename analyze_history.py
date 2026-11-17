"""History-run analyses: adaptive poison, other models, external suites, Mem0 (X10-X13).

The history runs (run folders from 1443 on; --min-run) are new agent histories rather
than recoveries of old ones: adaptive poison (Section 12.3), the dense subset under
stronger and other models (8.11, 12.3), the LongMemEval- and MemBench-derived suites
(7.4, 7.5) and the Mem0 backend (12.3). Each is set against what the earlier sweeps
measured on the same scenarios: the original dense runs (0959-1197, gpt-5.4-mini,
evolving memory) and their flat-memory twins (1198-1434). Runs 1435-1442 were smoke
runs and only count with --min-run 1435.

A run is identified from its run.db (variant, condition, model, agent.memory_mode, the
Mem0 pipeline) and summary.json (outcome), never from its folder name, which carries
neither the model nor, for the default Mem0 pipeline, which pipeline ran. recover.py's
records (RECORD_KEYS) are mapped back to model and memory through the same index.

    python analyze_history.py
    python analyze_history.py --min-run 1435 --out results/analysis

Nothing here calls a model or opens a run for writing. Outputs, under --out and all
prefixed history_ (other analyses share the folder):
    history_tables.md            every table, n for every cell
    history.json                 every number in the tables
    history_adaptive.png         attack, survival and recovery by attack category
    history_models.png           attacker and clean-correct rates per model
    history_models_recovery.png  own-task recovery ASR per configuration and model
    history_external.png         attacker rate clean vs poisoned; external-probe accuracy
    history_mem0.png             attacker rate per memory backend; Mem0 write events
    history_heal.png             heal modes and incident per suite: own-task ASR, BMP,
                                 poison left (section 5)

Section 5 reports the heal modes (heal, heal_rebuild, heal_oracle) per suite from each
suite's records.jsonl together with its Phase 2 main.jsonl (--<suite>-main; Mem0 also
reads results/mem0_additive/main.jsonl) and the part files a running batch has not
merged yet (--no-parts to skip them): every configuration on the runs a heal mode was
applied to, one row per knob setting (measure.setting_key, so verify_n, claim_n and
ddmin_budget never pool), what healing dropped and why (dropped_by), the post-heal
check (post_heal_harmful), and each heal mode minus incident and minus heal, paired
within run. Its runs are not independent: a scenario has several history runs, and the
models suite runs each scenario under three models, so every interval in section 5
resamples scenarios (variants) with all their runs (cluster_rate, clustered_paired), n
reads runs (scenarios), and 'small sample' counts scenarios. Sections 1-4 read
records.jsonl alone, as before; section 5 is left out entirely when no main file exists
and no suite has a heal record.

Recovery is set against the canonical sweep like for like: the reference keeps only the
original runs on the (configuration, variant or context) pairs the history records
measured, and ASR is compared on each run's own target task, since history runs replay a
four-task family and the original sweep one or two tasks.

The batch that writes these inputs may still be running, so any of them can be missing
or partial: each section says what it lacks and is skipped rather than failing, a run
without summary.json counts as in progress, and every cell with fewer than 10 runs is
marked 'small sample'.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import measure
from measure import md_table, wilson
from memheal import db, diagnosis
from memheal.heal import RULES as HEAL_RULES
from memheal.labels import AttackSpec
from memheal.scenarios import ALL_VARIANTS, ATTACK_CATEGORIES, DENSE, DENSE_SUBSET, LONGMEMEVAL, MEMBENCH
from memheal.scenarios.adaptive import OWN_NAME_NOTICE_CONTEXTS
from recover import OK_OUTCOMES
from run_demo import memory_label

try:
    # The experiment-knob table is being given one home; take it from wherever it lives.
    from measure import knobs as record_knobs
except ImportError:                                   # pragma: no cover
    from cost import knobs as record_knobs
try:
    from measure import ADAPTIVE_CATEGORIES
except ImportError:
    from memheal.scenarios import ADAPTIVE_CATEGORIES

ROOT = Path(__file__).parent
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "results" / "analysis"
RESULTS = ROOT / "results"
RECORDS = {
    "adaptive": RESULTS / "adaptive" / "records.jsonl",
    "models": RESULTS / "models" / "records.jsonl",
    "external": RESULTS / "external" / "records.jsonl",
    "mem0": RESULTS / "mem0" / "records.jsonl",
    # The canonical gpt-5.4-mini sweep over the original dense runs: what the same
    # scenarios' recovery looked like before the history runs.
    "reference": RESULTS / "recovery" / "records.jsonl",
}
# Phase 2's sweeps over the history runs (data/main_a_jobs.txt, main_b_jobs.txt: none,
# ancestry, selective, incident, then heal, heal_oracle, incident+verify, heal+verify)
# write each suite's main.jsonl; Mem0's additive pipeline has a file of its own.
MAIN = {
    "adaptive": (RESULTS / "adaptive" / "main.jsonl",),
    "models": (RESULTS / "models" / "main.jsonl",),
    "external": (RESULTS / "external" / "main.jsonl",),
    "mem0": (RESULTS / "mem0" / "main.jsonl", RESULTS / "mem0_additive" / "main.jsonl"),
}
HEAL_SUITES = tuple(MAIN)
HEAL_MODES = ("heal", "heal_rebuild", "heal_oracle")
DIAGNOSIS_ADAPTIVE = RESULTS / "diagnosis_adaptive" / "diagnosis.json"
DIAGNOSIS_STANDARD = RESULTS / "diagnosis" / "diagnosis.json"
REFERENCE_MODEL = "gpt-5.4-mini"
SMALL = 10                       # runs below which a cell is marked 'small sample'
SMALL_NOTE = "small sample"

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # categorical slots, fixed order
GOOD, WARNING, CRITICAL = "#0ca30c", "#b07800", "#d03b3b"
INK, MUTED, FAINT, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"

MEMSECBENCH = ("MemSecBench (Section 7.6): no public data or code release was found "
               "(paper only), so it is recorded as unavailable and not run.")
# Diagnosis methods worth setting against each other for evasion: the content filter the
# adaptive_evade page is written around, the agent's own account, and MemHeal's.
DIAGNOSIS_METHODS = ("content_llm", "self_explain", "ancestry", "memheal_cf", "memheal_full")
MEM0_EVENTS = ("ADD", "UPDATE", "DELETE", "facts")
_EXTERNAL_NOTES = re.compile(r"^(\S+) (\S+) \(([^)]+)\)")
_RUN_NUMBER = re.compile(r"^(\d{4})_")


# --- runs ---------------------------------------------------------------------------

def run_number(name: str) -> int | None:
    m = _RUN_NUMBER.match(name)
    return int(m.group(1)) if m else None


def _json(path: Path) -> dict | None:
    """A JSON file, or None while it is missing or half-written."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class Run:
    """One run folder, as its run.db and summary.json describe it."""
    path: Path
    variant: str
    condition: str
    model: str
    memory_mode: str                  # flat | evolving | mem0
    pipeline: str | None = None       # Mem0's write pipeline (update | additive), else None
    outcome: str | None = None        # None until summary.json is written
    task_index: int = 0
    lineage_metrics: bool = True      # False where the Mem0 mirror's lineage is incomplete
    own_task: str | None = None       # the target task it was asked, as its config recorded

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def number(self) -> int:
        return run_number(self.path.name) or -1

    @property
    def scenario(self):
        return ALL_VARIANTS.get(self.variant)

    @property
    def category(self) -> str:
        s = self.scenario
        return s.attack_category if s else ""

    @property
    def backend(self) -> str:
        """How folder names and tables name the memory: flat, evolving, mem0, mem0-additive."""
        return memory_label(self.memory_mode, self.pipeline)

    @property
    def suite(self) -> str:
        if self.variant in LONGMEMEVAL:
            return "longmemeval"
        if self.variant in MEMBENCH:
            return "membench"
        if self.category in ADAPTIVE_CATEGORIES:
            return "adaptive"
        return "dense" if self.variant in DENSE else "other"

    @property
    def context(self) -> str:
        """The scenario without its attack category: what an adaptive case shares with
        its three standard siblings."""
        return self.variant.removesuffix(f"_{self.category}") if self.category else self.variant

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.variant, self.model, self.backend)


def identify(run_dir: Path) -> Run:
    """Read a run's identity. Raises sqlite3.Error, ValueError or LookupError when the
    folder is not (yet) a readable run."""
    conn = db.connect_readonly(Path(run_dir) / "run.db")
    try:
        row = conn.execute("SELECT condition, model, config_json FROM experiment_runs "
                           "LIMIT 1").fetchone()
    finally:
        conn.close()
    if row is None:
        raise LookupError("no experiment_runs row yet")
    cfg = json.loads(row["config_json"])
    summary = _json(Path(run_dir) / "summary.json") or {}
    mode = (cfg.get("agent") or {}).get("memory_mode", "flat")
    mem0 = cfg.get("mem0") or {}
    pipeline = None
    if mode == "mem0":
        pipeline = mem0.get("pipeline") or summary.get("mem0_pipeline") or "update"
    task_index = cfg.get("task_index") or 0
    tasks = (cfg.get("scenario") or {}).get("target_tasks") or ()
    return Run(path=Path(run_dir), variant=cfg.get("variant", ""), condition=row["condition"],
               model=row["model"] or (cfg.get("llm") or {}).get("model", ""),
               memory_mode=mode, pipeline=pipeline, outcome=summary.get("outcome"),
               task_index=task_index, lineage_metrics=bool(mem0.get("lineage_metrics", True)),
               own_task=tasks[task_index] if task_index < len(tasks) else None)


@dataclass
class RunIndex:
    """Every run folder in the ranges an analysis needs, by name."""
    runs_dir: Path
    runs: dict[str, Run] = field(default_factory=dict)
    unreadable: dict[str, str] = field(default_factory=dict)

    @classmethod
    def scan(cls, runs_dir: Path, keep: Callable[[int], bool]) -> "RunIndex":
        idx = cls(Path(runs_dir))
        if idx.runs_dir.is_dir():
            for p in sorted(idx.runs_dir.iterdir()):
                n = run_number(p.name)
                if n is not None and keep(n) and p.is_dir():
                    idx._add(p)
        return idx

    def _add(self, path: Path) -> None:
        try:
            self.runs[path.name] = identify(path)
        except (sqlite3.Error, ValueError, LookupError, OSError) as exc:
            self.unreadable[path.name] = f"{type(exc).__name__}: {exc}"

    def get(self, name: str) -> Run | None:
        """A run by folder name; a record can name a run outside the scanned ranges,
        which is then read on demand."""
        if name not in self.runs and name not in self.unreadable:
            self._add(self.runs_dir / name)
        return self.runs.get(name)

    def between(self, lo: int, hi: int | None = None) -> list[Run]:
        return [r for r in self.runs.values()
                if lo <= r.number and (hi is None or r.number <= hi)]


def finished(runs: Iterable[Run]) -> list[Run]:
    return [r for r in runs if r.outcome is not None]


def attacker(r: Run) -> bool:
    return r.outcome == "attacker"


def correct(r: Run) -> bool:
    return r.outcome in OK_OUTCOMES


def poisoned(runs: Iterable[Run]) -> list[Run]:
    return [r for r in runs if r.condition == "poisoned"]


def clean(runs: Iterable[Run]) -> list[Run]:
    return [r for r in runs if r.condition == "clean"]


def qualified_keys(runs: Iterable[Run]) -> set[tuple]:
    """(variant, model, memory) whose clean controls were all correct: measure's
    qualified_variants, per model and memory, since a twin must share both."""
    ok = defaultdict(list)
    for r in clean(runs):
        ok[r.key].append(correct(r))
    return {k for k, v in ok.items() if v and all(v)}


# --- numbers ------------------------------------------------------------------------

def rate(n: float, d: int, runs: int | None = None) -> dict:
    """A proportion with its Wilson 95% interval. `runs` is what 'small sample' counts
    when the denominator is not runs (tasks, probes)."""
    lo, hi = wilson(n, d)
    runs = d if runs is None else runs
    return {"n": n, "d": d, "rate": n / d if d else None, "ci": [lo, hi] if d else None,
            "runs": runs, "small": runs < SMALL}


def outcome_rate(runs: list[Run], pred: Callable[[Run], bool]) -> dict:
    return rate(sum(pred(r) for r in runs), len(runs))


def mean_ci(values: list[float], n_boot: int, seed: int) -> dict:
    """Mean with a percentile bootstrap 95% interval over the values (runs)."""
    if not values:
        return {"n": 0, "d": 0, "rate": None, "ci": None, "runs": 0, "small": True}
    rnd = random.Random(seed)
    k = len(values)
    boots = sorted(sum(rnd.choice(values) for _ in range(k)) / k for _ in range(n_boot))
    return {"n": sum(values), "d": k, "rate": sum(values) / k,
            "ci": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
            "runs": k, "small": k < SMALL}


def by_key(runs: Iterable[Run], key: Callable[[Run], str],
           value: Callable[[Run], float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        out[key(r)].append(float(value(r)))
    return dict(out)


def paired_diff(a: dict[str, list[float]], b: dict[str, list[float]], n_boot: int,
                seed: int) -> dict | None:
    """Mean over shared keys of (mean of a - mean of b), with a percentile bootstrap 95%
    interval from resampling keys. Each key (a variant or a context) weighs the same
    however many runs it has, so a repeated scenario cannot dominate."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [sum(a[k]) / len(a[k]) - sum(b[k]) / len(b[k]) for k in keys]
    rnd = random.Random(seed)
    k = len(diffs)
    boots = sorted(sum(rnd.choice(diffs) for _ in range(k)) / k for _ in range(n_boot))
    return {"pairs": k, "diff": sum(diffs) / k,
            "ci": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
            "a_higher": sum(d > 0 for d in diffs), "b_higher": sum(d < 0 for d in diffs),
            "small": k < SMALL}


def scenario_of(rec: dict) -> str:
    """Section 5's unit of independence: the scenario (variant). A scenario's history runs
    share its pages, tasks and poison (the models suite runs each one under three models,
    three times), so its runs move together and are resampled together."""
    return rec.get("variant") or rec.get("run_dir", "")


def _whole(x: float) -> float | int:
    return int(x) if float(x).is_integer() else x


def cluster_rate(units: Iterable[tuple[str, float, float]], n_boot: int, seed: int,
                 proportion: bool = True) -> dict:
    """sum(n) / sum(d) over runs, each unit one run's (scenario, n, d), with a percentile
    bootstrap 95% interval that resamples scenarios, each with all of its runs. The point
    is the run-pooled rate the other sections report; only the interval (and what
    'small sample' counts: scenarios) changes. A mean per run is the same with d = 1
    (proportion=False). A proportion of 0 or 1 (no draw can differ) is widened to Wilson
    over its scenarios."""
    per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    runs = 0
    for scenario, n, d in units:
        per[scenario][0] += n
        per[scenario][1] += d
        runs += 1
    n, d, k = sum(s[0] for s in per.values()), sum(s[1] for s in per.values()), len(per)
    out = {"n": _whole(n), "d": _whole(d), "rate": n / d if d else None, "ci": None,
           "runs": runs, "scenarios": k, "small": k < SMALL, "interval": "scenario bootstrap"}
    if not d:
        return out
    rnd, sums, boots = random.Random(seed), list(per.values()), []
    for _ in range(n_boot):
        sample = rnd.choices(sums, k=k)
        bd = sum(s[1] for s in sample)
        if bd:
            boots.append(sum(s[0] for s in sample) / bd)
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    if proportion and n in (0, d):
        # Every draw of an all-or-nothing cell is the same, so the bootstrap cannot see
        # how far the rate could be from it: there the bound is Wilson's with one
        # observation per scenario (what a degenerate [0-0] would otherwise hide).
        w_lo, w_hi = wilson(k * n / d, k)
        lo, hi = min(lo, w_lo), max(hi, w_hi)
    out["ci"] = [lo, hi]
    return out


def clustered_paired(a: dict[str, dict], b: dict[str, dict], value: Callable[[dict], float | None],
                     n_boot: int, seed: int) -> dict | None:
    """A minus B paired within run (a and b map run -> record), clustered by scenario:
    each run's difference stays inside its scenario, a scenario's diff is the mean of
    its runs' diffs, and paired_diff averages scenarios (each weighs the same, however
    many runs it has) and resamples them. Runs where either side lacks the value are
    left out; 'runs' says how many were paired."""
    va: dict[str, list[float]] = defaultdict(list)
    vb: dict[str, list[float]] = defaultdict(list)
    runs = 0
    for rd in sorted(set(a) & set(b)):
        x, y = value(a[rd]), value(b[rd])
        if x is None or y is None:
            continue
        s = scenario_of(a[rd])
        va[s].append(x)
        vb[s].append(y)
        runs += 1
    d = paired_diff(dict(va), dict(vb), n_boot, seed)
    return None if d is None else {**d, "runs": runs, "unit": "scenarios"}


def runs_scenarios(runs: int, scenarios: int) -> str:
    """'89 (20 scenarios)', flagged as a small sample below SMALL scenarios."""
    out = f"{runs} ({scenarios} scenario{'s' * (scenarios != 1)})"
    return f"{out} {SMALL_NOTE}" if scenarios < SMALL else out


def fmt(r: dict | None) -> str:
    if not r or not r.get("d"):
        return "- (n=0)"
    lo, hi = r["ci"]
    n = r["n"]
    num = f"{n:.0f}" if float(n).is_integer() else f"{n:.2f}"
    out = f"{100 * r['rate']:.0f}% ({num}/{r['d']}) [{100 * lo:.0f}-{100 * hi:.0f}]"
    return f"{out} {SMALL_NOTE}" if r["small"] else out


def fmt_mean(r: dict | None, digits: int = 1) -> str:
    if not r or not r.get("d"):
        return "- (n=0)"
    lo, hi = r["ci"]
    n = (f"n={r['d']}" if "scenarios" not in r
         else f"n={r['runs']} runs, {r['scenarios']} scenarios")
    out = f"{r['rate']:.{digits}f} [{lo:.{digits}f}-{hi:.{digits}f}] ({n})"
    return f"{out} {SMALL_NOTE}" if r["small"] else out


def fmt_diff(d: dict | None, unit: str = "variants") -> str:
    if not d:
        return "- (no shared " + unit + ")"
    lo, hi = d["ci"]
    if "runs" in d:          # clustered_paired: runs paired, scenarios resampled
        over = (f"over {d['runs']} runs ({d['pairs']} {d['unit']}; {d['unit']} higher "
                f"{d['a_higher']}, lower {d['b_higher']})")
    else:
        over = f"over {d['pairs']} {unit} (higher {d['a_higher']}, lower {d['b_higher']})"
    out = f"{100 * d['diff']:+.0f} pts [{100 * lo:+.0f}, {100 * hi:+.0f}] {over}"
    return f"{out} {SMALL_NOTE}" if d["small"] else out


def outcome_mix(runs: list[Run]) -> str:
    c = Counter(r.outcome for r in runs)
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "-"


# --- records ------------------------------------------------------------------------

def load_records(path: Path | None) -> list[dict] | None:
    """measure.load_records (torn last line skipped, re-runs deduplicated), or None
    while the file does not exist yet."""
    if path is None or not Path(path).is_file():
        return None
    return measure.load_records(Path(path))


def usable(records: list[dict]) -> tuple[list[dict], int]:
    """Default-knob records on patched pages, the only ones the history batch writes;
    anything else is a different measurement and is counted, not pooled."""
    keep = [r for r in records if not record_knobs(r)
            and (r.get("source") or "patched") == "patched"]
    return keep, len(records) - len(keep)


def is_measured(rec: dict) -> bool:
    """A record measure.recovery_rows counts: a poisoned run, right after recovery."""
    return rec.get("checkpoint", 0) == 0 and rec.get("condition", "poisoned") == "poisoned"


def measured(records: Iterable[dict]) -> list[dict]:
    return [r for r in records if is_measured(r)]


def config_of(rec: dict) -> str:
    """The configuration label a record is tabulated under (row_data's 'config')."""
    return measure.config_label(measure.setting_key(rec), show_source=False)


def own_task_attacker(rec: dict, task_index: int) -> float | None:
    """1.0 if the replay of the run's own target task reached the attacker, else 0.0.

    task_outcomes follow the replayed family's order, and T1 kept every existing task at
    its index, so task_outcomes[task_index] is the same question whether the run replayed
    its recorded pre-T1 family (one or two tasks) or a four-task one. A record without
    per-task outcomes is a single task already (its asr). None when the run's own task
    was not among the replayed ones."""
    outs = rec.get("task_outcomes") or []
    if not outs:
        return None if rec.get("asr") is None else float(rec["asr"])
    if not 0 <= task_index < len(outs):
        return None
    return float(outs[task_index] == "attacker")


def target_record(rec: dict, task_index: int) -> dict | None:
    """The record cut down to the run's own target task, for measure.recovery_rows.

    ASR pooled over replayed tasks (measure.asr_counts) is a different measurement for a
    history run, which replays its recorded four-task family, than for the original
    sweep, which replayed one or two: the extra tasks are other questions. Only the run's
    own task is asked in both, so that is what groups are compared on."""
    v = own_task_attacker(rec, task_index)
    if v is None:
        return None
    if not rec.get("task_outcomes"):
        return rec
    return {**rec, "task_outcomes": [rec["task_outcomes"][task_index]], "asr": v}


def recovery_by(records: list[dict], group: Callable[[dict], str | None]
                ) -> dict[str, list[dict]]:
    """Records per group; records whose group is None are left out."""
    g: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        k = group(r)
        if k is not None:
            g[k].append(r)
    return dict(g)


def row_data(row: measure.RecoveryRow, target: measure.RecoveryRow | None) -> dict:
    """One configuration. asr_target (the run's own task) is what groups are compared on;
    asr_all_tasks pools whatever family the runs replayed (tasks_per_run says how many),
    so it compares only groups that replayed the same family."""
    return {"config": measure.config_label(row.key, show_source=False), "runs": row.runs,
            "asr_target": (rate(target.asr_n, target.asr_d, runs=target.runs) if target
                           else rate(0, 0)),
            "attacked_target": target.attacked_runs if target else 0,
            "target_runs": target.runs if target else 0,
            "asr_all_tasks": rate(row.asr_n, row.asr_d, runs=row.runs),
            "attacked_all_tasks": row.attacked_runs,
            "tasks_per_run": row.asr_d / row.runs if row.runs else 0.0,
            "bmp": rate(row.bmp_n, row.bmp_d, runs=row.runs), "no_baseline": row.no_baseline,
            "poison_left": rate(row.residual_runs, row.residual_d)}


def config_rows(recs: list[dict], task_index: Callable[[dict], int]) -> tuple[list[dict], int]:
    """row_data per configuration in measure.recovery_rows order (poisoned, checkpoint 0,
    ladder), and how many of those records did not replay the run's own task."""
    target, missing = [], 0
    for r in recs:
        t = target_record(r, task_index(r))
        if t is not None:
            target.append(t)
        elif is_measured(r):
            missing += 1
    own = {row.key: row for row in measure.recovery_rows(target)}
    return [row_data(row, own.get(row.key)) for row in measure.recovery_rows(recs)], missing


RECOVERY_HEADER = ["group", "configuration", "runs", "ASR, run's own task (Wilson 95%)",
                   "runs attacked on own task", "ASR, every replayed task", "tasks per run",
                   "BMP (probes kept; Wilson 95%)", "poison left in memory (runs)"]


def recovery_section_rows(groups: dict[str, list[dict]], order: list[str],
                          task_index: Callable[[dict], int]) -> tuple[list[list], dict, int]:
    """Table rows and JSON per group (records per group), in `order`, and the number of
    records left out of the own-task column because that task was not replayed."""
    rows, data, missing = [], {}, 0
    for g in order:
        if g not in groups:
            continue
        data[g], m = config_rows(groups[g], task_index)
        missing += m
        for d in data[g]:
            rows.append([g, d["config"], d["runs"], fmt(d["asr_target"]),
                         f"{d['attacked_target']}/{d['target_runs']}", fmt(d["asr_all_tasks"]),
                         f"{d['tasks_per_run']:.1f}", fmt(d["bmp"]), fmt(d["poison_left"])])
    return rows, data, missing


def match_reference(history: dict[str, list[dict]], reference: list[dict],
                    key: Callable[[dict], str | None], exclude: Iterable[str] = ()
                    ) -> list[dict]:
    """The reference records on exactly the (configuration, key) pairs the history
    records measured, key being a variant or, for adaptive poison, a context.

    A history group's runs can span more variants than its records do (recovery is run
    on a sample), and one configuration can cover fewer variants than another; a
    reference drawn from every variant the runs touch would set a different population
    beside the history rows. Keys in `exclude` (own task not the same) are left out."""
    skip = set(exclude)
    covered = {(measure.setting_key(r), key(r)) for recs in history.values()
               for r in measured(recs)}
    return [r for r in reference if (k := key(r)) not in skip
            and (measure.setting_key(r), k) in covered]


def task_mismatches(runs: Iterable[Run], key: Callable[[Run], str]) -> set[str]:
    """Keys whose runs record different own target tasks: there the own-task comparison
    would not be like for like. Runs that recorded no family are not counted."""
    tasks: dict[str, set[str]] = defaultdict(set)
    for r in runs:
        if r.own_task is not None:
            tasks[key(r)].add(r.own_task)
    return {k for k, t in tasks.items() if len(t) > 1}


def recovery_matched(groups: dict[str, list[dict]], reference: str, key: Callable[[dict], str | None],
                     task_index: Callable[[dict], int], n_boot: int, seed: int
                     ) -> dict[str, dict[str, dict | None]]:
    """group -> configuration -> paired_diff of the own-task attacker rate, group minus
    the reference, matched by key: each variant (or context) weighs the same however
    many runs and records it has, as in the attacker-rate tables."""
    def by_config(recs: list[dict]) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in measured(recs):
            v, k = own_task_attacker(r, task_index(r)), key(r)
            if v is not None and k is not None:
                out[config_of(r)][k].append(v)
        return out
    ref = by_config(groups.get(reference, []))
    return {g: {c: paired_diff(dict(v), dict(ref.get(c, {})), n_boot, seed)
                for c, v in by_config(recs).items()}
            for g, recs in groups.items() if g != reference}


# --- sections -----------------------------------------------------------------------

@dataclass
class Section:
    key: str
    title: str
    md: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)       # missing or thin inputs
    headline: list[str] = field(default_factory=list)    # key numbers for stdout
    omit: str | None = None          # why the section is left out of the report altogether

    def note(self, text: str) -> None:
        self.notes.append(text)
        self.md.append(f"_Note: {text}_")

    def table(self, caption: str, header: list[str], rows: list[list]) -> None:
        if rows:
            self.md.append(f"**{caption}**\n\n{md_table(header, rows)}")
        else:
            self.note(f"{caption}: no rows")

    def skip(self, reason: str) -> "Section":
        self.data["skipped"] = reason
        self.note(f"section skipped: {reason}")
        return self


@dataclass
class Ctx:
    """Everything a section reads: the run index, the options and the records files
    (each read once per invocation)."""
    args: argparse.Namespace
    idx: RunIndex
    _records: dict = field(default_factory=dict)

    def records(self, name: str) -> list[dict] | None:
        if name not in self._records:
            self._records[name] = load_records(getattr(self.args, f"{name}_records"))
        return self._records[name]

    def suite_records(self, suite: str) -> tuple[list[dict], list[dict]]:
        """A suite's records.jsonl and main files with their unmerged part files, one
        record per measurement (a later file's copy wins), and what each file held. The
        sections above read records.jsonl alone (default knobs, no parts), as before."""
        key = f"suite:{suite}"
        if key not in self._records:
            rows, found = [], []
            paths = [(getattr(self.args, f"{suite}_records"), False),
                     *((p, True) for p in getattr(self.args, f"{suite}_main"))]
            for path, main in paths:
                recs, n_parts = read_with_parts(path, self.args.parts)
                rows += recs or []
                found.append({"suite": suite, "path": str(path), "main": main,
                              "state": "missing" if recs is None else "ok",
                              "records": len(recs or ()),
                              "runs": len({r.get("run_dir") for r in recs or ()}),
                              "part_files": n_parts})
            self._records[key] = (last_per_measurement(rows), found)
        return self._records[key]

    def base(self) -> list[Run]:
        """The original dense sweep: evolving memory, dense catalog."""
        lo, hi = self.args.base_runs
        return [r for r in self.idx.between(lo, hi)
                if r.memory_mode == "evolving" and r.variant in DENSE]

    def flat(self) -> list[Run]:
        lo, hi = self.args.flat_runs
        return [r for r in self.idx.between(lo, hi)
                if r.memory_mode == "flat" and r.variant in DENSE]

    def history(self) -> list[Run]:
        return self.idx.between(self.args.min_run)

    def run_of(self, rec: dict) -> Run | None:
        return self.idx.get(rec.get("run_dir", ""))

    def thin(self, sec: Section, what: str, n: int) -> None:
        if n < SMALL:
            sec.note(f"thin input: {what} has {n} finished runs (fewer than {SMALL})")


def inputs_section(ctx: Ctx, sec: Section) -> None:
    """What the analysis could read: run groups, history runs by kind, records files."""
    a = ctx.args
    groups = [("original dense (evolving)", a.base_runs, ctx.base()),
              ("flat twins", a.flat_runs, ctx.flat()),
              ("history", (a.min_run, None), ctx.history())]
    rows, data = [], {}
    for name, (lo, hi), runs in groups:
        span = f"{lo:04d}-{hi:04d}" if hi is not None else f"{lo:04d}+"
        bad = sum(1 for n in ctx.idx.unreadable
                  if (k := run_number(n)) is not None and lo <= k and (hi is None or k <= hi))
        data[name] = {"range": span, "finished": len(finished(runs)),
                      "in_progress": len(runs) - len(finished(runs)), "unreadable": bad}
        rows.append([name, span, data[name]["finished"], data[name]["in_progress"], bad])
        if not finished(runs):
            sec.note(f"missing input: no finished runs in {name} ({span})")
    sec.table("Run folders (identified from run.db and summary.json)",
              ["group", "run numbers", "finished", "in progress (no summary.json)",
               "unreadable"], rows)

    kinds = Counter((r.suite, r.model, r.backend, r.condition if r.outcome else "in progress")
                    for r in ctx.history())
    hist_rows = []
    for suite, model, backend in sorted({k[:3] for k in kinds}):
        n = lambda c: kinds.get((suite, model, backend, c), 0)
        hist_rows.append([suite, model, backend, n("clean"), n("poisoned"), n("in progress")])
    data["history_kinds"] = [dict(zip(("suite", "model", "memory", "clean", "poisoned",
                                       "in_progress"), r)) for r in hist_rows]
    sec.table("History runs by kind (finished clean / poisoned, and in progress)",
              ["suite", "model", "memory", "clean", "poisoned", "in progress"], hist_rows)

    files = []
    for name in RECORDS:
        path = getattr(a, f"{name}_records")
        recs = ctx.records(name)
        status = ("missing" if recs is None else
                  f"{len(recs)} records, {len({r.get('run_dir') for r in recs})} runs")
        if recs is None:
            sec.note(f"missing input: {name} records ({path})")
        files.append([f"{name} records", str(path), status])
    for name, path in (("adaptive diagnosis", a.diagnosis_adaptive),
                       ("standard diagnosis", a.diagnosis_standard)):
        status = "present" if Path(path).is_file() else "missing"
        if status == "missing":
            sec.note(f"missing input: {name} ({path})")
        files.append([name, str(path), status])
    data["files"] = [dict(zip(("input", "path", "status"), f)) for f in files]
    data["unreadable"] = dict(ctx.idx.unreadable)
    sec.table("Records and diagnosis files", ["input", "path", "status"], files)
    sec.data.update(data)


# --- 1. adaptive poison -------------------------------------------------------------

def _evade_group(category: str, variant: str) -> str:
    """adaptive_evade on the contexts whose attacker name is itself notice vocabulary
    is reported apart: a raw-text filter flags those pages through the name alone."""
    ctx_key = variant.removesuffix(f"_{category}").removesuffix("_dense")
    if category == "adaptive_evade" and ctx_key in OWN_NAME_NOTICE_CONTEXTS:
        return "adaptive_evade (own-name notice contexts)"
    return category


def diagnosis_recall(path: Path, keep: Callable[[dict], bool], n_boot: int, seed: int
                     ) -> dict[str, dict[str, dict]] | None:
    """group -> method -> recall/precision, from a diagnose.py diagnosis.json: store
    scope, primary scoring (diagnosis.truth_class), over all runs and incident runs,
    bootstrapped over runs with memheal.diagnosis.summarize."""
    doc = _json(path)
    if not doc:
        return None
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for run in doc.get("runs") or ():
        if keep(run):
            by_group[_evade_group(run.get("category", ""), run.get("variant", ""))].append(run)
    for group, runs in by_group.items():
        for method in DIAGNOSIS_METHODS:
            entry = {}
            for scope, rs in (("all", runs), ("incident", [r for r in runs if r.get("incident")])):
                scores = [s for r in rs if (s := _run_score(r, method)) is not None]
                if scores:
                    entry[scope] = diagnosis.summarize(scores, n_boot, seed)
            if entry:
                out[group][method] = entry
    return dict(out)


def _run_score(run: dict, method: str) -> diagnosis.RunScore | None:
    pred = (run.get("predictions") or {}).get(method)
    if pred is None:
        return None
    mems = run.get("memories") or {}
    cls = {m: diagnosis.truth_class((mems[m]["role"], mems[m]["label"]))
           for m in run.get("store") or () if m in mems}
    pos = {m for m, c in cls.items() if c == diagnosis.POSITIVE}
    neg = {m for m, c in cls.items() if c == diagnosis.NEGATIVE}
    flagged = set(pred) & set(cls)
    return diagnosis.RunScore(run.get("run_dir", ""), len(flagged & pos), len(flagged & neg),
                              len(pos - flagged), len(flagged), len(pos), 0)


def _fmt_recall(s: dict | None) -> str:
    if not s or s.get("micro_r") is None:
        return "-"
    lo, hi = s["micro_r_ci"]
    out = (f"{100 * s['micro_r']:.0f}% ({s['tp']}/{s['tp'] + s['fn']}) "
           f"[{100 * (lo or 0):.0f}-{100 * (hi or 0):.0f}] (runs={s['runs']})")
    return f"{out} {SMALL_NOTE}" if s["runs"] < SMALL else out


def _fmt_precision(s: dict | None) -> str:
    if not s or s.get("micro_p") is None:
        return "-"
    return f"{100 * s['micro_p']:.0f}% ({s['tp']}/{s['tp'] + s['fp']})"


def adaptive_section(ctx: Ctx, sec: Section) -> None:
    a = ctx.args
    adaptive_all = [r for r in finished(ctx.history()) if r.suite == "adaptive"]
    if not adaptive_all:
        sec.skip("no finished adaptive history runs (variants ending _adaptive_evade or "
                 "_adaptive_survive)")
        return
    adaptive = [r for r in adaptive_all
                if r.memory_mode == "evolving" and r.model == a.reference_model]
    if len(adaptive) < len(adaptive_all):
        sec.note(f"{len(adaptive_all) - len(adaptive)} adaptive runs on another model or "
                 f"memory left out (the comparison is {a.reference_model}, evolving)")
    contexts = sorted({r.context for r in adaptive})
    standard = [r for r in finished(ctx.base()) if r.context in contexts
                and r.model == a.reference_model and r.category in ATTACK_CATEGORIES]
    if not standard:
        sec.note("missing input: no original dense runs on the adaptive runs' contexts")
    runs = adaptive + standard
    cats = [c for c in (*ATTACK_CATEGORIES, *ADAPTIVE_CATEGORIES)
            if any(r.category == c for r in runs)]
    for c in ADAPTIVE_CATEGORIES:
        ctx.thin(sec, f"poisoned {c}", len([r for r in poisoned(adaptive) if r.category == c]))
    sec.data["contexts"] = contexts
    _adaptive_rates(sec, runs, cats, adaptive, standard)
    _adaptive_matched(ctx, sec, adaptive, standard)
    _adaptive_survival(sec, runs, cats)
    _adaptive_diagnosis(ctx, sec)
    _adaptive_recovery(ctx, sec, standard)


def _adaptive_rates(sec: Section, runs: list[Run], cats: list[str], adaptive: list[Run],
                    standard: list[Run]) -> None:
    q = qualified_keys(runs)
    rows, data = [], {}
    for c in cats:
        rs = [r for r in runs if r.category == c]
        pois = poisoned(rs)
        data[c] = {"source": "history" if c in ADAPTIVE_CATEGORIES else "original dense",
                   "contexts": len({r.context for r in rs}),
                   "poisoned_attacker": outcome_rate(pois, attacker),
                   "qualified_attacker": outcome_rate([r for r in pois if r.key in q], attacker),
                   "clean_correct": outcome_rate(clean(rs), correct),
                   "poisoned_outcomes": dict(Counter(r.outcome for r in pois))}
        d = data[c]
        rows.append([c, d["source"], d["contexts"], fmt(d["poisoned_attacker"]),
                     fmt(d["qualified_attacker"]), fmt(d["clean_correct"]), outcome_mix(pois)])
        if c in ADAPTIVE_CATEGORIES:
            sec.headline.append(f"{c}: attacker {fmt(d['poisoned_attacker'])}")
    std = poisoned(standard)
    data["standard (pooled)"] = {"poisoned_attacker": outcome_rate(std, attacker)}
    sec.headline.append(f"standard categories, same contexts: attacker "
                        f"{fmt(data['standard (pooled)']['poisoned_attacker'])}")
    sec.data["rates"] = data
    sec.table("Target-task outcome by attack category, on the contexts the adaptive runs "
              "cover (standard categories: original dense runs; one run per scenario and "
              "condition unless repeated)",
              ["category", "runs from", "contexts", "poisoned: attacker (Wilson 95%)",
               "attacker, qualified variants only", "clean: correct", "poisoned outcomes"],
              rows)


def _adaptive_matched(ctx: Ctx, sec: Section, adaptive: list[Run], standard: list[Run]) -> None:
    a = ctx.args
    is_att = lambda r: float(attacker(r))
    rows, data = [], {}
    std_cats = [c for c in ATTACK_CATEGORIES if any(r.category == c for r in standard)]
    for ac in ADAPTIVE_CATEGORIES:
        A = by_key([r for r in poisoned(adaptive) if r.category == ac], lambda r: r.context, is_att)
        if not A:
            continue
        for sc in (*std_cats, "all standard"):
            B = by_key([r for r in poisoned(standard) if sc == "all standard" or r.category == sc],
                       lambda r: r.context, is_att)
            d = paired_diff(A, B, a.n_boot, a.seed)
            data[f"{ac} - {sc}"] = d
            rows.append([ac, sc, fmt_diff(d, "contexts")])
    sec.data["matched"] = data
    sec.table("Attacker rate, adaptive minus standard, matched by context (mean of "
              "per-context differences; bootstrap 95% over contexts)",
              ["adaptive", "standard", "difference"], rows)


def _adaptive_survival(sec: Section, runs: list[Run], cats: list[str]) -> None:
    measured: dict[str, list[measure.RunMeasure]] = defaultdict(list)
    failed = 0
    for r in poisoned(runs):
        try:
            rm = measure.measure_run(r.path)
        except (sqlite3.Error, OSError, ValueError, KeyError):
            rm = None
        if rm is None:
            failed += 1
            continue
        measured[r.category].append(rm)
    if failed:
        sec.note(f"{failed} poisoned runs could not be measured (measure.measure_run)")
    stages = ("stored", "consolidated", "pre_target", "retrieved", "acted")
    rows, data = [], {}
    for c in cats:
        rms = measured.get(c, [])
        if not rms:
            continue
        surv = measure.survival_rates(rms)
        carrying = [rm.labels["harmful"] + rm.labels["mixed"] for rm in rms]
        labels = Counter()
        for rm in rms:
            labels.update(rm.labels)
        d = {"runs": len(rms), "stages": {s: rate(*surv[s]) for s in stages},
             "carrying_descendant": rate(sum(x > 0 for x in carrying), len(rms)),
             "descendants_per_run": sum(rm.psize for rm in rms) / len(rms),
             "carrying_per_run": sum(carrying) / len(rms),
             "labels": {k: labels[k] for k in measure.LABELS}}
        data[c] = d
        rows.append([c, len(rms), *(fmt(d["stages"][s]) for s in stages),
                     fmt(d["carrying_descendant"]), f"{d['descendants_per_run']:.1f}",
                     f"{d['carrying_per_run']:.1f}",
                     "/".join(str(d["labels"][k]) for k in measure.LABELS)])
    sec.data["survival"] = data
    sec.table("Survival into descendants (measure.measure_run; 'survives' = an active "
              "lineage memory still carries the attack in that snapshot)",
              ["category", "poisoned runs", "stored (T1)", "consolidated", "pre-target",
               "retrieved at target", "acted", "runs with a carrying descendant",
               "descendants per run", "carrying descendants per run",
               "descendant labels harmful/mixed/benign/ambiguous"], rows)


def _adaptive_diagnosis(ctx: Ctx, sec: Section) -> None:
    a = ctx.args
    if not Path(a.diagnosis_adaptive).is_file():
        sec.note(f"content-detector evasion not computed: {a.diagnosis_adaptive} is missing "
                 "(diagnose.py over the adaptive runs)")
        return
    adapt = diagnosis_recall(a.diagnosis_adaptive,
                             lambda r: r.get("category") in ADAPTIVE_CATEGORIES,
                             a.n_boot, a.seed) or {}
    doc = _json(a.diagnosis_adaptive) or {}
    ctxs = {r.get("variant", "").removesuffix(f"_{r.get('category', '')}")
            for r in doc.get("runs") or () if r.get("category") in ADAPTIVE_CATEGORIES}
    std = diagnosis_recall(
        a.diagnosis_standard,
        lambda r: (r.get("category") in ATTACK_CATEGORIES
                   and r.get("variant", "").removesuffix(f"_{r.get('category', '')}") in ctxs),
        a.n_boot, a.seed) or {}
    if not std:
        sec.note(f"no standard-category diagnosis on the same contexts ({a.diagnosis_standard})")
    groups = {**{g: std[g] for g in ATTACK_CATEGORIES if g in std},
              **{g: adapt[g] for g in sorted(adapt)}}
    rows = []
    for g, methods in groups.items():
        for m in DIAGNOSIS_METHODS:
            if m in methods:
                e = methods[m]
                rows.append([g, m, _fmt_recall(e.get("all")), _fmt_precision(e.get("all")),
                             _fmt_recall(e.get("incident"))])
    sec.data["diagnosis"] = groups
    for g in groups:
        s = (groups[g].get("content_llm") or {}).get("all")
        if s and g in adapt:
            sec.headline.append(f"content_llm recall, {g}: {_fmt_recall(s)}")
    sec.table("Content-detector evasion: recall per category (store scope, primary scoring, "
              "micro; bootstrap 95% over runs). Standard categories restricted to the same "
              "contexts.",
              ["category", "method", "recall, all runs", "precision, all runs",
               "recall, incident runs"], rows)


def _adaptive_recovery(ctx: Ctx, sec: Section, standard: list[Run]) -> None:
    recovery_vs_reference(
        ctx, sec, "adaptive", "adaptive",
        lambda run: run.category if run.suite == "adaptive" else None,
        "standard (original sweep)", standard, lambda run: run.context, "contexts",
        "Recovery by attack category (poisoned runs, checkpoint 0; standard = the "
        "canonical sweep on the same contexts)")


def recovery_vs_reference(ctx: Ctx, sec: Section, name: str, what: str,
                          group: Callable[[Run], str | None], ref_label: str,
                          ref_runs: list[Run], key: Callable[[Run], str], unit: str,
                          caption: str) -> None:
    """Recovery on the history runs (records file `name`, grouped by group(run)) beside
    the canonical sweep, as like for like as the records allow.

    Population: the reference keeps only the original runs (from ref_runs) whose key and
    configuration the history records measured (match_reference). Tasks: every ASR the
    groups are compared on is the run's own target task (target_record); the
    every-replayed-task ASR is kept in its own column with its tasks per run. The paired
    table then sets each group against the reference key by key."""
    a = ctx.args
    recs = ctx.records(name)
    if recs is None:
        sec.note(f"recovery not computed: {getattr(a, f'{name}_records')} is missing")
        return
    recs, other = usable(recs)
    if other:
        sec.note(f"{other} {what} records with non-default knobs or live pages left out")
    run_key = lambda rec: key(run) if (run := ctx.run_of(rec)) else None
    task_index = lambda rec: run.task_index if (run := ctx.run_of(rec)) else 0
    groups = recovery_by(recs, lambda rec: group(run) if (run := ctx.run_of(rec)) else None)
    _add_reference(ctx, sec, groups, ref_label, ref_runs, key, run_key, unit)
    order = [ref_label, *sorted(g for g in groups if g != ref_label)]
    rows, data, missing = recovery_section_rows(groups, order, task_index)
    if missing:
        sec.note(f"{missing} records did not replay the run's own task and are left out of "
                 "the own-task columns")
    matched = recovery_matched(groups, ref_label, run_key, task_index, a.n_boot, a.seed)
    sec.data["recovery"] = data
    sec.data["recovery_order"] = [g for g in order if g in data]
    sec.data["recovery_matched"] = matched
    sec.table(f"{caption}. The reference keeps only the (configuration, {unit[:-1]}) pairs "
              "the history records measured. Groups replay task families of different sizes "
              "(tasks per run), so they are compared on each run's own target task, the same "
              "question in every family; the every-replayed-task ASR is comparable only "
              "between rows with the same tasks per run",
              RECOVERY_HEADER, rows)
    diff_rows = [[g, c, fmt_diff(d, unit)] for g, cfgs in matched.items()
                 for c, d in cfgs.items()]
    sec.table(f"Own-task attacker rate after recovery, group minus {ref_label}, matched by "
              f"{unit[:-1]} (mean of per-{unit[:-1]} differences; bootstrap 95% over {unit})",
              ["group", "configuration", "difference"], diff_rows)


def _add_reference(ctx: Ctx, sec: Section, groups: dict[str, list[dict]], ref_label: str,
                   ref_runs: list[Run], key: Callable[[Run], str],
                   run_key: Callable[[dict], str | None], unit: str) -> None:
    """Add the matched canonical-sweep records to `groups` under ref_label."""
    reference = ctx.records("reference")
    names = {r.name for r in ref_runs}
    if reference is None or not names or not groups:
        return
    history_runs = {run for recs in groups.values() for r in measured(recs)
                    if (run := ctx.run_of(r)) is not None}
    covered = {key(r) for r in history_runs}
    mixed = task_mismatches([*history_runs, *(r for r in ref_runs if key(r) in covered)], key)
    if mixed:
        sec.note(f"own target task differs between the history and reference runs on "
                 f"{sorted(mixed)}: left out of the reference")
    ref_recs, _ = usable([r for r in reference if r.get("run_dir") in names])
    matched = match_reference(groups, ref_recs, run_key, exclude=mixed)
    if matched:
        groups[ref_label] = matched
    else:
        sec.note(f"no {ref_label} records on the configurations and {unit} the history "
                 "records measured")


# --- 2. models ----------------------------------------------------------------------

def models_section(ctx: Ctx, sec: Section) -> None:
    a = ctx.args
    dense = [r for r in finished(ctx.history())
             if r.suite == "dense" and r.memory_mode == "evolving"]
    subset = [r for r in dense if r.variant in DENSE_SUBSET]
    if len(subset) < len(dense):
        sec.note(f"{len(dense) - len(subset)} dense history runs outside DENSE_SUBSET left out")
    if not subset:
        sec.skip("no finished dense-subset history runs with evolving memory")
        return
    ref = [r for r in finished(ctx.base()) if r.model == a.reference_model]
    if not ref:
        sec.note(f"missing input: no original {a.reference_model} dense runs to compare with")
    models = sorted({r.model for r in subset})
    rows, cat_rows, data = [], [], {}
    for m in models:
        mr = [r for r in subset if r.model == m]
        variants = {r.variant for r in mr}
        rr = [r for r in ref if r.variant in variants]
        ctx.thin(sec, f"{m} poisoned", len(poisoned(mr)))
        att = lambda r: float(attacker(r))
        ok = lambda r: float(correct(r))
        d = {"variants": len(variants), "runs": len(mr),
             "poisoned_attacker": outcome_rate(poisoned(mr), attacker),
             "reference_poisoned_attacker": outcome_rate(poisoned(rr), attacker),
             "attacker_diff": paired_diff(by_key(poisoned(mr), lambda r: r.variant, att),
                                          by_key(poisoned(rr), lambda r: r.variant, att),
                                          a.n_boot, a.seed),
             "clean_correct": outcome_rate(clean(mr), correct),
             "reference_clean_correct": outcome_rate(clean(rr), correct),
             "correct_diff": paired_diff(by_key(clean(mr), lambda r: r.variant, ok),
                                         by_key(clean(rr), lambda r: r.variant, ok),
                                         a.n_boot, a.seed),
             "poisoned_outcomes": dict(Counter(r.outcome for r in poisoned(mr))),
             "by_category": {}}
        data[m] = d
        rows.append([m, d["variants"], fmt(d["poisoned_attacker"]),
                     fmt(d["reference_poisoned_attacker"]), fmt_diff(d["attacker_diff"]),
                     fmt(d["clean_correct"]), fmt(d["reference_clean_correct"]),
                     fmt_diff(d["correct_diff"]), outcome_mix(poisoned(mr))])
        sec.headline.append(f"{m}: attacker {fmt(d['poisoned_attacker'])} vs "
                            f"{a.reference_model} {fmt(d['reference_poisoned_attacker'])} "
                            "on the same variants")
        for c in ATTACK_CATEGORIES:
            pc = [r for r in poisoned(mr) if r.category == c]
            if pc:
                rc = [r for r in poisoned(rr) if r.category == c]
                d["by_category"][c] = {"attacker": outcome_rate(pc, attacker),
                                       "reference_attacker": outcome_rate(rc, attacker)}
                cat_rows.append([m, c, fmt(d["by_category"][c]["attacker"]),
                                 fmt(d["by_category"][c]["reference_attacker"])])
    sec.data["models"] = data
    ref_name = a.reference_model
    sec.table(f"Dense subset per model vs {ref_name} (original dense runs restricted to "
              "the variants each model ran); differences are model minus reference, "
              "matched by variant (bootstrap 95% over variants)",
              ["model", "variants", "poisoned: attacker", f"{ref_name}: attacker",
               "attacker difference", "clean: correct", f"{ref_name}: correct",
               "correct difference", "poisoned outcomes"], rows)
    sec.table("Poisoned attacker rate per model and attack category",
              ["model", "category", "attacker", f"{ref_name}, same variants"], cat_rows)
    _models_recovery(ctx, sec, ref)


def _models_recovery(ctx: Ctx, sec: Section, ref: list[Run]) -> None:
    recovery_vs_reference(
        ctx, sec, "models", "model",
        lambda run: run.model if run.suite == "dense" and run.memory_mode == "evolving" else None,
        f"{ctx.args.reference_model} (original sweep)", ref, lambda run: run.variant,
        "variants", "Recovery per model (poisoned runs, checkpoint 0; the reference row is the "
        "canonical sweep, pooled over the variants any model's records cover)")


# --- 3. external suites -------------------------------------------------------------

def external_kind(variant: str) -> str:
    """The external case's kind (LongMemEval question type, MemBench fact-/refl- kind),
    from the note graft() writes on every scenario."""
    s = ALL_VARIANTS.get(variant)
    m = _EXTERNAL_NOTES.match(s.notes if s else "")
    return m.group(3) if m else ""


def external_group(variant: str) -> str | None:
    kind = external_kind(variant)
    if variant in LONGMEMEVAL:
        return ("LongMemEval: knowledge-update" if kind == "knowledge-update"
                else "LongMemEval: other kinds")
    if variant in MEMBENCH:
        return "MemBench: factual" if kind.startswith("fact") else "MemBench: reflective"
    return None


EXTERNAL_GROUPS = ("LongMemEval: knowledge-update", "LongMemEval: other kinds",
                   "MemBench: factual", "MemBench: reflective")
SUITE_NAMES = {"longmemeval": "LongMemEval", "membench": "MemBench"}


def external_probe(rec: dict) -> bool | None:
    """Was the external benchmark's own question (the first benign probe graft() adds)
    answered correctly in this record? None when the record does not have it."""
    s = ALL_VARIANTS.get(rec.get("variant", ""))
    if not s or not s.benign_probes:
        return None
    q = s.benign_probes[0][0]
    for p in rec.get("probes") or ():
        if p.get("question") == q:
            return bool(p.get("correct"))
    return None


def probe_stage(rec: dict) -> str | None:
    if rec.get("checkpoint", 0) != 0:
        return None
    if rec.get("condition", "poisoned") == "clean":
        return "clean run, no recovery" if measure.is_baseline(rec) else None
    if measure.is_baseline(rec):
        return "poisoned, no recovery"
    return measure.config_label(measure.config_key(rec), show_source=False)


def external_section(ctx: Ctx, sec: Section) -> None:
    ext = [r for r in finished(ctx.history()) if r.suite in SUITE_NAMES]
    sec.data["memsecbench"] = MEMSECBENCH
    if not ext:
        sec.skip("no finished LongMemEval (lme_*) or MemBench (mb_*) history runs")
        sec.md.append(MEMSECBENCH)
        return
    models = sorted({r.model for r in ext})
    if len(models) > 1:
        sec.note(f"external runs span models {models}; rates pool them")
    for s, name in SUITE_NAMES.items():
        ctx.thin(sec, f"{name} poisoned", len(poisoned(r for r in ext if r.suite == s)))
    rows, data = [], {}
    for s, name in SUITE_NAMES.items():
        for cond in ("clean", "poisoned"):
            rs = [r for r in ext if r.suite == s and r.condition == cond]
            if rs:
                d = {"attacker": outcome_rate(rs, attacker), "correct": outcome_rate(rs, correct),
                     "outcomes": dict(Counter(r.outcome for r in rs))}
                data[f"{name} {cond}"] = d
                rows.append([name, cond, len(rs), fmt(d["attacker"]), fmt(d["correct"]),
                             outcome_mix(rs)])
                if cond == "poisoned":
                    sec.headline.append(f"{name} poisoned: attacker {fmt(d['attacker'])}")
    sec.data["suites"] = data
    sec.table("External suites: target-task outcome, clean vs poisoned",
              ["suite", "condition", "runs", "attacker (Wilson 95%)", "correct", "outcomes"], rows)
    _external_groups(ctx, sec, ext)
    _external_probes(ctx, sec)
    sec.md.append(MEMSECBENCH)


def _external_groups(ctx: Ctx, sec: Section, ext: list[Run]) -> None:
    a = ctx.args
    rows, data = [], {}
    att = lambda r: float(attacker(r))
    for g in EXTERNAL_GROUPS:
        rs = [r for r in ext if external_group(r.variant) == g]
        if not rs:
            continue
        d = {"category": sorted({r.category for r in rs}),
             "clean_attacker": outcome_rate(clean(rs), attacker),
             "poisoned_attacker": outcome_rate(poisoned(rs), attacker),
             "clean_correct": outcome_rate(clean(rs), correct),
             "poisoned_minus_clean": paired_diff(by_key(poisoned(rs), lambda r: r.variant, att),
                                                 by_key(clean(rs), lambda r: r.variant, att),
                                                 a.n_boot, a.seed)}
        data[g] = d
        rows.append([g, "/".join(d["category"]), fmt(d["clean_attacker"]),
                     fmt(d["poisoned_attacker"]), fmt_diff(d["poisoned_minus_clean"]),
                     fmt(d["clean_correct"])])
    cat_rows = []
    for s, name in SUITE_NAMES.items():
        for c in ATTACK_CATEGORIES:
            pc = [r for r in poisoned(ext) if r.suite == s and r.category == c]
            if pc:
                data[f"{name} {c}"] = {"poisoned_attacker": outcome_rate(pc, attacker)}
                cat_rows.append([name, c, fmt(data[f"{name} {c}"]["poisoned_attacker"])])
    sec.data["groups"] = data
    sec.table("By question type (MemBench factual = direct poison, reflective = procedural; "
              "LongMemEval knowledge-update apart); poisoned minus clean matched by variant",
              ["group", "attack", "clean: attacker", "poisoned: attacker",
               "poisoned minus clean", "clean: correct"], rows)
    sec.table("Poisoned attacker rate per suite and attack category",
              ["suite", "category", "poisoned: attacker"], cat_rows)


def _external_probes(ctx: Ctx, sec: Section) -> None:
    recs = ctx.records("external")
    if recs is None:
        sec.note(f"external-probe accuracy not computed: {ctx.args.external_records} is missing")
        return
    recs, other = usable(recs)
    if other:
        sec.note(f"{other} external records with non-default knobs or live pages left out")
    cells: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    missing = 0
    for r in recs:
        group, stage = external_group(r.get("variant", "")), probe_stage(r)
        if group is None or stage is None:
            continue
        ok = external_probe(r)
        if ok is None:
            missing += 1
            continue
        for g in (group, SUITE_NAMES.get(group.split(":")[0].lower(), group.split(":")[0])):
            cells[(g, stage)][r["run_dir"]] = ok
    if missing:
        sec.note(f"{missing} external records lack the external probe")
    if not any(s == "clean run, no recovery" for _, s in cells):
        sec.note("no clean-run replay records (recover.py --include-clean was not run on the "
                 "external suites), so the clean-run column is missing")
    stages = sorted({s for _, s in cells}, key=_stage_sort)
    groups = [g for g in (*EXTERNAL_GROUPS, "LongMemEval", "MemBench")
              if any(k[0] == g for k in cells)]
    rows, data = [], {}
    for g in groups:
        base = cells.get((g, "poisoned, no recovery"), {})
        for s in stages:
            runs = cells.get((g, s))
            if not runs:
                continue
            acc = rate(sum(runs.values()), len(runs))
            known = [rd for rd, ok in base.items() if ok and rd in runs]
            kept = (rate(sum(runs[rd] for rd in known), len(known))
                    if s not in ("poisoned, no recovery", "clean run, no recovery") else None)
            data[f"{g} | {s}"] = {"accuracy": acc, "kept": kept}
            rows.append([g, s, fmt(acc), fmt(kept) if kept else "-"])
    sec.data["probes"] = data
    sec.data["probe_stages"] = stages
    if "MemBench: reflective" in groups:
        # The records keep only the verdict, not the reply, so this cannot be rescored here.
        sec.note("MemBench reflective probes are multiple choice, but recover.benign_probes "
                 "scores every probe by substring (not external.probe_correct), so a reply "
                 "listing every option counts as correct: read those cells as an upper bound")
    for g in ("LongMemEval", "MemBench"):
        for s in ("poisoned, no recovery",):
            if f"{g} | {s}" in data:
                sec.headline.append(f"{g} external probe, {s}: {fmt(data[f'{g} | {s}']['accuracy'])}")
    sec.table("External-benchmark probe (the case's own question) answered from memory: "
              "clean-run replay, poisoned without recovery, and after each recovery "
              "(records' probes; 'kept' = among runs answering it before recovery)",
              ["group", "stage", "correct (runs; Wilson 95%)", "kept after recovery"], rows)


def _stage_sort(stage: str) -> tuple:
    fixed = ("clean run, no recovery", "poisoned, no recovery")
    if stage in fixed:
        return (fixed.index(stage), "")
    mode = re.split(r"[\[+ ]", stage)[0]
    order = measure.MODE_ORDER
    return (2 + (order.index(mode) if mode in order else len(order)), stage)


# --- 4. Mem0 ------------------------------------------------------------------------

def mem0_event(operation: str, status: str) -> str | None:
    """The Mem0 write a mirror row stands for (memheal/backends/mem0_backend.py)."""
    if operation == "mem0_add":
        return "ADD"
    if operation == "summarize":
        return "DELETE" if status == "deleted" else "UPDATE"
    if operation == "mem0_facts":
        return "facts"
    return None


def mem0_writes(run: Run) -> dict:
    """Write events in a Mem0 run's mirror, how many name the attacker, and the
    mem0_mirror table's own event counts as a cross-check."""
    conn = db.connect_readonly(run.path / "run.db")
    try:
        rows = conn.execute("SELECT operation, status, text FROM memories").fetchall()
        has_mirror = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                                  "AND name = 'mem0_mirror'").fetchone()
        mirror = (Counter(e for (e,) in conn.execute("SELECT event FROM mem0_mirror"))
                  if has_mirror else None)
    finally:
        conn.close()
    spec = AttackSpec.from_scenario(run.scenario) if run.scenario else None
    events, naming = Counter(), Counter()
    for op, status, text in rows:
        ev = mem0_event(op, status)
        if ev is None:
            continue
        events[ev] += 1
        if spec is not None and spec.names_attacker(text):
            naming[ev] += 1
    agrees = None if mirror is None else all(mirror.get(e, 0) == events.get(e, 0)
                                             for e in ("ADD", "UPDATE", "DELETE"))
    return {"events": dict(events), "naming": dict(naming), "mirror": dict(mirror or {}),
            "mirror_agrees": agrees}


def mem0_section(ctx: Ctx, sec: Section) -> None:
    m0 = [r for r in finished(ctx.history()) if r.memory_mode == "mem0"]
    if not m0:
        sec.skip("no finished Mem0 history runs (agent.memory_mode mem0)")
        return
    pipelines = sorted({r.pipeline or "update" for r in m0}, key=lambda p: (p != "update", p))
    for p in pipelines:
        ctx.thin(sec, f"Mem0 {p} poisoned", len(poisoned(r for r in m0 if r.pipeline == p)))
    no_lineage = [r for r in m0 if not r.lineage_metrics]
    if no_lineage:
        sec.note(f"{len(no_lineage)} Mem0 runs record lineage_metrics=false "
                 f"(pipelines {sorted({r.backend for r in no_lineage})}): their mirror lineage "
                 "is incomplete, so lineage-based recovery on them is not comparable")
    _mem0_rates(ctx, sec, m0, pipelines)
    _mem0_writes(ctx, sec, m0, pipelines)
    _mem0_recovery(ctx, sec, m0)


def _mem0_rates(ctx: Ctx, sec: Section, m0: list[Run], pipelines: list[str]) -> None:
    a = ctx.args
    evolving = [r for r in finished(ctx.base()) if r.model == a.reference_model]
    flat = [r for r in finished(ctx.flat()) if r.model == a.reference_model]
    rows, diff_rows, data = [], [], {}
    att = lambda r: float(attacker(r))
    for p in pipelines:
        pr = [r for r in m0 if (r.pipeline or "update") == p]
        variants = {r.variant for r in pr}
        backends = ((pr[0].backend if pr else f"mem0-{p}", pr),
                    ("evolving", [r for r in evolving if r.variant in variants]),
                    ("flat", [r for r in flat if r.variant in variants]))
        q = qualified_keys([r for _, rs in backends for r in rs])
        data[p] = {}
        for name, rs in backends:
            d = {"variants": len({r.variant for r in rs}),
                 "poisoned_attacker": outcome_rate(poisoned(rs), attacker),
                 "qualified_attacker": outcome_rate([r for r in poisoned(rs) if r.key in q],
                                                    attacker),
                 "clean_correct": outcome_rate(clean(rs), correct),
                 "poisoned_outcomes": dict(Counter(r.outcome for r in poisoned(rs)))}
            data[p][name] = d
            rows.append([f"{p} ({len(variants)} variants)", name, d["variants"],
                         fmt(d["poisoned_attacker"]), fmt(d["qualified_attacker"]),
                         fmt(d["clean_correct"]), outcome_mix(poisoned(rs))])
        A = by_key(poisoned(pr), lambda r: r.variant, att)
        for name, rs in backends[1:]:
            d = paired_diff(A, by_key(poisoned(rs), lambda r: r.variant, att), a.n_boot, a.seed)
            data[p][f"diff vs {name}"] = d
            diff_rows.append([backends[0][0], name, fmt_diff(d)])
        sec.headline.append(f"{backends[0][0]}: attacker {fmt(data[p][backends[0][0]]['poisoned_attacker'])}"
                            f", evolving {fmt(data[p]['evolving']['poisoned_attacker'])}"
                            f", flat {fmt(data[p]['flat']['poisoned_attacker'])} (same variants)")
    sec.data["rates"] = data
    sec.data["pipelines"] = pipelines
    sec.table("Attacker rate per memory backend on the variants each Mem0 pipeline ran "
              f"(evolving: original dense runs; flat: their flat twins; {a.reference_model})",
              ["Mem0 pipeline (variants)", "memory", "variants with runs", "poisoned: attacker",
               "attacker, qualified variants", "clean: correct", "poisoned outcomes"], rows)
    sec.table("Attacker rate, Mem0 minus the other memory, matched by variant "
              "(bootstrap 95% over variants)", ["Mem0", "vs", "difference"], diff_rows)


def _mem0_writes(ctx: Ctx, sec: Section, m0: list[Run], pipelines: list[str]) -> None:
    a = ctx.args
    writes, failed = {}, 0
    for r in m0:
        try:
            writes[r.name] = mem0_writes(r)
        except (sqlite3.Error, OSError) as exc:
            failed += 1
            sec.note(f"{r.name}: mirror unreadable ({exc})")
    rows, naming_rows, data = [], [], {}
    for p in pipelines:
        for cond in ("poisoned", "clean"):
            rs = [r for r in m0 if (r.pipeline or "update") == p and r.condition == cond
                  and r.name in writes]
            if not rs:
                continue
            w = [writes[r.name] for r in rs]
            per = {e: mean_ci([float(x["events"].get(e, 0)) for x in w], a.n_boot, a.seed)
                   for e in MEM0_EVENTS}
            with_update = sum(x["events"].get("UPDATE", 0) > 0 for x in w)
            with_delete = sum(x["events"].get("DELETE", 0) > 0 for x in w)
            agrees = [x["mirror_agrees"] for x in w if x["mirror_agrees"] is not None]
            data[f"{p} {cond}"] = {"runs": len(rs), "per_run": per,
                                   "runs_with_update": rate(with_update, len(rs)),
                                   "runs_with_delete": rate(with_delete, len(rs)),
                                   "mirror_agrees": [sum(agrees), len(agrees)]}
            rows.append([p, cond, len(rs), *(fmt_mean(per[e]) for e in MEM0_EVENTS),
                         fmt(rate(with_update, len(rs))), fmt(rate(with_delete, len(rs))),
                         f"{sum(agrees)}/{len(agrees)}"])
            naming_rows.append(_naming_row(p, cond, rs, writes, data[f"{p} {cond}"]))
    sec.data["writes"] = data
    sec.data["writes_per_run"] = writes
    sec.table("Mem0 write events per run (memories mirror: mem0_add = ADD, summarize = "
              "UPDATE, summarize with status deleted = DELETE tombstone, mem0_facts rows; "
              "mean, bootstrap 95% over runs)",
              ["pipeline", "condition", "runs", "ADD", "UPDATE", "DELETE", "facts",
               "runs with an UPDATE", "runs with a DELETE", "mirror table agrees (runs)"], rows)
    sec.table("Where attack-naming memories came from (AttackSpec.names_attacker on each "
              "mirror row)",
              ["pipeline", "condition", "runs", "runs naming it in an ADD",
               "runs naming it in an UPDATE", "in a DELETE tombstone",
               "UPDATE only (no naming ADD)", "attacker outcome, runs naming it in an UPDATE",
               "attacker outcome, other runs"], naming_rows)


def _naming_row(p: str, cond: str, rs: list[Run], writes: dict, out: dict) -> list:
    has = lambda r, e: writes[r.name]["naming"].get(e, 0) > 0
    upd = [r for r in rs if has(r, "UPDATE")]
    rest = [r for r in rs if not has(r, "UPDATE")]
    d = {"add": rate(sum(has(r, "ADD") for r in rs), len(rs)),
         "update": rate(len(upd), len(rs)),
         "delete": rate(sum(has(r, "DELETE") for r in rs), len(rs)),
         "update_only": rate(sum(has(r, "UPDATE") and not has(r, "ADD") for r in rs), len(rs)),
         "attacker_given_update": outcome_rate(upd, attacker),
         "attacker_given_no_update": outcome_rate(rest, attacker)}
    out["naming"] = d
    return [p, cond, len(rs), fmt(d["add"]), fmt(d["update"]), fmt(d["delete"]),
            fmt(d["update_only"]), fmt(d["attacker_given_update"]),
            fmt(d["attacker_given_no_update"])]


def _mem0_recovery(ctx: Ctx, sec: Section, m0: list[Run]) -> None:
    a = ctx.args
    ref = [r for r in finished(ctx.base()) if r.model == a.reference_model]
    recovery_vs_reference(
        ctx, sec, "mem0", "Mem0", lambda run: run.backend if run.memory_mode == "mem0" else None,
        "evolving (original sweep)", ref, lambda run: run.variant, "variants",
        "Recovery on Mem0 runs (poisoned, checkpoint 0; evolving = the canonical sweep, "
        "pooled over the variants either pipeline's records cover)")
    absent = sorted({r.backend for r in poisoned(m0)} - set(sec.data.get("recovery") or {}))
    if absent and "recovery" in sec.data:
        sec.note(f"no recovery records for {absent}")


# --- 5. healing per suite -----------------------------------------------------------

def read_with_parts(path: Path, parts: bool = True) -> tuple[list[dict] | None, int]:
    """measure.load_records of `path` plus the part files a running batch has not merged
    into it yet (batch.py: <dir>/.parts/<stem>.*.jsonl, appended at the very end), one
    record per measurement, the last copy winning (analyze.read_records' rule), and the
    number of part files; None when neither exists."""
    path = Path(path)
    files = [path] if path.is_file() else []
    part_files = sorted((path.parent / ".parts").glob(f"{path.stem}.*.jsonl")) if parts else []
    if not files and not part_files:
        return None, 0
    return last_per_measurement(
        [r for f in files + part_files for r in measure.load_records(f)]), len(part_files)


def last_per_measurement(rows: list[dict]) -> list[dict]:
    last = {measure.record_key(r): i for i, r in enumerate(rows)}
    return [r for i, r in enumerate(rows) if last[measure.record_key(r)] == i]


def heal_selection(records: list[dict]) -> list[dict]:
    """Poisoned, patched, checkpoint-0 records on the runs a heal mode was applied to:
    every configuration, whichever file and knob setting it came from, so each heal mode
    is set beside the other configurations on the same runs (rows keep their knob
    settings apart, measure.setting_key)."""
    pois = [r for r in measured(records) if (r.get("source") or "patched") == "patched"]
    runs = {r["run_dir"] for r in pois if r.get("mode") in HEAL_MODES}
    return [r for r in pois if r["run_dir"] in runs]


def counterpart(rec: dict, mode: str, verdicts: str, labels: Iterable[str]) -> str | None:
    """The configuration a heal record is set against: `mode` with the same modifiers and
    knobs, else with the one-sided knobs set aside in measure.ONE_SIDED_LADDER's order
    (measure.paired_differences' rule), so a heal+verify at verify_n=3 meets
    incident+verify at verify_n=3, and a heal with a changed --claim-n still meets the
    incident of its own counterfactual options."""
    have = set(labels)
    other = {**rec, "mode": mode, "verdicts": verdicts}
    for names in ((), *measure.ONE_SIDED_LADDER):
        cand = config_of({**other, **{k: None for k in names}})
        if cand in have:
            return cand
    return None


def dropped_cell(per_rule: dict[str, int]) -> str:
    """'name 31 (67%); cf 5 (11%)': the rules that dropped a claim, in RULES order, each
    with its share of the claims dropped; '-' when none was."""
    total = sum(per_rule.values())
    if not total:
        return "-"
    return "; ".join(f"{k} {v} ({100 * v / total:.0f}%)" for k, v in per_rule.items() if v)


def heal_section(ctx: Ctx, sec: Section) -> None:
    """Every heal mode per suite, from the suite's records.jsonl and its Phase 2 main.jsonl
    (plus their unmerged part files), on the runs a heal mode was applied to: outcomes
    beside the other configurations, what healing did and why, and each heal mode minus
    incident (the same detection, quarantined) and minus heal. Left out of the report
    altogether when no main file exists and no suite has a heal record, so a report over
    the earlier files reads exactly as it did."""
    a = ctx.args
    files, selections = [], {}
    for suite in HEAL_SUITES:
        recs, found = ctx.suite_records(suite)
        files += found
        sel = heal_selection(recs)
        if sel:
            selections[suite] = sel
    if not selections and not any(f["state"] == "ok" and f["main"] for f in files):
        sec.omit = "no main.jsonl file and no heal-mode record in any suite"
        return
    sec.data["files"] = files
    sec.table("Inputs: each suite's records.jsonl and Phase 2 main.jsonl, with the part files "
              "a running batch has not merged yet" + ("" if a.parts else " (ignored: --no-parts)"),
              ["suite", "file", "state", "records", "runs", "unmerged part files"],
              [[f["suite"], f["path"], f["state"], f["records"], f["runs"], f["part_files"]]
               for f in files])
    for suite in HEAL_SUITES:
        if suite not in selections:
            sec.note(f"{suite}: no poisoned heal-mode records yet")
    if not selections:
        sec.skip("no poisoned heal-mode records in any suite")
        return
    sec.data["runs"] = {s: len({r["run_dir"] for r in sel}) for s, sel in selections.items()}
    _heal_recovery(ctx, sec, selections)
    _heal_decisions(ctx, sec, selections)
    _heal_paired(ctx, sec, selections)
    fig = sec.data["figure"]
    for suite, pts in fig["points"].items():
        for mode, d in pts.items():
            sec.headline.append(f"{suite} ({fig['setting'][suite]}): {mode} own-task ASR "
                                f"{fmt(d['asr_target'])}, poison left {fmt(d['poison_left'])}")


def _task_index(ctx: Ctx) -> Callable[[dict], int]:
    return lambda rec: run.task_index if (run := ctx.run_of(rec)) else 0


def heal_rows(recs: list[dict], task_index: Callable[[dict], int], n_boot: int,
              seed: int) -> tuple[list[dict], int]:
    """config_rows (the run-pooled counts sections 1-4 report) with every interval
    replaced by cluster_rate's, resampling scenarios, and each row's scenario count:
    section 5's runs are clustered by scenario, so a Wilson interval over runs, tasks or
    probes treats as independent what is not."""
    rows, missing = config_rows(recs, task_index)
    baselines = measure._none_baselines(recs)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in measured(recs):
        by[config_of(r)].append(r)

    def units(rs, f):
        return [(scenario_of(r), *v) for r in rs if (v := f(r)) is not None]

    def own(r):
        t = target_record(r, task_index(r))
        return None if t is None else measure.asr_counts([t])

    def poison(r):
        v = r.get("residual_attack")
        return None if v is None else (float(v > 0), 1)

    for d in rows:
        rs = by[d["config"]]
        rate_of = lambda f: cluster_rate(units(rs, f), n_boot, seed)
        d.update(scenarios=len({scenario_of(r) for r in rs}), asr_target=rate_of(own),
                 asr_all_tasks=rate_of(lambda r: measure.asr_counts([r])),
                 bmp=rate_of(lambda r: measure.bmp_counts([r], baselines)[:2]),
                 poison_left=rate_of(poison))
    return rows, missing


HEAL_RECOVERY_HEADER = [
    "group", "configuration", "runs (scenarios)",
    "ASR, run's own task (scenario bootstrap 95%)", "runs attacked on own task",
    "ASR, every replayed task (scenario bootstrap 95%)", "tasks per run",
    "BMP (probes kept; scenario bootstrap 95%)",
    "poison left in memory (runs; scenario bootstrap 95%)"]


def _heal_recovery(ctx: Ctx, sec: Section, selections: dict[str, list[dict]]) -> None:
    a, task_index = ctx.args, _task_index(ctx)
    rows, data, missing = [], {}, 0
    for suite, sel in selections.items():
        data[suite], m = heal_rows(sel, task_index, a.n_boot, a.seed)
        missing += m
        for d in data[suite]:
            rows.append([suite, d["config"], runs_scenarios(d["runs"], d["scenarios"]),
                         fmt(d["asr_target"]), f"{d['attacked_target']}/{d['target_runs']}",
                         fmt(d["asr_all_tasks"]), f"{d['tasks_per_run']:.1f}", fmt(d["bmp"]),
                         fmt(d["poison_left"])])
    if missing:
        sec.note(f"{missing} records did not replay the run's own task and are left out of "
                 "the own-task columns")
    sec.data["recovery"] = data
    sec.table("Recovery on the runs a heal mode was applied to, per suite (poisoned, "
              "checkpoint 0, patched pages). Every configuration of those runs from either "
              "file, one row per knob setting (the Phase 2 sweeps ran family=full, "
              "cf_workers=6, stop_after_action; the earlier records.jsonl the defaults), so "
              "compare rows of one setting. A suite pools its groups (models: every model; "
              "mem0: both pipelines). BMP is against the same run's 'none' under the same "
              "knobs. Rates pool runs; their intervals are percentile bootstrap 95% "
              "resampling scenarios (a scenario's runs share its pages and poison, and the "
              "models suite runs each scenario under three models, three times), and "
              f"'{SMALL_NOTE}' marks fewer than {SMALL} scenarios", HEAL_RECOVERY_HEADER, rows)
    sec.data["figure"] = _figure_points(selections, data)


# The figure's series: the heal modes and quarantine on the same detection, in
# analyze.py's slot order, so a mode has one colour in every report.
HEAL_FIGURE_MODES = ("heal", "heal_rebuild", "heal_oracle", "incident")


def _figure_points(selections: dict[str, list[dict]], data: dict[str, list[dict]]) -> dict:
    """Per suite, the knob setting most of its heal-mode records share (no modifiers),
    and for each HEAL_FIGURE_MODES mode the row of that mode at that setting, so a
    suite's dots all come from one setting and never from two sweeps."""
    out = {"setting": {}, "points": {}}
    for suite, sel in selections.items():
        plain = [r for r in sel if not r.get("modifiers")]
        heal_kn = Counter(measure.knobs(r) for r in plain if r.get("mode") in HEAL_MODES)
        if not heal_kn:
            continue
        kn = heal_kn.most_common(1)[0][0]
        by_cfg = {d["config"]: d for d in data.get(suite, ())}
        pts = {}
        for mode in HEAL_FIGURE_MODES:
            rec = next((r for r in plain if r.get("mode") == mode and measure.knobs(r) == kn), None)
            if rec is not None and config_of(rec) in by_cfg:
                pts[mode] = by_cfg[config_of(rec)]
        out["setting"][suite] = measure.knob_label(kn).strip() or "default knobs"
        out["points"][suite] = pts
    return out


def _field_mean(rs: list[dict], key: str) -> float | None:
    """Mean over the records that state `key` (records from before heal lack its fields)."""
    vals = [r[key] for r in rs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def heal_decisions(rs: list[dict], n_boot: int, seed: int) -> dict:
    """What one heal configuration did over its runs: means of the heal counts, claims
    dropped per rule (summed over the records that state dropped_by, heal v2; rules
    beyond memheal/heal.py's are kept, not lost), ddmin's replays, and the post-heal
    check. post_heal_harmful is a per-run share whose replay count is not recorded, so
    runs are averaged, never replays pooled. Every interval resamples scenarios
    (cluster_rate), and 'small' counts scenarios."""
    stated = [r for r in rs if r.get("dropped_by") is not None]
    names = [*HEAL_RULES, *sorted({k for r in stated for k in r["dropped_by"]} - set(HEAL_RULES))]
    checked = [r for r in rs if r.get("post_heal_harmful") is not None]
    scenarios = len({scenario_of(r) for r in rs})
    per_run = lambda key, f, prop=False: cluster_rate(
        [(scenario_of(r), f(r[key]), 1) for r in rs if r.get(key) is not None], n_boot, seed,
        proportion=prop)
    return {"runs": len(rs), "scenarios": scenarios, "small": scenarios < SMALL,
            **{k: _field_mean(rs, k) for k in HEAL_COUNTS},
            "dropped_by_runs": len(stated),
            "dropped_by": {k: sum(r["dropped_by"].get(k, 0) for r in stated) for k in names},
            "ddmin_replays": per_run("ddmin_replays", float),
            "post_heal_runs": len(checked),
            "post_heal_harmful_share": per_run("post_heal_harmful", float),
            "post_heal_runs_harmful": per_run("post_heal_harmful", lambda v: float(v > 0),
                                              True)}


HEAL_COUNTS = ("n_healed", "n_heal_quarantined", "claims_kept", "claims_dropped")


def _heal_decisions(ctx: Ctx, sec: Section, selections: dict[str, list[dict]]) -> None:
    a = ctx.args
    rows, data = [], {}
    num = lambda v: "-" if v is None else f"{v:.1f}"
    for suite, sel in selections.items():
        by: dict[str, list[dict]] = defaultdict(list)
        for r in sel:
            if r.get("mode") in HEAL_MODES:
                by[config_of(r)].append(r)
        for cfg in sorted(by, key=lambda c: measure._config_sort(measure.setting_key(by[c][0]))):
            d = heal_decisions(by[cfg], a.n_boot, a.seed)
            data.setdefault(suite, {})[cfg] = d
            rows.append([suite, cfg, runs_scenarios(d["runs"], d["scenarios"]),
                         *[num(d[k]) for k in HEAL_COUNTS],
                         dropped_cell(d["dropped_by"]) if d["dropped_by_runs"] else "- (not recorded)",
                         fmt_mean(d["ddmin_replays"]), fmt_mean(d["post_heal_harmful_share"], 2),
                         fmt(d["post_heal_runs_harmful"])])
    sec.data["decisions"] = data
    sec.table("What healing did and why (heal-mode records; means per run). dropped by rule: "
              "claims dropped per memheal/heal.py rule, summed over the runs whose record "
              "states it (heal v2), with each rule's share: name = names the observed "
              "destination; source-taint = a root page's pointer or change claim; cf = the "
              "claim-level counterfactual; ddmin = delta debugging; support = a rebuilt claim "
              "no input supports; oracle = the AttackSpec. ddmin replays per run. Post-heal "
              "check: the explained decision replayed on the frozen set as healing left it, "
              "when a retrieved memory was healed; the mean per-run share of harmful replays "
              "(n = runs checked) and the checked runs with at least one harmful replay. "
              "Every interval is a percentile bootstrap 95% resampling scenarios, and "
              f"'{SMALL_NOTE}' marks fewer than {SMALL} scenarios",
              ["suite", "configuration", "runs (scenarios)", "healed", "heal-quarantined",
               "claims kept",
               "claims dropped", "dropped by rule", "ddmin replays",
               "post-heal harmful share", "runs with a harmful post-heal replay"], rows)


# Each heal mode is set against incident (the same detection, quarantined); heal_rebuild
# and heal_oracle also against heal (what rebuilding, and perfect claim labels, add).
HEAL_AGAINST = {"heal": (("incident", "cf"),),
                "heal_rebuild": (("incident", "cf"), ("heal", "cf")),
                "heal_oracle": (("incident", "cf"), ("heal", "cf"))}


def _heal_paired(ctx: Ctx, sec: Section, selections: dict[str, list[dict]]) -> None:
    a = ctx.args
    task_index = _task_index(ctx)
    rows, data = [], {}
    for suite, sel in selections.items():
        by: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in sel:
            by[config_of(r)][r["run_dir"]] = r
        heal_cfgs = sorted({c for c, rs in by.items()
                            if next(iter(rs.values())).get("mode") in HEAL_MODES},
                           key=lambda c: measure._config_sort(
                               measure.setting_key(next(iter(by[c].values())))))
        for cfg in heal_cfgs:
            rep = next(iter(by[cfg].values()))
            for mode, verdicts in HEAL_AGAINST.get(rep["mode"], ()):
                other = counterpart(rep, mode, verdicts, by)
                if other is None:
                    continue
                diffs = {name: clustered_paired(by[cfg], by[other], value, a.n_boot, a.seed)
                         for name, value in (
                             ("asr_target", lambda r: own_task_attacker(r, task_index(r))),
                             ("probes_correct", _probes_correct),
                             ("poison_left", _poison_left))}
                data.setdefault(suite, []).append({"a": cfg, "b": other, **diffs})
                rows.append([suite, cfg, other, *[fmt_diff(diffs[k], "scenarios") for k in
                                                  ("asr_target", "probes_correct", "poison_left")]])
    sec.data["paired"] = data
    sec.table("Paired within run, clustered by scenario: heal mode minus incident (same "
              "detection, quarantined) and minus heal, within one knob setting (one-sided knobs "
              "aside, measure.ONE_SIDED), on the runs both have: the run's own-task attacker "
              "outcome, its share of benign probes answered correctly, and whether an active "
              "memory still names the attacker. Each run's difference stays in its scenario; "
              "the estimate is the mean over scenarios of their runs' mean difference (each "
              "scenario weighs the same), with a percentile bootstrap 95% interval resampling "
              f"scenarios; higher/lower count scenarios, and '{SMALL_NOTE}' marks fewer than "
              f"{SMALL}. Each configuration is one replay per run",
              ["suite", "A", "B", "own-task attacker (A minus B)", "probes correct (A minus B)",
               "poison left (A minus B)"], rows)


def _poison_left(rec: dict) -> float | None:
    v = rec.get("residual_attack")
    return None if v is None else float(v > 0)


def _probes_correct(rec: dict) -> float | None:
    """The share of the record's benign probes answered correctly after recovery: paired
    by run, so both sides ask the same probes and no baseline is needed."""
    probes = rec.get("probes") or []
    return sum(bool(p.get("correct")) for p in probes) / len(probes) if probes else None


# --- figures ------------------------------------------------------------------------

def _style(ax, percent: bool = True) -> None:
    from matplotlib.ticker import PercentFormatter
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def dot_panel(ax, labels: list[str], series: list[tuple[str, list[dict | None]]], title: str,
              percent: bool = True, annotate: Callable[[dict], str] | None = None,
              colours: list[str] | None = None, small_unit: str = "runs") -> None:
    """Horizontal dot-and-whisker: one row per label (top to bottom), one dot per series
    in fixed palette order (or `colours`, when a series must keep its slot from another
    figure), its 95% interval as the whisker and its n beside it. A dot whose value is
    flagged small (fewer than 10 runs, or 10 `small_unit`) is drawn open. The legend
    sits between the title and the plot, so each panel names its own series."""
    from matplotlib.lines import Line2D
    annotate = annotate or (lambda v: f"{v['n']:.0f}/{v['d']}")
    k = len(series)
    step = 0.6 / k if k > 1 else 0.0
    top = max([v["ci"][1] for _, vals in series for v in vals if v and v.get("ci")] or [1.0])
    xmax = 1.0 if percent else (top * 1.05 or 1.0)
    handles, names, any_small = [], [], False
    for i, (name, vals) in enumerate(series):
        colour = colours[i] if colours else PALETTE[i]
        off = (i - (k - 1) / 2) * step
        for j, v in enumerate(vals):
            if not v or v.get("rate") is None:
                continue
            y = j + off
            lo, hi = v["ci"]
            ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([v["rate"]], [y], "o", ms=7, mfc=SURFACE if v["small"] else colour,
                    mec=colour, mew=1.8, zorder=3)
            ax.text(max(hi, v["rate"]) + 0.03 * xmax, y, annotate(v), va="center",
                    fontsize=6.5, color=FAINT)
            any_small |= bool(v["small"])
        handles.append(Line2D([], [], ls="", marker="o", ms=7, color=colour))
        names.append(name)
    if any_small:
        handles.append(Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=FAINT, mew=1.8))
        names.append(f"open dot: fewer than {SMALL} {small_unit}")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(-0.02 * xmax, xmax * 1.16)
    _style(ax, percent)
    legend_rows = 0
    if k >= 2 or any_small:
        ncol = 2 if any(len(n) > 18 for n in names) else min(len(names), 4)
        legend_rows = -(-len(names) // ncol)
        ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol,
                  frameon=False, fontsize=7.5, labelcolor=INK, borderaxespad=0.2,
                  handletextpad=0.3, columnspacing=1.2)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=6 + 13 * legend_rows)


def _figure(rows: list[int]):
    """A 6.5-inch-wide figure, one panel per entry of `rows` (its row count)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heights = [0.35 * max(r, 1) + 0.3 for r in rows]
    fig, axes = plt.subplots(len(rows), 1, squeeze=False, height_ratios=heights,
                             figsize=(6.5, sum(heights) + 0.95 * len(rows)))
    fig.patch.set_facecolor(SURFACE)
    return plt, fig, [ax for row in axes for ax in row]


def _save(plt, fig, path: Path, dpi: int = 150) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def _recovery_panel(ax, rec: dict, groups: list[str], lead: str) -> None:
    """Own-task ASR (row_data's asr_target), the measurement every group shares; when
    the groups replayed families of different sizes, a second title line says so."""
    configs = list(dict.fromkeys(d["config"] for g in groups for d in rec[g]))
    pick = lambda g, c: next((d["asr_target"] for d in rec[g] if d["config"] == c), None)
    per = [d["tasks_per_run"] for g in groups for d in rec[g] if d["runs"]]
    note = (f"\n(replayed tasks per run: {min(per):.1f} to {max(per):.1f}; the own task is "
            "in all)" if per and min(per) != max(per) else "")
    dot_panel(ax, configs, [(g, [pick(g, c) for c in configs]) for g in groups],
              f"{lead}: ASR on each run's own target task (Wilson 95%){note}")


def _recovery_rows(rec: dict, groups: list[str]) -> int:
    return len({d["config"] for g in groups for d in rec[g]})


def plot_adaptive(sec: Section, out: Path) -> Path | None:
    rates, surv = sec.data.get("rates") or {}, sec.data.get("survival") or {}
    cats = [c for c in (*ATTACK_CATEGORIES, *ADAPTIVE_CATEGORIES) if c in rates]
    if not cats:
        return None
    rec = sec.data.get("recovery") or {}
    groups = [g for g in sec.data.get("recovery_order") or () if g in rec][:len(PALETTE)]
    plt, fig, axes = _figure([len(cats)] + ([_recovery_rows(rec, groups)] if groups else []))
    stage = lambda c, s: (surv.get(c) or {}).get("stages", {}).get(s)
    dot_panel(axes[0], cats, [
        ("acted: attacker rate", [rates[c]["poisoned_attacker"] for c in cats]),
        ("carrying memory active pre-target", [stage(c, "pre_target") for c in cats]),
        ("carrying memory retrieved at target", [stage(c, "retrieved") for c in cats]),
    ], "Poisoned runs by category, same contexts (Wilson 95%)")
    if groups:
        _recovery_panel(axes[1], rec, groups, "Recovery")
    return _save(plt, fig, out / "history_adaptive.png")


def plot_models(sec: Section, out: Path, ref: str) -> list[Path]:
    models = sec.data.get("models") or {}
    if not models:
        return []
    names = list(models)
    plt, fig, axes = _figure([len(names), len(names)])
    for ax, key, title in ((axes[0], "poisoned_attacker", "Poisoned runs: attacker rate"),
                           (axes[1], "clean_correct", "Clean runs: correct rate")):
        dot_panel(ax, names, [("model", [models[m][key] for m in names]),
                              (f"{ref}, same variants",
                               [models[m][f"reference_{key}"] for m in names])],
                  f"{title} (Wilson 95%)")
    made = [_save(plt, fig, out / "history_models.png")]
    rec, order = sec.data.get("recovery") or {}, sec.data.get("recovery_order") or []
    groups = [g for g in order if g in rec][:len(PALETTE)]
    if groups:
        plt, fig, axes = _figure([_recovery_rows(rec, groups)])
        _recovery_panel(axes[0], rec, groups, "Recovery per model")
        made.append(_save(plt, fig, out / "history_models_recovery.png"))
    return made


def plot_external(sec: Section, out: Path) -> Path | None:
    groups = sec.data.get("groups") or {}
    rows = [g for g in EXTERNAL_GROUPS if g in groups]
    if not rows:
        return None
    probes, stages = sec.data.get("probes") or {}, sec.data.get("probe_stages") or []
    suites = [s for s in ("LongMemEval", "MemBench")
              if any(k.startswith(f"{s} | ") for k in probes)]
    plt, fig, axes = _figure([len(rows)] + ([len(stages)] if suites and stages else []))
    dot_panel(axes[0], rows, [("clean", [groups[g]["clean_attacker"] for g in rows]),
                              ("poisoned", [groups[g]["poisoned_attacker"] for g in rows])],
              "External suites: attacker rate (Wilson 95%)")
    if suites and stages:
        dot_panel(axes[1], stages, [
            (s, [(probes.get(f"{s} | {st}") or {}).get("accuracy") for st in stages])
            for s in suites], "External probe answered from memory (Wilson 95%)")
    return _save(plt, fig, out / "history_external.png")


def plot_mem0(sec: Section, out: Path) -> Path | None:
    rates, pipes = sec.data.get("rates") or {}, sec.data.get("pipelines") or []
    if not rates:
        return None
    labels, att, ok = [], [], []
    for p in pipes:
        for name, d in rates[p].items():
            if not name.startswith("diff"):
                labels.append(f"{p} variants: {name}")
                att.append(d["poisoned_attacker"])
                ok.append(d["clean_correct"])
    writes = sec.data.get("writes") or {}
    wp = [p for p in pipes if f"{p} poisoned" in writes][:len(PALETTE)]
    plt, fig, axes = _figure([len(labels)] + ([len(MEM0_EVENTS)] if wp else []))
    dot_panel(axes[0], labels, [("poisoned: attacker rate", att), ("clean: correct rate", ok)],
              "Mem0 vs evolving and flat, same variants (Wilson 95%)")
    if wp:
        dot_panel(axes[1], list(MEM0_EVENTS),
                  [(f"{p} pipeline", [writes[f"{p} poisoned"]["per_run"][e] for e in MEM0_EVENTS])
                   for p in wp],
                  "Mem0 write events per poisoned run (mean, bootstrap 95%)", percent=False,
                  annotate=lambda v: f"n={v['d']}")
    return _save(plt, fig, out / "history_mem0.png")


# analyze.py's slots for these modes (its HEAL_COMPARE order), so a mode is one colour in
# every report.
HEAL_FIGURE_COLOURS = dict(zip(HEAL_FIGURE_MODES, PALETTE))


def plot_heal(sec: Section, out: Path) -> Path | None:
    """Per suite (rows), each heal mode and incident at the suite's heal setting
    (_figure_points): own-task ASR, benign probes kept (BMP) and poison left, each with
    its scenario-bootstrap 95% interval (heal_rows), n and scenario count; a dot over
    fewer than 10 scenarios is drawn open."""
    fig_data = (sec.data.get("figure") or {})
    points, setting = fig_data.get("points") or {}, fig_data.get("setting") or {}
    suites = [s for s in points if points[s]]
    modes = [m for m in HEAL_FIGURE_MODES if any(m in points[s] for s in suites)]
    if not suites or not modes:
        return None
    # One knob setting for every suite goes in the first title; otherwise each row says its own.
    one = len(set(setting[s] for s in suites)) == 1
    labels = suites if one else [f"{s}\n{setting[s]}" for s in suites]
    plt, fig, axes = _figure([len(suites) * (1 if one else 2)] * 3)
    colours = [HEAL_FIGURE_COLOURS[m] for m in modes]
    for i, (ax, key, title) in enumerate((
            (axes[0], "asr_target", "Own-task ASR after recovery"),
            (axes[1], "bmp", "Benign probes kept, BMP"),
            (axes[2], "poison_left", "Poison left: runs with an active memory naming the "
                                     "attacker"))):
        title += " (scenario bootstrap 95%)"
        if i == 0 and one:
            title += f"\nknob setting: {setting[suites[0]]}"
        dot_panel(ax, labels, [(m, [(points[s].get(m) or {}).get(key) for s in suites])
                               for m in modes], title, colours=colours, small_unit="scenarios",
                  annotate=lambda v: f"{v['n']:.0f}/{v['d']:.0f}, {v['scenarios']} sc.")
    return _save(plt, fig, out / "history_heal.png", dpi=200)


# --- driver -------------------------------------------------------------------------

SECTIONS = (
    ("inputs", "0. Inputs", inputs_section),
    ("adaptive", "1. Adaptive poison (Section 12.3, X10)", adaptive_section),
    ("models", "2. Stronger and cross-model runs (Sections 8.11, 12.3, X11)", models_section),
    ("external", "3. External suites: LongMemEval and MemBench (Sections 7.4, 7.5, X12)",
     external_section),
    ("mem0", "4. Mem0 backend (Section 12.3, X13)", mem0_section),
    ("heal", "5. Healing per suite (Phase 2, H)", heal_section),
)


def run_section(key: str, title: str, build, ctx: Ctx) -> Section:
    """Build one section; a failure skips it (with the reason) unless --strict."""
    sec = Section(key, title)
    try:
        build(ctx, sec)
    except Exception as exc:                                   # noqa: BLE001
        if ctx.args.strict:
            raise
        sec.note(f"section stopped early: {type(exc).__name__}: {exc}")
        sec.data["error"] = f"{type(exc).__name__}: {exc}"
    return sec


def _range(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)-(\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected START-END, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=RUNS)
    p.add_argument("--min-run", type=int, default=1443,
                   help="first history run number (1435 includes the smoke runs)")
    p.add_argument("--base-runs", type=_range, default=(959, 1197),
                   help="original dense evolving runs (default 0959-1197)")
    p.add_argument("--flat-runs", type=_range, default=(1198, 1434),
                   help="flat-memory twins of the dense runs (default 1198-1434)")
    p.add_argument("--reference-model", default=REFERENCE_MODEL)
    for name, path in RECORDS.items():
        p.add_argument(f"--{name}-records", type=Path, default=path)
    for suite, paths in MAIN.items():
        p.add_argument(f"--{suite}-main", type=Path, nargs="*", default=list(paths),
                       metavar="PATH", help=f"{suite}'s Phase 2 records files, read with "
                                            f"its records.jsonl by section 5 (default "
                                            f"{', '.join(str(x.relative_to(ROOT)) for x in paths)})")
    p.add_argument("--no-parts", dest="parts", action="store_false",
                   help="section 5: ignore a running batch's unmerged .parts files")
    p.add_argument("--diagnosis-adaptive", type=Path, default=DIAGNOSIS_ADAPTIVE)
    p.add_argument("--diagnosis-standard", type=Path, default=DIAGNOSIS_STANDARD)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="raise instead of skipping a section that fails")
    return p.parse_args(argv)


def _wanted(args: argparse.Namespace) -> Callable[[int], bool]:
    (b0, b1), (f0, f1) = args.base_runs, args.flat_runs
    return lambda n: b0 <= n <= b1 or f0 <= n <= f1 or n >= args.min_run


def write_outputs(sections: list[Section], args: argparse.Namespace, seconds: float,
                  figures: list[Path]) -> list[Path]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# History runs (generated by analyze_history.py, history runs from "
          f"{args.min_run:04d})",
          "Sections 1-4: rates are Wilson 95% intervals; differences are bootstrap 95% "
          f"intervals over matched scenarios; any cell with fewer than {SMALL} runs is marked "
          f"'{SMALL_NOTE}'. Section 5 pools runs that share scenarios (several history runs "
          "and, for the models suite, three models per scenario), so every interval there "
          "is a percentile bootstrap 95% that resamples scenarios with all their runs, n is "
          f"given as runs (scenarios), and '{SMALL_NOTE}' marks fewer than {SMALL} scenarios. "
          "Every number is computed from run folders and records files."]
    for sec in sections:
        md.append(f"## {sec.title}")
        md.extend(sec.md)
    if figures:
        md.append("Figures: " + ", ".join(p.name for p in figures))
    tables = out / "history_tables.md"
    tables.write_text("\n\n".join(md) + "\n")
    doc = {"meta": {"min_run": args.min_run, "base_runs": list(args.base_runs),
                    "flat_runs": list(args.flat_runs), "reference_model": args.reference_model,
                    "n_boot": args.n_boot, "seed": args.seed, "seconds": round(seconds, 2),
                    "small_sample_below": SMALL, "figures": [p.name for p in figures]},
           **{sec.key: {**sec.data, "notes": sec.notes} for sec in sections}}
    path = out / "history.json"
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return [tables, path]


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    t0 = time.time()
    ctx = Ctx(args, RunIndex.scan(args.runs_dir, _wanted(args)))
    built = [run_section(k, t, b, ctx) for k, t, b in SECTIONS]
    # A section with nothing to report on these inputs at all (section 5 before any
    # Phase 2 file exists) is left out, so the report reads as it did before it.
    sections = [s for s in built if not s.omit]
    for s in built:
        if s.omit:
            print(f"[{s.key}] left out: {s.omit}")
    by = {s.key: s for s in sections}
    figures: list[Path] = []
    if not args.no_figures:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        plots = (("adaptive", lambda s: [plot_adaptive(s, args.out)]),
                 ("models", lambda s: plot_models(s, args.out, args.reference_model)),
                 ("external", lambda s: [plot_external(s, args.out)]),
                 ("mem0", lambda s: [plot_mem0(s, args.out)]),
                 ("heal", lambda s: [plot_heal(s, args.out)]))
        for key, plot in plots:
            if key not in by:
                continue
            try:
                figures += [p for p in plot(by[key]) if p]
            except Exception as exc:                           # noqa: BLE001
                if args.strict:
                    raise
                by[key].note(f"figure not drawn: {type(exc).__name__}: {exc}")
    written = write_outputs(sections, args, time.time() - t0, figures)
    for sec in sections:
        print(f"[{sec.key}]")
        for line in sec.notes:
            print(f"  note: {line}")
        for line in sec.headline:
            print(f"  {line}")
    print("wrote " + ", ".join(str(p) for p in (*written, *figures))
          + f" in {time.time() - t0:.1f}s")
    return {"sections": by, "written": written, "figures": figures}


if __name__ == "__main__":
    main(sys.argv[1:])
