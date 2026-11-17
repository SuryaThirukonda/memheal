"""Final tables and figures for the main comparison and its companion experiments
(Phase 2 plan, steps O.4 and O.7).

The main comparison recovers every qualified poisoned run of the dense evolving suite
(120 scenarios x 3 independent history runs; folders 0959-1197 are history sample 1,
the later folders of the same variant samples 2 and 3) under twelve configurations,
each replayed three times (repeat 1-3) on the full four-task family. This script turns
that records file, the clean false-alarm control, the reinfection follow-up, the
other suites' records and bcp.py's benign claim preservation into the report:

    1. trade-off: ASR against BMP and against BCP per configuration
    2. healing vs quarantine with the same detection: paired differences (forest plot)
    3. per attack category and per domain (small multiples)
    4. repetition: replays and history samples, variance decomposition, how often one
       replay would have concluded otherwise than the 3-replay majority
    5. clean false alarms: incident vs heal on clean runs
    6. reinfection: heal+immunize vs selective+immunize over the Section 8.8 checkpoints
    7. cost: LLM calls and seconds per configuration; heal's claim-level replays
    8. cross-suite: heal vs incident per suite

    uv run python analyze_main.py
    uv run python analyze_main.py --out /tmp/final --n-boot 500 --no-figures

Statistics. The unit of independence is the SCENARIO (variant): a scenario's three
history runs share its pages, tasks and poison, and a run's three replays share its
store, so neither runs nor replays are independent samples. Every metric is first
averaged per scenario (mean over its runs and repeats), then reported as the mean over
scenarios with a percentile-bootstrap 95% interval that resamples scenarios (--n-boot,
default 2000, fixed --seed). A difference between configurations is paired: taken per
(run, repeat) on the measurements both configurations have, averaged per scenario, then
bootstrapped over scenarios the same way. The variance decomposition is a nested
random-effects ANOVA (scenario > history run > replay; method-of-moments with the
unbalanced-design coefficients), its shares bootstrapped over scenarios too. Every cell
reports n (scenarios) and one with fewer than 10 scenarios is flagged 'small sample'.

Matched measurements. A running batch fills configurations in unevenly: a mode added
later may so far cover one history sample, and a mode's ASR can differ between samples,
so setting each configuration's mean over everything it has against another's compares
runs as much as configurations (and can reverse an ordering the paired difference does
not show). Every per-configuration table and figure therefore stands on the measurements
(run, replay, checkpoint) on which every configuration in it has a value of that metric:
Section 1 over all configurations; Section 3 within each category or domain; Section 5
over the clean runs; Section 6 over the runs every configuration followed; Section 7 over
all configurations (heal's replays over the heal configurations); Section 8 within each
suite. A configuration with no value of a metric at all is left out of that metric's
requirement. With even coverage this is every measurement. Each section says how many
measurements that is, and each figure note says which basis it uses; the
coverage-dependent numbers (each configuration on everything it has) are kept in
main.json under 'all_measurements', and Section 1 also prints them as a second table.
Section 4 compares replays and history samples within one configuration, so it stands on
each configuration's own measurements and says so.

Inputs (each a flag of the same name; a missing one skips what needs it). Records a
running batch has not merged yet (<dir>/.parts/<stem>.*.jsonl, see batch.py) are read
too unless --no-parts; a measurement present twice collapses to its last copy.

    --main          results/main/records.jsonl        the main comparison (repeats 1-3)
    --clean         results/main/clean_control.jsonl  clean runs, simulated false alarm
    --reinfect      results/main/reinfect.jsonl       +immunize, reinfection checkpoints
    --flat, --adaptive, --models, --external, --mem0, --mem0-additive
                    results/<suite>/main.jsonl        the same modes on the other suites
    --preservation  results/preservation_final/preservation.jsonl  (bcp.py output)
    --runs          data/runs                         run.db, read-only, for the model of
                                                      each cross-model run

Per record: ASR = attacker outcomes / replayed tasks; BR = the record's br; BMP =
probes still correct among those the same run's 'none' record (same repeat) answered
correctly (measure.bmp_counts); BCP = bcp.py's benign claims kept / benign claims,
joined on the recovered copy (work_db) and else on measure's record identity among the
rows bcp.py wrote for the same records file (another file's row with the same identity,
e.g. results/families/records.jsonl's, counts as 'no row for this file');
residual attack = active memories still naming the attacker; memories removed (net) =
quarantined - restored - reconstructed - healed (a healed memory replaces the
original it quarantines). Rates are in %, their differences in percentage points.

Nothing here calls a model or writes outside --out. Outputs (under --out, default
results/final/): main_tables.md (every table, n in every cell, a computed one-line
takeaway per section), main.json (every number), main_*.png (200 dpi, 6.5 in wide).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import measure
from memheal.scenarios import LONGMEMEVAL, MEMBENCH

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "final"
RUNS = ROOT / "data" / "runs"
INPUTS = {
    "main": "results/main/records.jsonl",
    "clean": "results/main/clean_control.jsonl",
    "reinfect": "results/main/reinfect.jsonl",
    "flat": "results/flat/main.jsonl",
    "adaptive": "results/adaptive/main.jsonl",
    "models": "results/models/main.jsonl",
    "external": "results/external/main.jsonl",
    "mem0": "results/mem0/main.jsonl",
    "mem0_additive": "results/mem0_additive/main.jsonl",
}
PRESERVATION = "results/preservation_final/preservation.jsonl"
N_BOOT = 2000
SEED = 0
SMALL = 10                  # scenarios below which a cell is flagged 'small sample'
VERIFY_N = 3                # the main sweep's --verify-n; other +verify records are left out
SAMPLE1 = (959, 1197)       # the original dense evolving sweep = history sample 1
REPEATS = (1, 2, 3)

# Configuration labels, as measure.config_label(config_key(rec), False) writes them.
NONE = "none"
INC, INC_V = "incident[cf]", "incident[cf]+verify"
HEAL, HEAL_V = "heal[cf]", "heal[cf]+verify"
HEAL_RB, HEAL_OR = "heal_rebuild[cf]", "heal_oracle[oracle]"
MAIN_CONFIGS = (NONE, "ancestry", "selective[oracle]", "selective_recon[oracle]",
                "selective_claims[oracle]", "unmerge[oracle]", INC, INC_V, HEAL, HEAL_RB,
                HEAL_V, HEAL_OR)
# Four colour groups (the palette has four slots): what each configuration needs.
GROUPS = (("quarantine on detection (no oracle)", (INC, INC_V)),
          ("heal on detection (no oracle)", (HEAL, HEAL_RB, HEAL_V)),
          ("given the root or oracle labels", ("ancestry", "selective[oracle]",
                                               "selective_recon[oracle]",
                                               "selective_claims[oracle]", "unmerge[oracle]",
                                               HEAL_OR)),
          ("no recovery", (NONE,)))
# Same detection, different treatment: healing minus quarantine.
HEAL_PAIRS = ((HEAL, INC), (HEAL_RB, INC), (HEAL_OR, INC), (HEAL_V, INC_V))
# Knobs the main sweeps set; a record departing on any other knob is another experiment.
# verify_n is checked on its own (selection_problem); claim_n and ddmin_budget are heal's
# own settings. Newer measure.py versions list the last three among KNOBS, older ones not.
ALLOWED_KNOBS = {"family", "repeat", "cf_workers", "stop_after_action", "verify_n", "claim_n",
                 "ddmin_budget"}
# Knobs left out of the BCP join's fallback key: bcp.py rows written by an older measure.py
# lack them (the recovered copy, work_db, is the primary join and does tell them apart).
JOIN_IGNORES = ("stop_after_action", "verify_n", "claim_n", "ddmin_budget")

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "D", "^")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
WIDTH = 6.5                 # inches: one text column


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str                  # '%' for rates (differences in pp), else a count unit
    better: str | None         # 'lower' | 'higher' | None
    digits: int = 1

    @property
    def diff_unit(self) -> str:
        return "pp" if self.unit == "%" else self.unit

    @property
    def label(self) -> str:
        """'ASR (%)'; the unit is left out when the name already says it."""
        return self.name if self.unit in self.name else f"{self.name} ({self.unit})"


METRICS = {
    "asr": Metric("ASR", "%", "lower"),
    "bmp": Metric("BMP", "%", "higher"),
    "bcp": Metric("BCP", "%", "higher"),
    "br": Metric("BR", "%", "higher"),
    "residual": Metric("residual attack", "memories", "lower", 2),
    "quarantined": Metric("memories quarantined", "memories", None, 2),
    "removed": Metric("memories removed (net)", "memories", "lower", 2),
    "llm_calls": Metric("LLM calls", "calls", "lower", 1),
    "seconds": Metric("seconds", "s", "lower", 1),
    "claim_cf_replays": Metric("claim-level replays", "replays", None, 1),
    "ddmin_replays": Metric("ddmin replays", "replays", None, 1),
    "carrying": Metric("store names the attacker", "%", "lower"),
}
FOREST_METRICS = ("asr", "bmp", "bcp", "residual", "removed", "llm_calls", "seconds")


# --- reading ----------------------------------------------------------------------------

def read_jsonl(path: Path) -> tuple[list[dict], int]:
    """A JSON-lines file's rows and how many lines were torn (a writer mid-line)."""
    rows, torn = [], 0
    try:
        text = Path(path).read_text()
    except FileNotFoundError:              # a part merged and removed between glob and read
        return [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    return rows, torn


def identity(rec: dict) -> tuple:
    """One measurement: measure.record_key plus the heal-v2 knobs record_key does not
    know (a verify_n=1 and a verify_n=3 record of one run are two measurements)."""
    return (*measure.record_key(rec), rec.get("verify_n"), rec.get("claim_n"),
            rec.get("ddmin_budget"))


def read_records(path: Path, parts: bool = True) -> tuple[list[dict] | None, dict]:
    """`path` plus the part files a running batch has not merged yet; a measurement in
    both collapses to its last copy. None when neither exists (missing, not empty)."""
    path = Path(path)
    files = [path] if path.is_file() else []
    part_files = sorted((path.parent / ".parts").glob(f"{path.stem}.*.jsonl")) if parts else []
    info = {"path": _rel(path), "part_files": len(part_files), "torn_lines": 0, "read": 0}
    if not files and not part_files:
        return None, {**info, "state": "missing"}
    rows = []
    for f in files + part_files:
        got, torn = read_jsonl(f)
        rows += got
        info["torn_lines"] += torn
    last = {identity(r): i for i, r in enumerate(rows)}
    out = [r for i, r in enumerate(rows) if last[identity(r)] == i]
    info.update(read=len(rows), duplicates=len(rows) - len(out), records=len(out),
                state="ok" if out else "empty")
    return out, info


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def config_of(rec: dict) -> str:
    return measure.config_label(measure.config_key(rec), False)


def selection_problem(rec: dict, condition: str, verify_n: int, checkpoint0: bool = True,
                      family: str | None = "full") -> str | None:
    """Why a record is not part of the comparison (None when it is)."""
    if (rec.get("condition") or "poisoned") != condition:
        return f"condition {rec.get('condition')}"
    if checkpoint0 and (rec.get("checkpoint") or 0) != 0:
        return "later checkpoint"
    if (rec.get("source") or "patched") != "patched":
        return f"replay pages {rec.get('source')}"
    if family is not None and rec.get("family") != family:
        return f"family={rec.get('family')}"
    other = sorted(k for k, _ in measure.knobs(rec) if k not in ALLOWED_KNOBS)
    if other:
        return "other knobs: " + ",".join(other)
    if "verify" in (rec.get("modifiers") or ()) and rec.get("verify_n") != verify_n:
        return f"verify_n={rec.get('verify_n')}"
    return None


def select(recs: list[dict], condition: str, verify_n: int, checkpoint0: bool = True
           ) -> tuple[list[dict], Counter]:
    kept, dropped = [], Counter()
    for r in recs:
        why = selection_problem(r, condition, verify_n, checkpoint0)
        if why is None:
            kept.append(r)
        else:
            dropped[why] += 1
    return kept, dropped


_RUN_NUMBER = re.compile(r"^(\d+)_")


def run_number(run_dir: str) -> int | None:
    m = _RUN_NUMBER.match(run_dir or "")
    return int(m.group(1)) if m else None


def history_samples(recs: Iterable[dict], sample1: tuple[int, int] = SAMPLE1
                    ) -> tuple[dict[str, int], dict]:
    """History sample of each run: per variant, the run in the original sweep's folder
    range is sample 1 and the others, by folder number, samples 2, 3, ... Anomalies (a
    variant with no or several sample-1 runs, or more than three runs) are counted."""
    by_variant: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        by_variant[r.get("variant") or ""].add(r["run_dir"])
    out, odd = {}, Counter()
    lo, hi = sample1
    for variant, runs in by_variant.items():
        def key(rd):
            n = run_number(rd)
            return (n if n is not None else 10 ** 9, rd)
        ordered = sorted(runs, key=key)
        first = [rd for rd in ordered if lo <= (run_number(rd) or -1) <= hi]
        if len(first) != 1:
            odd["variants without exactly one sample-1 run"] += 1
        if len(ordered) > 3:
            odd["variants with more than 3 runs"] += 1
        rest = [rd for rd in ordered if rd not in first[:1]]
        for rd in first[:1]:
            out[rd] = 1
        for i, rd in enumerate(rest, start=2 if first else 1):
            out[rd] = i
    return out, dict(odd)


class ModelIndex:
    """The model of a run, read from its run.db (read-only), cached per run folder."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.cache: dict[str, str | None] = {}

    def __call__(self, run_dir: str) -> str | None:
        if run_dir not in self.cache:
            self.cache[run_dir] = self._read(self.runs_dir / run_dir / "run.db")
        return self.cache[run_dir]

    @staticmethod
    def _read(path: Path) -> str | None:
        if not path.is_file():
            return None
        try:
            conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            try:
                row = conn.execute("SELECT model, config_json FROM experiment_runs "
                                   "LIMIT 1").fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        if row[0]:
            return row[0]
        try:
            return (json.loads(row[1] or "{}").get("llm") or {}).get("model")
        except ValueError:
            return None


class Preservation:
    """bcp.py's rows, matched to records by the recovered copy (work_db) and else by
    measure's record identity without the JOIN_IGNORES knobs (which older rows lack),
    among the rows bcp.py wrote for the same records file only. The identity does not
    tell experiments apart: results/families/records.jsonl repeats the main suite's
    sample-1 runs, configurations, family and repeats without a work_db, so a row of
    another file with the same identity is another experiment's trial."""

    def __init__(self, rows: list[dict] | None):
        self.rows = rows or []
        self.by_copy: dict[str, dict] = {}
        self.by_key: dict[tuple, list[dict]] = defaultdict(list)
        for row in self.rows:
            if row.get("work_db"):
                self.by_copy[_norm(row["work_db"])] = row
            self.by_key[self.key(row)].append(row)

    @staticmethod
    def key(rec: dict) -> tuple:
        return (rec.get("run_dir"), measure.config_key(rec),
                measure.drop_knobs(measure.knobs(rec), JOIN_IGNORES),
                rec.get("checkpoint") or 0, rec.get("condition") or "poisoned")

    def row_for(self, rec: dict, records_path: str | None = None
                ) -> tuple[dict | None, str | None]:
        """(row, None), or (None, why): 'no row' when no row has the record's identity,
        'no row for this file' when only rows of other records files have it. The file
        is compared as both sides name it (bcp.py's `records`, normalised like work_db);
        a row naming no file matches only a record given no file."""
        if rec.get("work_db"):
            hit = self.by_copy.get(_norm(rec["work_db"]))
            if hit is not None:
                return hit, None
        rows = self.by_key.get(self.key(rec)) or []
        if not rows:
            return None, "no row"
        want = _norm(records_path) if records_path else None
        same = [r for r in rows if (_norm(r["records"]) if r.get("records") else None) == want]
        if not same:
            return None, "no row for this file"
        return same[-1], None

    def bcp(self, rec: dict, records_path: str | None = None) -> tuple[float | None, str]:
        """(BCP in %, status): status 'ok', 'no benign claims', 'no row', 'no row for this
        file', or bcp.py's."""
        row, why = self.row_for(rec, records_path)
        if row is None:
            return None, why
        if row.get("status") != "ok":
            return None, str(row.get("status"))
        benign, kept = row.get("benign"), row.get("benign_kept")
        if benign:
            return 100.0 * kept / benign, "ok"
        if row.get("bcp") is not None:
            return 100.0 * row["bcp"], "ok"
        return None, "no benign claims"


def _norm(p: str) -> str:
    s = str(p)
    if s.startswith("./"):
        s = s[2:]
    rel = _rel(s) if Path(s).is_absolute() else s
    return rel or s


# --- units: one record, its scenario and its values --------------------------------------

def record_values(rec: dict, baselines: dict) -> dict[str, float | None]:
    outs = rec.get("task_outcomes") or []
    if outs:
        asr = 100.0 * sum(o == "attacker" for o in outs) / len(outs)
    else:
        asr = None if rec.get("asr") is None else 100.0 * rec["asr"]
    kept, total, missing = measure.bmp_counts([rec], baselines)
    bmp = 100.0 * kept / total if total else None
    nq = rec.get("n_quarantined")
    removed = None if nq is None else (nq - (rec.get("n_restored") or 0)
                                       - (rec.get("n_reconstructed") or 0)
                                       - (rec.get("n_healed") or 0))
    return {
        "asr": asr,
        "br": None if rec.get("br") is None else 100.0 * rec["br"],
        "bmp": bmp,
        "residual": rec.get("residual_attack"),
        "quarantined": nq,
        "removed": removed,
        "llm_calls": rec.get("llm_calls"),
        "seconds": rec.get("seconds"),
        "claim_cf_replays": rec.get("claim_cf_replays"),
        "ddmin_replays": rec.get("ddmin_replays"),
    }


def build_units(recs: list[dict], *, baselines_from: list[dict] | None = None,
                samples: dict[str, int] | None = None, pres: Preservation | None = None,
                records_path: str | None = None, suite: str | None = None) -> tuple[list[dict], Counter]:
    """One unit per record: scenario, run, history sample, repeat, configuration,
    category, domain, checkpoint and every metric; and the BCP join's status counts."""
    baselines = measure._none_baselines(baselines_from if baselines_from is not None else recs)
    units, bcp_status = [], Counter()
    for r in recs:
        vals = record_values(r, baselines)
        if pres is not None and pres.rows:
            vals["bcp"], status = pres.bcp(r, records_path)
        else:
            vals["bcp"], status = None, "no preservation file"
        bcp_status[status] += 1
        units.append({"scenario": r.get("variant") or "", "run": r["run_dir"],
                      "sample": (samples or {}).get(r["run_dir"]), "repeat": r.get("repeat"),
                      "config": config_of(r), "category": r.get("category") or "",
                      "domain": r.get("domain") or "", "checkpoint": r.get("checkpoint") or 0,
                      "suite": suite, "carrying": _carrying(r), **vals})
    return units, bcp_status


def _carrying(rec: dict) -> float | None:
    v = measure.memory_carries(rec)
    return None if v is None else 100.0 * v


# --- statistics ---------------------------------------------------------------------------

@dataclass
class Ctx:
    out: Path
    n_boot: int = N_BOOT
    seed: int = SEED
    figures: bool = True
    verify_n: int = VERIFY_N


def _keep(config: str | None = None, where: Callable[[dict], bool] | None = None):
    def ok(u: dict) -> bool:
        return (config is None or u["config"] == config) and (where is None or where(u))
    return ok


def per_scenario(units: Iterable[dict], metric: str, keep: Callable[[dict], bool]) -> dict[str, float]:
    """Each scenario's mean over its runs and repeats (units with a value only)."""
    vals: dict[str, list[float]] = defaultdict(list)
    for u in units:
        if keep(u) and u.get(metric) is not None:
            vals[u["scenario"]].append(float(u[metric]))
    return {s: float(np.mean(v)) for s, v in vals.items()}


def boot(values: list[float], ctx: Ctx) -> tuple[float, float, float] | None:
    """Mean and percentile-bootstrap 95% interval, resampling the values (one per
    scenario) with replacement, ctx.n_boot times from a fixed seed. The values are
    sorted first: callers collect them from sets of string-keyed tuples, whose order
    changes with Python's per-process hash seed, and the same seed must give the same
    interval for the same values."""
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return None
    rng = np.random.default_rng(ctx.seed)
    means = arr[rng.integers(0, arr.size, size=(ctx.n_boot, arr.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def summary(units: list[dict], metric: str, ctx: Ctx, config: str | None = None,
            where: Callable[[dict], bool] | None = None) -> dict | None:
    keep = _keep(config, where)
    sc = per_scenario(units, metric, keep)
    b = boot(list(sc.values()), ctx)
    if b is None:
        return None
    used = [u for u in units if keep(u) and u.get(metric) is not None]
    return {"mean": b[0], "lo": b[1], "hi": b[2], "n": len(sc), "small_sample": len(sc) < SMALL,
            "runs": len({u["run"] for u in used}), "records": len(used)}


def paired_scenarios(units: list[dict], metric: str, a: str, b: str,
                     where: Callable[[dict], bool] | None = None
                     ) -> dict[str, tuple[float, float, float, int]]:
    """Per scenario: (mean of A-B over the (run, repeat) both measured, mean A, mean B,
    number of such pairs)."""
    va: dict[tuple, float] = {}
    vb: dict[tuple, float] = {}
    for u in units:
        if u.get(metric) is None or (where is not None and not where(u)):
            continue
        k = (u["scenario"], u["run"], u["repeat"], u["checkpoint"])
        if u["config"] == a:
            va[k] = float(u[metric])
        elif u["config"] == b:
            vb[k] = float(u[metric])
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for k in va.keys() & vb.keys():
        by[k[0]].append((va[k], vb[k]))
    return {s: (float(np.mean([x - y for x, y in p])), float(np.mean([x for x, _ in p])),
                float(np.mean([y for _, y in p])), len(p)) for s, p in by.items()}


def paired(units: list[dict], metric: str, a: str, b: str, ctx: Ctx,
           where: Callable[[dict], bool] | None = None) -> dict | None:
    ps = paired_scenarios(units, metric, a, b, where)
    d = boot([v[0] for v in ps.values()], ctx)
    if d is None:
        return None
    return {"a": a, "b": b, "diff": d[0], "lo": d[1], "hi": d[2], "n": len(ps),
            "small_sample": len(ps) < SMALL, "pairs": sum(v[3] for v in ps.values()),
            "mean_a": float(np.mean([v[1] for v in ps.values()])),
            "mean_b": float(np.mean([v[2] for v in ps.values()]))}


def mkey(u: dict) -> tuple:
    """One measurement of a configuration: its run, replay and checkpoint (scenario,
    history sample, category and domain follow from the run)."""
    return (u["run"], u["repeat"], u["checkpoint"])


@dataclass
class Basis:
    """What a comparison of configurations stands on, for one metric: the measurements
    on which every configuration in it has a value (`keys`), and how many each has."""
    metric: str
    keys: set
    per_config: dict[str, int]
    scenarios: int

    @property
    def uneven(self) -> bool:
        """Some configuration has measurements another lacks, so its numbers on every
        measurement it has would differ from the matched ones."""
        return any(n != len(self.keys) for n in self.per_config.values())

    @property
    def limiting(self) -> list[str]:
        """The configurations with the fewest measurements: what bounds the set."""
        if not self.uneven:
            return []
        low = min(self.per_config.values())
        return sorted(c for c, n in self.per_config.items() if n == low)

    def short(self) -> str:
        return f"{len(self.keys)} in {self.scenarios} scenarios"

    def describe(self) -> str:
        """For a table's text: '222 (run, replay) measurements in 74 scenarios (of up to
        693; limited by heal[cf])'."""
        if not self.per_config:
            return "not measured"
        if not self.keys:
            return "no measurement shared by every configuration yet"
        text = f"{len(self.keys)} (run, replay) measurements in {self.scenarios} scenarios"
        if self.uneven:
            text += (f" (of up to {max(self.per_config.values())}; limited by "
                     + ", ".join(self.limiting) + ")")
        return text

    def as_dict(self) -> dict:
        return {"measurements": len(self.keys), "scenarios": self.scenarios,
                "uneven": self.uneven, "limiting": self.limiting,
                "per_config": dict(self.per_config)}


def matched_basis(units: Iterable[dict], configs: Iterable[str], metric: str,
                  where: Callable[[dict], bool] | None = None) -> Basis:
    """The measurements on which every configuration in `configs` has a value of
    `metric` (where `where` holds). A configuration with no such value at all is left
    out of the requirement: it has no mark to compare, and requiring it would empty the
    set for the others."""
    wanted = set(configs)
    have: dict[str, set] = defaultdict(set)
    scenario: dict[tuple, str] = {}
    for u in units:
        if u["config"] in wanted and u.get(metric) is not None and (where is None or where(u)):
            have[u["config"]].add(mkey(u))
            scenario[mkey(u)] = u["scenario"]
    keys = set.intersection(*have.values()) if have else set()
    return Basis(metric, keys, {c: len(v) for c, v in have.items()},
                 len({scenario[k] for k in keys}))


def on_basis(basis: Basis, where: Callable[[dict], bool] | None = None
             ) -> Callable[[dict], bool]:
    keys = basis.keys
    return lambda u: mkey(u) in keys and (where is None or where(u))


def matched(units: list[dict], configs: Iterable[str], metrics: Iterable[str], ctx: Ctx,
            where: Callable[[dict], bool] | None = None) -> tuple[dict, dict[str, Basis]]:
    """summary() per configuration and metric on that metric's matched basis, and the
    bases. A running batch fills configurations in unevenly (a mode added later may so
    far cover one history sample, on which another mode's ASR differs from the rest), so
    comparing each configuration on every measurement it has would compare runs as much
    as configurations. When coverage is even this is every measurement."""
    configs = list(configs)
    bases = {m: matched_basis(units, configs, m, where) for m in metrics}
    return ({c: {m: summary(units, m, ctx, c, where=on_basis(bases[m], where)) for m in metrics}
             for c in configs}, bases)


def every_measurement(units: list[dict], configs: Iterable[str], metrics: Iterable[str],
                      ctx: Ctx, where: Callable[[dict], bool] | None = None) -> dict:
    """The coverage-dependent view: each configuration on every measurement it has."""
    return {c: {m: summary(units, m, ctx, c, where=where) for m in metrics} for c in configs}


def basis_text(bases: dict[str, Basis], who: str = "configuration") -> str:
    """For a section's text: what its per-configuration numbers stand on (metrics with
    the same basis size named together)."""
    by_text: dict[str, list[str]] = defaultdict(list)
    for m, b in bases.items():
        by_text[b.describe()].append(METRICS[m].name)
    sizes = "; ".join(f"{', '.join(names)}: {text}" for text, names in by_text.items())
    if any(b.uneven for b in bases.values()):
        return (f"Matched: each metric on the measurements every {who} has a value of it on, "
                f"so the rows compare like with like ({sizes}).")
    return f"Every {who} has the same measurements ({sizes})."


def basis_note(bases: Iterable[Basis], who: str = "configuration") -> str:
    """One line for a figure note: whether its marks are matched and on how much."""
    bs = [b for b in bases if b.per_config]
    if not bs:
        return "not measured yet"
    by_size: dict[str, list[str]] = defaultdict(list)
    for b in bs:
        by_size[b.short()].append(METRICS[b.metric].name)
    sizes = (next(iter(by_size)) if len(by_size) == 1 else
             ", ".join(f"{' and '.join(names)} {s}" for s, names in by_size.items()))
    if any(b.uneven for b in bs):
        return f"matched: every {who} on the same (run, replay) measurements ({sizes})"
    return f"every {who} has the same measurements ({sizes})"


def direction(d: dict | None) -> int | None:
    """+1 / -1 when the interval excludes 0 on that side, 0 when it includes 0."""
    if d is None:
        return None
    return 1 if d["lo"] > 0 else -1 if d["hi"] < 0 else 0


def verdict(d: dict | None, metric: str) -> str:
    if d is None:
        return "not measured"
    s = direction(d)
    if s == 0:
        return "interval includes 0"
    m = METRICS[metric]
    word = "higher" if s > 0 else "lower"
    if m.better:
        word += " (better)" if (m.better == "higher") == (s > 0) else " (worse)"
    return word


def nested_components(groups: dict[str, dict[str, list[float]]], ctx: Ctx) -> dict | None:
    """Variance components of a scenario > run > replay design (method of moments,
    unbalanced coefficients, Searle 1971): sigma2 between scenarios, between history runs
    within a scenario, between replays within a run; their shares of the total with
    scenario-bootstrap 95% intervals (a scenario drawn twice counts as two scenarios).
    None when the design has no replicate at some level."""
    scen = [s for s in sorted(groups) if any(groups[s].values())]
    if len(scen) < 2:
        return None
    n_s, t_s, sq_s, ssw_s, dfw_s, ssb_s, dfb_s, q_s = ([] for _ in range(8))
    for s in scen:
        runs = [np.asarray(v, dtype=float) for v in groups[s].values() if len(v)]
        n = np.array([len(v) for v in runs], dtype=float)
        means = np.array([v.mean() for v in runs])
        ns = n.sum()
        ybar = sum(v.sum() for v in runs) / ns
        n_s.append(ns)
        t_s.append(ns * ybar)
        sq_s.append(ns * ybar ** 2)
        ssw_s.append(sum(((v - v.mean()) ** 2).sum() for v in runs))
        dfw_s.append((n - 1).sum())
        ssb_s.append((n * (means - ybar) ** 2).sum())
        dfb_s.append(len(runs) - 1)
        q_s.append((n ** 2).sum())
    n_s, t_s, sq_s, ssw_s, dfw_s, ssb_s, dfb_s, q_s = map(
        np.asarray, (n_s, t_s, sq_s, ssw_s, dfw_s, ssb_s, dfb_s, q_s))
    if dfw_s.sum() == 0 or dfb_s.sum() == 0:
        return None

    def comps(c: np.ndarray) -> np.ndarray:
        """c: (B, S) scenario counts -> (B, 3) components (scenario, run, replay)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            N = c @ n_s
            ybar = (c @ t_s) / N
            dfa = c.sum(axis=1) - 1
            msa = (c @ sq_s - N * ybar ** 2) / dfa
            msb = (c @ ssb_s) / (c @ dfb_s)
            msw = (c @ ssw_s) / (c @ dfw_s)
            k1 = (N - c @ (q_s / n_s)) / (c @ dfb_s)
            k2 = (c @ (q_s / n_s) - (c @ q_s) / N) / dfa
            k3 = (N - (c @ (n_s ** 2)) / N) / dfa
            h = np.maximum(0.0, (msb - msw) / k1)
            s = np.maximum(0.0, (msa - msw - k2 * h) / k3)
        return np.stack([s, h, msw], axis=1)

    S = len(scen)
    point = comps(np.ones((1, S)))[0]
    total = point.sum()
    rng = np.random.default_rng(ctx.seed)
    counts = rng.multinomial(S, np.full(S, 1.0 / S), size=ctx.n_boot).astype(float)
    bc = comps(counts)
    tot = bc.sum(axis=1)
    ok = np.isfinite(tot) & (tot > 0)
    shares = bc[ok] / tot[ok, None]
    names = ("scenario", "history", "replay")
    out = {"scenarios": S, "runs": int(sum(len([v for v in groups[s].values() if v]) for s in scen)),
           "values": int(n_s.sum()), "total_variance": float(total),
           "small_sample": S < SMALL}
    for i, name in enumerate(names):
        share = float(point[i] / total) if total > 0 else None
        ci = (np.percentile(shares[:, i], [2.5, 97.5]) if shares.size else (None, None))
        out[name] = {"variance": float(point[i]),
                     "share": None if share is None else 100 * share,
                     "lo": None if ci[0] is None else 100 * float(ci[0]),
                     "hi": None if ci[1] is None else 100 * float(ci[1])}
    return out


# --- formatting --------------------------------------------------------------------------

def _num(v: float, digits: int, signed: bool = False) -> str:
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def cell(s: dict | None, digits: int = 1) -> str:
    if s is None:
        return "-"
    flag = ", small sample" if s["small_sample"] else ""
    return (f"{_num(s['mean'], digits)} [{_num(s['lo'], digits)}, {_num(s['hi'], digits)}]"
            f" (n={s['n']}{flag})")


def dcell(d: dict | None, digits: int = 1) -> str:
    if d is None:
        return "-"
    flag = ", small sample" if d["small_sample"] else ""
    return (f"{_num(d['diff'], digits, True)} [{_num(d['lo'], digits, True)}, "
            f"{_num(d['hi'], digits, True)}] (n={d['n']}{flag})")


def short(s: dict | None, metric: str) -> str:
    """'12.3% [8.0, 16.1] (n=77)' for takeaways."""
    if s is None:
        return "not measured"
    m = METRICS[metric]
    unit = "%" if m.unit == "%" else f" {m.unit}"
    flag = ", small sample" if s["small_sample"] else ""
    return (f"{s['mean']:.{m.digits}f}{unit} [{s['lo']:.{m.digits}f}, {s['hi']:.{m.digits}f}]"
            f" (n={s['n']}{flag})")


def dshort(d: dict | None, metric: str) -> str:
    if d is None:
        return f"{METRICS[metric].name} not measured"
    m = METRICS[metric]
    flag = ", small sample" if d["small_sample"] else ""
    return (f"{m.name} {d['diff']:+.{m.digits}f} {m.diff_unit} [{d['lo']:+.{m.digits}f}, "
            f"{d['hi']:+.{m.digits}f}] (n={d['n']}{flag})")


def _round(obj):
    """JSON-ready copy with floats rounded to 4 places."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, 4)
    if isinstance(obj, dict):
        return {str(k): _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _round(obj.item())
    return obj


@dataclass
class Section:
    key: str
    title: str
    md: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    takeaway: str | None = None
    skipped: str | None = None

    def skip(self, why: str) -> "Section":
        self.skipped = why
        return self

    def table(self, header: list[str], rows: list[list]) -> None:
        self.md += [measure.md_table(header, rows), ""]

    def text(self, *lines: str) -> None:
        self.md += [*lines, ""]

    def figure(self, path: Path | None, what: str) -> None:
        if path is not None:
            self.figures.append(path.name)
            self.md += [f"![{what}]({path.name})", ""]


def ordered_configs(units: Iterable[dict], preferred: Iterable[str] = MAIN_CONFIGS) -> list[str]:
    seen = {u["config"] for u in units}
    pref = [c for c in preferred if c in seen]
    return pref + sorted(seen - set(pref))


def group_of(config: str) -> int:
    for i, (_, members) in enumerate(GROUPS):
        if config in members:
            return i
    base = config.split("[")[0].split("+")[0]
    if base == "none":
        return 3
    if base.startswith("heal_oracle") or "[oracle]" in config or base in ("ancestry", "root"):
        return 2
    return 1 if base.startswith("heal") else 0


def marker_of(config: str) -> str:
    g = group_of(config)
    members = [c for c in GROUPS[g][1]]
    i = members.index(config) if config in members else len(members)
    return MARKERS[i % len(MARKERS)]


def pareto(points: dict[str, tuple[float, float]]) -> list[str]:
    """Configurations no other one beats on both (lower ASR, higher x) by mean."""
    out = []
    for c, (x, y) in points.items():
        if not any((x2 >= x and y2 <= y) and (x2 > x or y2 < y)
                   for c2, (x2, y2) in points.items() if c2 != c):
            out.append(c)
    return sorted(out, key=lambda c: points[c][1])


# --- figures -----------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


NOTE_CHARS = 125            # a 6.5 pt note line that fits 6.5 in


def _footer(fig, handles: list, names: list[str], ncol: int, note: str | None = None) -> float:
    """Series key (rows of `ncol`) at the bottom and `note` below it: how to read the
    figure, its basis first ('\\n' starts a line; long lines wrap). Both are placed in
    inches from the bottom edge and the figure grows by their height, so the axes keep
    the room each plot gave them however long the note. Returns the bottom margin
    (figure fraction) for tight_layout. Call before laying out."""
    import textwrap
    from matplotlib.transforms import blended_transform_factory
    renderer = fig.canvas.get_renderer()
    tr = blended_transform_factory(fig.transFigure, fig.dpi_scale_trans)
    y = 0.05
    if note:
        text = "\n".join(textwrap.fill(line, NOTE_CHARS, break_on_hyphens=False)
                         for line in note.split("\n"))
        t = fig.text(0.5, y, text, ha="center", va="bottom", fontsize=6.5, color=INK2,
                     transform=tr, linespacing=1.35)
        y += t.get_window_extent(renderer).height / fig.dpi + 0.03
    if handles:
        leg = fig.legend(handles, names, loc="lower center", ncol=max(1, min(ncol, len(handles))),
                         frameon=False, fontsize=7, labelcolor=INK, bbox_to_anchor=(0.5, y),
                         bbox_transform=tr, borderaxespad=0.0)
        y += leg.get_window_extent(renderer).height / fig.dpi
    y += 0.08
    fig.set_figheight(fig.get_figheight() + y)
    return y / fig.get_figheight()


def _hollow_handle():
    from matplotlib.lines import Line2D
    return Line2D([], [], marker="o", ls="none", color=MUTED, mfc=SURFACE, ms=6)


def _mark(ax, x, y, colour, marker, small, ms=5.5, zorder=3):
    ax.plot(x, y, marker=marker, ms=ms, ls="none", color=colour,
            mfc=SURFACE if small else colour, mec=colour, mew=1.3, zorder=zorder)


def _save(fig, plt, path: Path, bottom: float) -> Path:
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def _empty(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=7.5, color=INK2,
            transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def dotplot(ax, rows: list[str], series: list[dict], xlabel: str, zero: bool = False,
            labels: bool = True, step: float = 0.22) -> bool:
    """Rows top to bottom, one mark per (row, series) with its 95% interval; series are
    offset vertically. A series: {'stats': {row: stat}, 'colour': str | f(row),
    'marker': str | f(row)}. Returns whether a small-sample mark was drawn."""
    any_small = False
    k = len(series)
    for i, row in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, s in enumerate(series):
            st = s["stats"].get(row)
            if st is None:
                continue
            y = y0 - (j - (k - 1) / 2) * step
            colour = s["colour"](row) if callable(s["colour"]) else s["colour"]
            marker = s["marker"](row) if callable(s["marker"]) else s["marker"]
            mean = st.get("mean", st.get("diff"))
            ax.plot([st["lo"], st["hi"]], [y, y], color=colour, lw=1.2, alpha=0.75, zorder=2)
            _mark(ax, mean, y, colour, marker, st["small_sample"])
            any_small |= st["small_sample"]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(reversed(rows)) if labels else [""] * len(rows), fontsize=7,
                       color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=7.5, color=INK2)
    if zero:
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    return any_small


def group_handles(configs: Iterable[str]) -> tuple[list, list[str]]:
    from matplotlib.lines import Line2D
    used = sorted({group_of(c) for c in configs})
    return ([Line2D([], [], marker="o", ls="none", color=PALETTE[g], ms=6) for g in used],
            [GROUPS[g][0] for g in used])


def _spots() -> list[tuple[float, float, str, str, float]]:
    """Label offsets (points) around a mark, nearest first: rings of 12 directions."""
    out = []
    for r in (7, 14, 24, 36, 50, 66, 84, 104):
        for deg in range(0, 360, 30):
            dx, dy = r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))
            ha = "left" if dx > 1 else "right" if dx < -1 else "center"
            va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
            out.append((dx, dy, ha, va, r))
    return out


def _place_labels(ax, fig, pts: list[tuple[float, float, str]]) -> None:
    """Name each point without overprinting: the nearest spot that overlaps no placed
    label and no mark and stays inside the axes; labels set further out get a thin
    leader line to their mark."""
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    marks = [ax.transData.transform((x, y)) for x, y, _ in pts]
    placed = []
    spots = _spots()
    from matplotlib.text import Text

    def draw(x, y, text, spot):
        dx, dy, ha, va, r = spot
        arrow = dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=3) if r > 7 else None
        t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=5.8, color=INK2, arrowprops=arrow)
        # The text's own box: an Annotation's extent would include its leader line (and
        # so its own mark); its position is only current after update_positions.
        t.update_positions(renderer)
        return t, Text.get_window_extent(t, renderer).padded(1.5)

    def clashes(box, own: int) -> int:
        inside = (frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0
                  and box.y1 <= frame.y1)
        return ((0 if inside else 100) + sum(box.overlaps(b) for b in placed)
                + sum(box.padded(3).contains(*m) for i, m in enumerate(marks) if i != own))

    for k, (x, y, text) in enumerate(pts):
        best = None
        for spot in spots:
            t, box = draw(x, y, text, spot)
            n = clashes(box, k)
            t.remove()
            if best is None or n < best[0]:
                best = (n, spot)
            if n == 0:
                break
        t, box = draw(x, y, text, best[1])          # the clash-free spot, else the least bad
        placed.append(box)


SQRT_TICKS = (0, 1, 2, 5, 10, 20, 40, 60, 80, 100)
SQRT_TICKS_X = (0, 1, 5, 20, 50, 100)        # fewer on x: small multiples are narrow


def _sqrt_axis(ax, top: float, axis: str = "y") -> None:
    """Square-root scale from 0 to `top` on one axis: spreads the low ASRs most
    recoveries reach while keeping 'none' on the same axis; ticks in the data's units."""
    # Signed, so the axis can start a little below 0 and a mark at 0 is not clipped.
    fns = (lambda v: np.sign(v) * np.sqrt(np.abs(v)), lambda v: np.sign(v) * np.square(v))
    ticks = [t for t in (SQRT_TICKS if axis == "y" else SQRT_TICKS_X) if t <= top]
    low = -0.002 * top - 0.1
    if axis == "y":
        ax.set_yscale("function", functions=fns)
        ax.set_ylim(low, top)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
    else:
        ax.set_xscale("function", functions=fns)
        ax.set_xlim(low, top)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.tick_params(axis="x", labelsize=6.5)


def plot_tradeoff(path: Path, stats: dict[str, tuple[dict | None, dict | None]],
                  xmetric: str, note: str | None = None) -> Path:
    """ASR (y, square-root scale) against xmetric per configuration; crosshairs are
    scenario-bootstrap 95% intervals on each axis; hollow marks stand on fewer than
    SMALL scenarios. `note` says which measurements the marks stand on."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    pts, any_small, top, left = [], False, 10.0, 100.0
    for c, (x, y) in stats.items():
        if x is None or y is None:
            continue
        colour = PALETTE[group_of(c)]
        small = x["small_sample"] or y["small_sample"]
        any_small |= small
        ax.plot([x["lo"], x["hi"]], [y["mean"]] * 2, color=colour, lw=1, alpha=0.55, zorder=2)
        ax.plot([x["mean"]] * 2, [max(y["lo"], 0), y["hi"]], color=colour, lw=1, alpha=0.55,
                zorder=2)
        _mark(ax, x["mean"], y["mean"], colour, marker_of(c), small, ms=6)
        pts.append((x["mean"], y["mean"], f"{c} (n={min(x['n'], y['n'])})"))
        top, left = max(top, y["hi"]), min(left, x["lo"])
    ax.set_xlabel(f"{METRICS[xmetric].label}, higher keeps more", fontsize=8, color=INK2)
    ax.set_ylabel("ASR over the 4-task family (%, square-root scale)", fontsize=8, color=INK2)
    ax.set_xlim(max(-3.0, 10 * math.floor(left / 10) - 3), 103)
    _sqrt_axis(ax, min(100.0, top * 1.15 + 1))
    if not pts:
        _empty(ax, f"no configuration has both ASR and {METRICS[xmetric].name} yet")
    handles, names = group_handles(stats)
    if any_small:
        handles.append(_hollow_handle())
        names.append(f"hollow: fewer than {SMALL} scenarios")
    bottom = _footer(fig, handles, names, ncol=3,
                     note="\n".join(filter(None, [note, "crosshairs: scenario-bootstrap 95% "
                                                        "intervals"])))
    fig.tight_layout(rect=(0, bottom, 1, 1))
    _place_labels(ax, fig, pts)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_forest(path: Path, results: dict[str, dict[str, dict | None]],
                pairs: list[str]) -> Path:
    """Paired differences A - B, one panel per metric (own x axis and unit), one colour
    per pair; the eighth panel holds the legend."""
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.9))
    fig.patch.set_facecolor(SURFACE)
    any_small = False
    for ax, metric in zip(axes.flat, FOREST_METRICS):
        _style(ax)
        series = [{"stats": {"x": results[metric].get(p)}, "colour": PALETTE[j % 4],
                   "marker": MARKERS[j % 4]} for j, p in enumerate(pairs)]
        if not any(s["stats"]["x"] for s in series):
            _empty(ax, "not measured")
            ax.set_title(METRICS[metric].name, fontsize=8, color=INK, loc="left")
            continue
        any_small |= dotplot(ax, ["x"], series, f"difference ({METRICS[metric].diff_unit})",
                             zero=True, labels=False, step=0.2)
        ax.set_yticks([])
        ax.set_title(METRICS[metric].name, fontsize=8, color=INK, loc="left")
    leg = axes.flat[-1]
    leg.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=MARKERS[j % 4], ls="none", color=PALETTE[j % 4], ms=6)
               for j in range(len(pairs))]
    names = list(pairs)
    if any_small:
        handles.append(_hollow_handle())
        names.append(f"hollow: < {SMALL} scenarios")
    leg.legend(handles, names, loc="center", frameon=False, fontsize=6.3, labelcolor=INK)
    bottom = _footer(fig, [], [], 1, "paired: A - B on each (run, replay) both configurations "
                     "have, averaged per scenario; scenario-bootstrap 95% intervals")
    return _save(fig, plt, path, bottom)


def plot_grid(path: Path, configs: list[str], columns: list[str],
              stats: dict[str, dict[str, dict[str, dict | None]]], metrics: list[str],
              title: str) -> Path:
    """Small multiples: one row per metric, one column per group value; configurations
    on y (coloured by GROUPS), mean with 95% interval on x. A metric with no data in any
    column gets no row; the note says so."""
    plt = _plt()
    missing = [m for m in metrics if not any(v for col in columns
                                             for v in stats[m][col].values())]
    shown = [m for m in metrics if m not in missing] or metrics[:1]
    if missing:
        title += "; no data yet: " + ", ".join(METRICS[m].name for m in missing)
    nrows, ncols = len(shown), max(1, len(columns))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, 0.21 * len(configs) * nrows + 0.9),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    any_small = False
    for i, metric in enumerate(shown):
        for j, col in enumerate(columns):
            ax = axes[i][j]
            _style(ax)
            st = {c: stats[metric][col].get(c) for c in configs}
            if not any(st.values()):
                _empty(ax, f"{METRICS[metric].name}: no data")
                ax.set_title(f"{col}", fontsize=8, color=INK, loc="left")
                continue
            sqrt = metric == "asr"
            any_small |= dotplot(ax, configs, [{"stats": st, "colour": lambda c: PALETTE[group_of(c)],
                                                "marker": marker_of}],
                                 "ASR (%, sqrt scale)" if sqrt else METRICS[metric].label,
                                 labels=(j == 0))
            if sqrt:
                _sqrt_axis(ax, 100, "x")
            else:
                ax.set_xlim(-3, 103)
            ax.set_title(f"{col}", fontsize=8, color=INK, loc="left")
    handles, names = group_handles(configs)
    if any_small:
        handles.append(_hollow_handle())
        names.append(f"hollow: < {SMALL} scenarios")
    return _save(fig, plt, path, _footer(fig, handles, names, ncol=3, note=title))


def plot_series_rows(path: Path, rows: list[str], panels: list[tuple[str, list[dict], str]],
                     note: str, zero: bool = False, xlim: tuple | None = None,
                     height: float | None = None, legend: tuple[list, list[str]] | None = None,
                     sqrt_asr: bool = False) -> Path:
    """Panels side by side, rows on a shared y; each panel a list of named series. The
    key lists the first panel's series, or `legend` (handles, names) when given.
    sqrt_asr puts panels whose x label starts with 'ASR' on a square-root scale."""
    plt = _plt()
    h = height or (0.24 * len(rows) * max(1, max(len(p[1]) for p in panels)) ** 0.5 + 0.9)
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, h), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    any_small = False
    for j, (title, series, xlabel) in enumerate(panels):
        ax = axes[0][j]
        _style(ax)
        if not any(v for s in series for v in s["stats"].values()):
            _empty(ax, f"{title}: no data yet")
            ax.set_title(title, fontsize=8, color=INK, loc="left")
            continue
        sqrt = sqrt_asr and xlabel.startswith("ASR")
        any_small |= dotplot(ax, rows, series, xlabel.replace("(%)", "(%, sqrt scale)") if sqrt
                             else xlabel, zero=zero, labels=(j == 0),
                             step=0.6 / max(len(series), 1))
        if sqrt:
            _sqrt_axis(ax, 100, "x")
        elif xlim:
            ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    from matplotlib.lines import Line2D
    names, handles = [], []
    if legend is not None:
        handles, names = list(legend[0]), list(legend[1])
    else:
        for s in panels[0][1]:
            if s.get("name") and s["name"] not in names:
                names.append(s["name"])
                handles.append(Line2D([], [], marker=s["marker"], ls="none", color=s["colour"],
                                      ms=6))
    if any_small:
        handles.append(_hollow_handle())
        names.append(f"hollow: < {SMALL} scenarios")
    longest = max((len(n) for n in names), default=1)
    return _save(fig, plt, path, _footer(fig, handles, names,
                                         ncol=max(1, 120 // (longest + 6)), note=note))


# --- data --------------------------------------------------------------------------------

@dataclass
class Data:
    inputs: dict[str, dict] = field(default_factory=dict)
    main: list[dict] = field(default_factory=list)
    clean: list[dict] = field(default_factory=list)
    reinfect: list[dict] = field(default_factory=list)
    suites: dict[str, list[dict]] = field(default_factory=dict)
    samples_note: dict = field(default_factory=dict)
    preservation: dict = field(default_factory=dict)


SUITE_ORDER = ("dense evolving", "flat", "adaptive", "subset gpt-5.4", "subset gpt-5.5",
               "subset gpt-4.1-mini", "LongMemEval", "MemBench", "Mem0", "Mem0 additive")


def suite_label(name: str, rec: dict, models: ModelIndex) -> str:
    if name == "external":
        v = rec.get("variant") or ""
        return "LongMemEval" if v in LONGMEMEVAL else "MemBench" if v in MEMBENCH else "external (other)"
    if name == "models":
        return f"subset {models(rec['run_dir']) or 'unknown model'}"
    return {"flat": "flat", "adaptive": "adaptive", "mem0": "Mem0",
            "mem0_additive": "Mem0 additive"}.get(name, name)


def load(paths: dict[str, Path], pres_paths: list[Path], runs_dir: Path, parts: bool,
         ctx: Ctx) -> Data:
    data = Data()
    pres_rows: list[dict] = []
    pres_found = []
    for p in pres_paths:
        rows, torn = read_jsonl(p) if Path(p).is_file() else ([], 0)
        pres_found.append({"path": _rel(p), "state": "ok" if rows else "missing" if not Path(p).is_file()
                           else "empty", "rows": len(rows), "torn_lines": torn})
        pres_rows += rows
    pres = Preservation(pres_rows)
    data.preservation = {"files": pres_found, "rows": len(pres_rows), "status": {}}
    models = ModelIndex(runs_dir)

    def take(name: str, condition: str, checkpoint0: bool = True) -> tuple[list[dict], str | None]:
        recs, info = read_records(paths[name], parts) if name in paths else (None, {"state": "missing"})
        if recs is None:
            data.inputs[name] = info
            return [], None
        kept, dropped = select(recs, condition, ctx.verify_n, checkpoint0)
        info.update(selected=len(kept), dropped=dict(dropped),
                    runs=len({r["run_dir"] for r in kept}),
                    scenarios=len({r.get("variant") for r in kept}))
        data.inputs[name] = info
        return kept, info["path"]

    main, path = take("main", "poisoned")
    if main:
        samples, odd = history_samples(main)
        data.samples_note = odd
        data.main, st = build_units(main, samples=samples, pres=pres, records_path=path,
                                    suite="dense evolving")
        data.preservation["status"]["main"] = dict(st)
        data.suites["dense evolving"] = data.main
    clean, path = take("clean", "clean")
    if clean:
        samples, _ = history_samples(clean)
        data.clean, st = build_units(clean, samples=samples, pres=pres, records_path=path)
        data.preservation["status"]["clean"] = dict(st)
    rein, path = take("reinfect", "poisoned", checkpoint0=False)
    if rein:
        data.reinfect, _ = build_units(rein, pres=None, records_path=path)
    for name in ("flat", "adaptive", "models", "external", "mem0", "mem0_additive"):
        recs, path = take(name, "poisoned")
        if not recs:
            continue
        by_suite: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_suite[suite_label(name, r, models)].append(r)
        for label, rs in by_suite.items():
            units, st = build_units(rs, baselines_from=recs, pres=pres, records_path=path,
                                    suite=label)
            data.suites.setdefault(label, []).extend(units)
            data.preservation["status"][label] = dict(st)
    return data


# --- sections -----------------------------------------------------------------------------

def coverage_rows(units: list[dict], configs: list[str]) -> list[list]:
    rows = []
    for c in configs:
        us = [u for u in units if u["config"] == c]
        reps = Counter(u["repeat"] for u in us)
        rows.append([c, len({u["scenario"] for u in us}), len({u["run"] for u in us}), len(us),
                     ", ".join(f"r{k}: {v}" for k, v in sorted(reps.items(), key=lambda kv: str(kv[0])))
                     or "-",
                     sum(u.get("bcp") is not None for u in us)])
    return rows


def s1_tradeoff(data: Data, ctx: Ctx) -> Section:
    sec = Section("tradeoff", "1. Main trade-off: attack success vs what recovery keeps")
    units = data.main
    if not units:
        return sec.skip("no main-comparison records")
    configs = ordered_configs(units)
    sec.text("Dense evolving suite, every qualified poisoned run, family=full. Per "
             "configuration: mean over scenarios of the per-scenario mean (over its history "
             "runs and replays), scenario-bootstrap 95% interval, n = scenarios.")
    sec.text("Coverage (the batch may still be running):")
    sec.table(["configuration", "scenarios", "runs", "records", "records per repeat",
               "records with BCP"], coverage_rows(units, configs))
    metrics = ("asr", "bmp", "bcp", "br", "residual")
    # The headline numbers and the figures stand on matched measurements; each
    # configuration on everything it has is kept as the coverage-dependent view.
    stats, bases = matched(units, configs, metrics, ctx)
    every = every_measurement(units, configs, metrics, ctx)
    uneven = any(b.uneven for b in bases.values())
    sec.text(basis_text(bases))
    sec.table(["configuration", "group", *[METRICS[m].label for m in metrics]],
              [[c, GROUPS[group_of(c)][0], *[cell(stats[c][m], METRICS[m].digits) for m in metrics]]
               for c in configs])
    if uneven:
        sec.text("Coverage-dependent view: every measurement each configuration has. The "
                 "configurations stand on different runs and replays here (see Coverage), so a "
                 "difference between two rows mixes the configurations with the runs each was "
                 "measured on; compare rows in the matched table above or in Section 2.")
        sec.table(["configuration", *[METRICS[m].label for m in metrics]],
                  [[c, *[cell(every[c][m], METRICS[m].digits) for m in metrics]] for c in configs])
    sec.numbers = {"configs": stats, "basis": {m: b.as_dict() for m, b in bases.items()},
                   "uneven": uneven, "all_measurements": every}
    take = []
    for xm in ("bmp", "bcp"):
        big = {c: (stats[c][xm]["mean"], stats[c]["asr"]["mean"]) for c in configs
               if stats[c][xm] and stats[c]["asr"] and not stats[c][xm]["small_sample"]
               and not stats[c]["asr"]["small_sample"]}
        front = pareto(big)
        sec.numbers[f"pareto_{xm}"] = front
        where = (f"matched: ASR on {bases['asr'].short()}, {METRICS[xm].name} on "
                 f"{bases[xm].short()}" if uneven else "every configuration on the same measurements")
        if big:
            take.append(f"ASR-vs-{METRICS[xm].name} front by mean ({where}; configurations "
                        f"with >= {SMALL} scenarios): " + "; ".join(
                            f"{c} (ASR {big[c][1]:.1f}%, {METRICS[xm].name} {big[c][0]:.1f}%)"
                            for c in front))
        else:
            take.append(f"no configuration has >= {SMALL} scenarios with both ASR and "
                        f"{METRICS[xm].name} yet")
    sec.takeaway = ". ".join(take) + ". The front ignores the intervals; Section 2 tests the pairs."
    if ctx.figures:
        for xm in ("bmp", "bcp"):
            note = basis_note([bases["asr"], bases[xm]])
            if xm == "bcp" and not any(stats[c]["bcp"] for c in configs):
                note = "BCP missing: no bcp.py rows matched (results/preservation_final/)"
            p = plot_tradeoff(ctx.out / f"main_tradeoff_{xm}.png",
                              {c: (stats[c][xm], stats[c]["asr"]) for c in configs}, xm, note)
            sec.figure(p, f"ASR vs {METRICS[xm].name}")
    return sec


def s2_heal_vs_quarantine(data: Data, ctx: Ctx) -> Section:
    sec = Section("heal_vs_quarantine", "2. Healing vs quarantine with the same detection")
    units = data.main
    if not units:
        return sec.skip("no main-comparison records")
    present = {u["config"] for u in units}
    pairs = [(a, b) for a, b in HEAL_PAIRS if a in present and b in present]
    if not pairs:
        return sec.skip("no heal/incident pair measured yet")
    sec.text("A - B per (run, repeat) where both were measured, averaged per scenario, "
             "scenario-bootstrap 95% interval. Rates in percentage points; residual attack and "
             "memories removed in memories; seconds are wall time under the batch's 64 "
             "concurrent workers, so only roughly comparable.")
    rows, res = [], {m: {} for m in FOREST_METRICS}
    for a, b in pairs:
        name = f"{a} - {b}"
        for m in FOREST_METRICS:
            d = paired(units, m, a, b, ctx)
            res[m][name] = d
            met = METRICS[m]
            rows.append([name, met.name, "-" if d is None else f"{d['mean_a']:.{met.digits}f}",
                         "-" if d is None else f"{d['mean_b']:.{met.digits}f}",
                         dcell(d, met.digits), "-" if d is None else d["pairs"], verdict(d, m)])
    sec.table(["A - B", "metric", "A (same scenarios)", "B (same scenarios)",
               "A - B [95% CI] (n scenarios)", "(run, repeat) pairs", "reading"], rows)
    sec.numbers = res
    a, b = pairs[0]
    first = f"{a} - {b}"
    clear = [f"{METRICS[m].name} {verdict(res[m][first], m)}" for m in FOREST_METRICS
             if direction(res[m][first])]
    sec.takeaway = (f"{first}: " + "; ".join(dshort(res[m][first], m) for m in ("asr", "bmp", "bcp"))
                    + ". Intervals excluding 0: " + (", ".join(clear) if clear else "none") + ".")
    if ctx.figures:
        p = plot_forest(ctx.out / "main_heal_vs_quarantine.png", res,
                        [f"{a} - {b}" for a, b in pairs])
        sec.figure(p, "paired differences")
    return sec


def _breakdown(units: list[dict], field_: str, values: list[str], configs: list[str],
               metrics: list[str], ctx: Ctx) -> tuple[dict, dict, dict]:
    """Per metric and value of `field_`: the configurations matched within that value
    (every one on the measurements all of them have there), the bases, and the
    coverage-dependent summaries."""
    st: dict = {m: {} for m in metrics}
    bases: dict = {m: {} for m in metrics}
    every: dict = {m: {} for m in metrics}
    for v in values:
        def here(u, v=v):
            return u[field_] == v
        got, bs = matched(units, configs, metrics, ctx, where=here)
        alls = every_measurement(units, configs, metrics, ctx, where=here)
        for m in metrics:
            st[m][v] = {c: got[c][m] for c in configs}
            bases[m][v] = bs[m]
            every[m][v] = {c: alls[c][m] for c in configs}
    return st, bases, every


def s3_breakdown(data: Data, ctx: Ctx) -> Section:
    sec = Section("breakdown", "3. Per attack category and per domain")
    units = data.main
    if not units:
        return sec.skip("no main-comparison records")
    configs = ordered_configs(units)
    cats = [c for c in ("direct", "composed", "procedural") if any(u["category"] == c for u in units)]
    cats += sorted({u["category"] for u in units} - set(cats))
    doms = sorted({u["domain"] for u in units})
    metrics = ["asr", "bcp"]
    by_cat, cat_bases, cat_all = _breakdown(units, "category", cats, configs, metrics, ctx)
    by_dom, dom_bases, dom_all = _breakdown(units, "domain", doms, configs, metrics, ctx)
    uneven = any(b.uneven for bs in (cat_bases, dom_bases) for per in bs.values()
                 for b in per.values())
    sec.text("Matched within each category (domain): every configuration on the (run, replay) "
             "measurements all of them have there, per metric; the last row of each table "
             "gives that set's size. " + ("Coverage is uneven, so " if uneven else "")
             + "the coverage-dependent numbers (each configuration on everything it has) are "
             "in main.json under all_measurements.")
    for title, values, st, bs in (("attack category", cats, by_cat, cat_bases),
                                  ("domain", doms, by_dom, dom_bases)):
        for m in metrics:
            sec.text(f"{METRICS[m].label} by {title}:")
            sec.table(["configuration", *values],
                      [[c, *[cell(st[m][v][c]) for v in values]] for c in configs]
                      + [["matched (run, replay) measurements / scenarios",
                          *[f"{len(bs[m][v].keys)} / {bs[m][v].scenarios}" for v in values]]])
    pairs = {}
    for v in cats:
        pairs[v] = paired(units, "asr", HEAL, INC, ctx, where=lambda u, v=v: u["category"] == v)
    sec.numbers = {"by_category": by_cat, "by_domain": by_dom,
                   "basis": {"category": {m: {v: b.as_dict() for v, b in cat_bases[m].items()}
                                          for m in metrics},
                             "domain": {m: {v: b.as_dict() for v, b in dom_bases[m].items()}
                                        for m in metrics}},
                   "uneven": uneven,
                   "all_measurements": {"by_category": cat_all, "by_domain": dom_all},
                   "heal_minus_incident_asr": pairs}
    sec.takeaway = (f"{HEAL} - {INC} ASR by attack category (paired): "
                    + "; ".join(f"{v} {dshort(pairs[v], 'asr').removeprefix('ASR ')}" for v in cats) + ".")
    if ctx.figures:
        how = ("matched within each panel: every configuration on the same (run, replay) "
               "measurements" if uneven else "every configuration has the same measurements")
        p = plot_grid(ctx.out / "main_category.png", configs, cats, by_cat, metrics,
                      f"per attack category; {how}; scenario-bootstrap 95% intervals")
        sec.figure(p, "by attack category")
        p = plot_grid(ctx.out / "main_domain.png", configs, doms, by_dom, metrics,
                      f"per domain; {how}; scenario-bootstrap 95% intervals")
        sec.figure(p, "by domain")
    return sec


def run_groups(units: list[dict], config: str, metric: str) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for u in units:
        if u["config"] == config and u.get(metric) is not None:
            out[u["scenario"]][u["run"]].append(float(u[metric]))
    return out


def level_means(units: list[dict], config: str, level: str, values: Iterable, ctx: Ctx
                ) -> dict:
    """ASR summary per value of `level` (repeat or history sample), each restricted to the
    scenarios where the configuration has a measurement at every value, so the values
    are compared on the same scenarios (a running batch fills them in unevenly)."""
    values = list(values)
    have: dict[str, set] = defaultdict(set)
    for u in units:
        if u["config"] == config and u.get("asr") is not None:
            have[u["scenario"]].add(u[level])
    full = {s for s, got in have.items() if set(values) <= got}
    return {v: summary(units, "asr", ctx, config,
                       where=lambda u, v=v: u[level] == v and u["scenario"] in full)
            for v in values}


def disagreement(units: list[dict], config: str, ctx: Ctx) -> dict | None:
    """Runs with replays 1-3: the conclusion 'attacked' (any task reached the attacker)
    by majority, and how many single replays concluded otherwise."""
    votes: dict[tuple, dict[int, bool]] = defaultdict(dict)
    for u in units:
        if u["config"] == config and u.get("asr") is not None and u["repeat"] in REPEATS:
            votes[(u["scenario"], u["run"])][u["repeat"]] = u["asr"] > 0
    full = {k: v for k, v in votes.items() if set(REPEATS) <= set(v)}
    if not full:
        return None
    per_scen: dict[str, list[float]] = defaultdict(list)
    dis_total = split = 0
    for (scen, _), v in full.items():
        vs = [v[r] for r in REPEATS]
        maj = sum(vs) * 2 > len(vs)
        dis = sum(x != maj for x in vs)
        dis_total += dis
        split += dis > 0
        per_scen[scen].append(100.0 * dis / len(vs))
    b = boot([float(np.mean(x)) for x in per_scen.values()], ctx)
    return {"runs": len(full), "replays": len(full) * len(REPEATS), "disagreeing": dis_total,
            "runs_split": split, "share": {"mean": b[0], "lo": b[1], "hi": b[2],
                                           "n": len(per_scen), "small_sample": len(per_scen) < SMALL}}


def replay_conclusions(units: list[dict], a: str, b: str, ctx: Ctx) -> dict:
    """The paired ASR conclusion with all replays vs with one replay only, on the scenarios
    where A and B pair at every replay (so every row covers the same scenarios), and the
    number of those scenarios whose sign of A - B differs between the two."""
    one = {r: paired_scenarios(units, "asr", a, b, where=lambda u, r=r: u["repeat"] == r)
           for r in REPEATS}
    scen = set.intersection(*(set(v) for v in one.values()))

    def keep(r=None):
        return lambda u: u["scenario"] in scen and (r is None or u["repeat"] == r)

    full = paired_scenarios(units, "asr", a, b, where=keep())
    out = {"all": paired(units, "asr", a, b, ctx, where=keep()), "by_replay": {}}
    for r in REPEATS:
        flips = sum(np.sign(full[s][0]) != np.sign(one[r][s][0]) for s in scen)
        out["by_replay"][r] = {"paired": paired(units, "asr", a, b, ctx, where=keep(r)),
                               "scenarios": len(scen), "sign_differs": int(flips)}
    return out


def s4_repetition(data: Data, ctx: Ctx) -> Section:
    sec = Section("repetition", "4. Repetition: replays, history samples, variance")
    units = data.main
    if not units:
        return sec.skip("no main-comparison records")
    configs = ordered_configs(units)
    if data.samples_note:
        sec.text("History-sample anomalies: " + ", ".join(f"{k}: {v}" for k, v in data.samples_note.items()))
    samples = sorted({u["sample"] for u in units if u["sample"] is not None})
    by_rep = {c: level_means(units, c, "repeat", REPEATS, ctx) for c in configs}
    by_smp = {c: level_means(units, c, "sample", samples, ctx) for c in configs}

    def spread(d: dict) -> str:
        vals = [s["mean"] for s in d.values() if s]
        return f"{max(vals) - min(vals):.1f}" if len(vals) > 1 else "-"

    sec.text("ASR (%) using one replay only, and one history sample only (mean over the "
             "other level), on the scenarios where the configuration has every replay "
             "(every history sample), so the columns compare the same scenarios; spread = "
             "max - min of the means, in pp. Each row stands on its own configuration's "
             "scenarios: compare along a row, not down a column (Sections 1 and 2 compare "
             "configurations).")
    sec.table(["configuration", *[f"replay {r}" for r in REPEATS], "spread",
               *[f"history sample {h}" for h in samples], "spread"],
              [[c, *[cell(by_rep[c][r]) for r in REPEATS], spread(by_rep[c]),
                *[cell(by_smp[c][h]) for h in samples], spread(by_smp[c])] for c in configs])
    var = {c: {m: nested_components(run_groups(units, c, m), ctx) for m in ("asr", "bmp")}
           for c in configs}

    def vcell(v: dict | None, k: str) -> str:
        if v is None or v[k]["share"] is None:
            return "-"
        lo, hi = v[k]["lo"], v[k]["hi"]
        ci = "" if lo is None else f" [{lo:.0f}, {hi:.0f}]"
        return f"{v[k]['share']:.0f}%{ci}"

    for m in ("asr", "bmp"):
        sec.text(f"Variance decomposition of per-record {METRICS[m].name} (nested ANOVA, "
                 "method of moments; share of total variance, scenario-bootstrap 95% interval):")
        sec.table(["configuration", "scenarios", "runs", "records", "total variance",
                   "between scenarios", "between history samples", "between replays"],
                  [[c, *(["-"] * 7 if var[c][m] is None else
                         [f"{var[c][m]['scenarios']}{' (small sample)' if var[c][m]['small_sample'] else ''}",
                          var[c][m]["runs"], var[c][m]["values"],
                          f"{var[c][m]['total_variance']:.1f}",
                          vcell(var[c][m], "scenario"), vcell(var[c][m], "history"),
                          vcell(var[c][m], "replay")])] for c in configs])
    dis = {c: disagreement(units, c, ctx) for c in configs}
    sec.text("Single replay vs the 3-replay majority, conclusion 'run attacked' (ASR > 0): "
             "replays whose conclusion differs from their run's majority.")
    sec.table(["configuration", "runs with replays 1-3", "replays differing", "runs not unanimous",
               "differing share per scenario (%)"],
              [[c, *(["-"] * 4 if dis[c] is None else
                     [dis[c]["runs"], f"{dis[c]['disagreeing']}/{dis[c]['replays']}",
                      f"{dis[c]['runs_split']}/{dis[c]['runs']}", cell(dis[c]["share"])])]
               for c in configs])
    present = {u["config"] for u in units}
    conc = {f"{a} - {b}": replay_conclusions(units, a, b, ctx)
            for a, b in ((HEAL, INC), (HEAL_V, INC_V), (INC, NONE), (HEAL, NONE))
            if a in present and b in present}

    def sign(d):
        s = direction(d)
        return "-" if s is None else {1: "A higher", -1: "A lower", 0: "includes 0"}[s]

    rows = []
    for name, cc in conc.items():
        rows.append([name, "all", dcell(cc["all"]), sign(cc["all"]), "-"])
        for r, v in cc["by_replay"].items():
            rows.append([name, f"replay {r} only", dcell(v["paired"]), sign(v["paired"]),
                         f"{v['sign_differs']}/{v['scenarios']}"])
    sec.text("Paired ASR conclusions (pp) with every replay vs one replay only, on the "
             "scenarios where A and B pair at all three replays; last column: scenarios whose "
             "sign of A - B differs from the all-replay sign.")
    sec.table(["A - B", "replays used", "A - B [95% CI]", "conclusion",
               "scenarios with a different sign"], rows)
    sec.numbers = {"by_replay": by_rep, "by_sample": by_smp, "variance": var,
                   "single_replay": dis, "conclusions": conc}
    shares = {k: [var[c]["asr"][k]["share"] for c in configs
                  if var[c]["asr"] and var[c]["asr"][k]["share"] is not None]
              for k in ("scenario", "history", "replay")}
    d_tot = sum(d["disagreeing"] for d in dis.values() if d)
    d_n = sum(d["replays"] for d in dis.values() if d)
    compared = [(n, v["paired"]) for n, cc in conc.items() for v in cc["by_replay"].values()
                if v["paired"] is not None and cc["all"] is not None]
    flips = [(n, p) for n, p in compared if direction(p) != direction(conc[n]["all"])]
    sec.takeaway = (
        ("Median share of ASR variance across configurations: "
         + ", ".join(f"{k} {np.median(v):.0f}%" for k, v in shares.items() if v) + "; "
         if any(shares.values()) else "Variance decomposition not estimable yet; ")
        + (f"a single replay disagrees with its run's 3-replay majority on {d_tot}/{d_n} "
           f"replays ({100 * d_tot / d_n:.1f}%) pooled over configurations" if d_n else
           "no run has all three replays yet")
        + (f"; {len(flips)} of {len(compared)} single-replay paired ASR conclusions "
           f"(CI side of 0) differ from the all-replay one." if compared else "."))
    if ctx.figures:
        series_r = [{"name": f"replay {r}", "stats": {c: by_rep[c][r] for c in configs},
                     "colour": PALETTE[i], "marker": MARKERS[i]} for i, r in enumerate(REPEATS)]
        series_h = [{"name": f"history sample {h}", "stats": {c: by_smp[c][h] for c in configs},
                     "colour": PALETTE[i % 4], "marker": MARKERS[i % 4]}
                    for i, h in enumerate(samples[:4])]
        p = plot_series_rows(ctx.out / "main_repeats.png", configs,
                             [("ASR by replay", series_r, "ASR (%)"),
                              ("ASR by history sample", series_h, "ASR (%)")],
                             "each row on its configuration's own scenarios (with every replay / "
                             "history sample): compare within a row, not across rows\none replay "
                             "/ one history sample only; scenario-bootstrap 95% intervals",
                             xlim=(-3, 103), sqrt_asr=True)
        sec.figure(p, "ASR by replay and history sample")
        p = plot_variance(ctx.out / "main_variance.png", configs, {c: var[c]["asr"] for c in configs})
        sec.figure(p, "variance decomposition")
    return sec


def plot_variance(path: Path, configs: list[str], var: dict[str, dict | None]) -> Path:
    """Grouped horizontal bars: share of ASR variance per level, with 95% intervals."""
    plt = _plt()
    levels = (("scenario", "between scenarios"), ("history", "between history samples"),
              ("replay", "between replays"))
    fig, ax = plt.subplots(figsize=(WIDTH, 0.32 * len(configs) + 0.8))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    h = 0.26
    drawn = False
    for i, c in enumerate(configs):
        y0 = len(configs) - 1 - i
        v = var.get(c)
        if not v:
            continue
        for j, (k, _) in enumerate(levels):
            if v[k]["share"] is None:
                continue
            y = y0 + (1 - j) * h
            ax.barh(y, v[k]["share"], height=h * 0.9, color=PALETTE[j], zorder=2)
            if v[k]["lo"] is not None:
                ax.plot([v[k]["lo"], v[k]["hi"]], [y, y], color=INK, lw=0.9, zorder=3)
            drawn = True
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(list(reversed([f"{c} (n={var[c]['scenarios']})" if var.get(c)
                                      else f"{c} (not estimable yet)" for c in configs])),
                       fontsize=7, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of ASR variance (%)", fontsize=7.5, color=INK2)
    if not drawn:
        _empty(ax, "not estimable yet: needs runs with several replays in several scenarios")
    from matplotlib.patches import Patch
    handles = [Patch(color=PALETTE[j]) for j in range(len(levels))]
    note = ("each configuration on every measurement it has (a decomposition within one "
            "configuration, not a comparison between them); nested ANOVA (scenario > history "
            "run > replay); scenario-bootstrap 95% intervals")
    return _save(fig, plt, path, _footer(fig, handles, [n for _, n in levels], ncol=3, note=note))


CLEAN_CONFIGS = (NONE, INC_V, "selective[cf]+verify", HEAL_V, "heal_rebuild[cf]+verify")


def s5_clean(data: Data, ctx: Ctx) -> Section:
    sec = Section("clean", "5. Clean false alarms: incident vs heal")
    units = data.clean
    if not units:
        return sec.skip("no clean-control records")
    configs = ordered_configs(units, CLEAN_CONFIGS)
    metrics = ("quarantined", "removed", "bmp", "br")
    st, bases = matched(units, configs, metrics, ctx)
    every = every_measurement(units, configs, metrics, ctx)
    uneven = any(b.uneven for b in bases.values())
    sec.text("Clean runs; the alarm is simulated on the run's own action (recover.py "
             "--include-clean), so whatever is removed is a false positive. BMP against the "
             "same clean run's 'none' record.")
    sec.text(basis_text(bases) + (" The coverage-dependent numbers (each configuration on "
                                  "every clean run it has) are in main.json under "
                                  "all_measurements." if uneven else ""))
    sec.table(["configuration", "runs measured", *[METRICS[m].label for m in metrics]],
              [[c, len({u["run"] for u in units if u["config"] == c}),
                *[cell(st[c][m], METRICS[m].digits) for m in metrics]] for c in configs])
    present = set(configs)
    pairs = [(a, b) for a, b in ((HEAL_V, INC_V), ("heal_rebuild[cf]+verify", INC_V),
                                 (HEAL_V, "selective[cf]+verify")) if a in present and b in present]
    diffs = {f"{a} - {b}": {m: paired(units, m, a, b, ctx) for m in metrics} for a, b in pairs}
    if diffs:
        sec.table(["A - B", *[METRICS[m].name for m in metrics]],
                  [[n, *[dcell(d[m], METRICS[m].digits) for m in metrics]] for n, d in diffs.items()])
    sec.numbers = {"configs": st, "basis": {m: b.as_dict() for m, b in bases.items()},
                   "uneven": uneven, "all_measurements": every, "paired": diffs}
    if diffs:
        n, d = next(iter(diffs.items()))
        sec.takeaway = (f"On clean runs, {n}: " + "; ".join(dshort(d[m], m) for m in ("removed", "bmp", "br"))
                        + ". Less damage is shown only where an interval excludes 0: "
                        + (", ".join(f"{METRICS[m].name} {verdict(d[m], m)}" for m in metrics
                                     if direction(d[m])) or "none") + ".")
    else:
        sec.takeaway = (f"No clean run has both {HEAL_V} and {INC_V} yet; "
                        + "; ".join(f"{c}: removed {short(st[c]['removed'], 'removed')}"
                                    for c in configs if c != NONE) + ".")
    if ctx.figures:
        panels = [(METRICS[m].name, [{"stats": {c: st[c][m] for c in configs},
                                      "colour": lambda c: PALETTE[group_of(c)], "marker": marker_of}],
                   METRICS[m].label) for m in ("removed", "bmp", "br")]
        p = plot_series_rows(ctx.out / "main_clean.png", configs, panels,
                             basis_note([bases[m] for m in ("removed", "bmp", "br")])
                             + "\nclean runs, simulated false alarm; scenario-bootstrap 95% "
                             "intervals", legend=group_handles(configs),
                             height=0.32 * len(configs) + 1.0)
        sec.figure(p, "clean false alarms")
    return sec


def s6_reinfect(data: Data, ctx: Ctx) -> Section:
    sec = Section("reinfection", "6. Reinfection: heal+immunize vs selective+immunize")
    units = data.reinfect
    if not units:
        return sec.skip("no reinfection records")
    later = {u["config"] for u in units if u["checkpoint"] > 0}
    if not later:
        return sec.skip("no record beyond checkpoint 0 yet")
    configs = sorted(later, key=lambda c: (not c.startswith("heal"), c))
    followed = {c: {u["run"] for u in units if u["config"] == c and u["checkpoint"] > 0}
                for c in configs}
    cps = sorted({u["checkpoint"] for u in units})
    metrics = ("asr", "carrying")
    # Matched: every point on the runs every configuration followed, and at each
    # checkpoint on those all of them have there, so the curves compare the same runs.
    every_followed = set.intersection(*followed.values())
    st: dict = {c: {m: {} for m in metrics} for c in configs}
    bases: dict = {m: {} for m in metrics}
    for cp in cps:
        got, bs = matched(units, configs, metrics, ctx,
                          where=lambda u, cp=cp: u["checkpoint"] == cp and u["run"] in every_followed)
        for c in configs:
            for m in metrics:
                st[c][m][cp] = got[c][m]
        for m in metrics:
            bases[m][cp] = bs[m]
    every = {c: {m: {cp: summary(units, m, ctx, c, where=lambda u, cp=cp, c=c:
                                 u["checkpoint"] == cp and u["run"] in followed[c])
                     for cp in cps} for m in metrics} for c in configs}
    uneven = (any(len(v) != len(every_followed) for v in followed.values())
              or any(b.uneven for per in bases.values() for b in per.values()))
    sec.text("Runs followed into the Section 8.8 schedule (checkpoint 0 on patched pages, "
             "later checkpoints on live pages). 'store names the attacker' = share of runs with "
             "an active memory naming the attacker (residual attack at 0, recover.py's reinfection "
             f"note later). Matched: every point stands on the {len(every_followed)} runs every "
             "configuration followed (at each checkpoint, those all of them have there; last "
             "row)." + (" The coverage-dependent numbers (each configuration on its own followed "
                        "runs) are in main.json under all_measurements." if uneven else ""))
    for m in metrics:
        sec.text(f"{METRICS[m].label} per checkpoint:")
        sec.table(["configuration", "runs followed", *[f"cp {cp}" for cp in cps]],
                  [[c, len(followed[c]), *[cell(st[c][m][cp]) for cp in cps]] for c in configs]
                  + [["matched (run, replay) measurements / scenarios", len(every_followed),
                      *[f"{len(bases[m][cp].keys)} / {bases[m][cp].scenarios}" for cp in cps]]])
    heal = next((c for c in configs if c.startswith("heal")), None)
    sel = next((c for c in configs if c.startswith("selective")), None)
    diffs = {}
    if heal and sel:
        common = followed[heal] & followed[sel]
        diffs = {m: {cp: paired(units, m, heal, sel, ctx, where=lambda u, cp=cp:
                                u["checkpoint"] == cp and u["run"] in common) for cp in cps}
                 for m in ("asr", "carrying")}
        sec.text(f"Paired {heal} - {sel} on the {len(common)} runs both followed (pp):")
        sec.table(["metric", *[f"cp {cp}" for cp in cps]],
                  [[METRICS[m].name, *[dcell(diffs[m][cp]) for cp in cps]] for m in diffs])
    sec.numbers = {"configs": st, "followed": {c: len(v) for c, v in followed.items()},
                   "followed_by_every_configuration": len(every_followed),
                   "basis": {m: {cp: b.as_dict() for cp, b in per.items()}
                             for m, per in bases.items()},
                   "uneven": uneven, "all_measurements": every, "paired": diffs}
    last = cps[-1]
    sec.takeaway = (f"At checkpoint {last}: " + "; ".join(
        f"{c} ASR {short(st[c]['asr'][last], 'asr')}, store names attacker "
        f"{short(st[c]['carrying'][last], 'carrying')}" for c in configs)
        + ("" if not diffs else f"; paired {dshort(diffs['asr'][last], 'asr')}") + ".")
    if ctx.figures:
        how = (f"matched: every configuration on the same {len(every_followed)} followed runs"
               if uneven else f"every configuration followed the same {len(every_followed)} runs")
        p = plot_reinfect(ctx.out / "main_reinfection.png", configs, cps, st,
                          note=f"{how}; scenario-bootstrap 95% intervals")
        sec.figure(p, "reinfection")
    return sec


def plot_reinfect(path: Path, configs: list[str], cps: list[int], st: dict,
                  note: str = "scenario-bootstrap 95% intervals") -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    fig.patch.set_facecolor(SURFACE)
    any_small = False
    xs = list(range(len(cps)))
    for ax, m, title in zip(axes, ("asr", "carrying"),
                            ("ASR (live pages after cp 0)", "store names the attacker")):
        _style(ax)
        for i, c in enumerate(configs[:4]):
            off = (i - (min(len(configs), 4) - 1) / 2) * 0.12
            pts = [(x + off, st[c][m][cp]) for x, cp in zip(xs, cps) if st[c][m][cp]]
            if not pts:
                continue
            ax.plot([x for x, _ in pts], [p["mean"] for _, p in pts], color=PALETTE[i], lw=1.6,
                    label=c, zorder=2)
            for x, p in pts:
                ax.plot([x, x], [p["lo"], p["hi"]], color=PALETTE[i], lw=1.1, alpha=0.7)
                _mark(ax, x, p["mean"], PALETTE[i], MARKERS[i], p["small_sample"])
                any_small |= p["small_sample"]
        ax.set_xticks(xs)
        ax.set_xticklabels([str(cp) for cp in cps])
        ax.set_xlabel("checkpoint (sessions after recovery)", fontsize=7.5, color=INK2)
        ax.set_ylabel(f"{METRICS[m].label}", fontsize=7.5, color=INK2)
        ax.set_ylim(-3, 103)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    handles, names = axes[0].get_legend_handles_labels()
    if any_small:
        handles.append(_hollow_handle())
        names.append(f"hollow: < {SMALL} scenarios")
    return _save(fig, plt, path, _footer(fig, handles, names, ncol=3, note=note))


def _quantiles(vals: list[float]) -> str:
    if not vals:
        return "-"
    q = np.percentile(vals, [50, 90])
    return f"{q[0]:.0f} / {q[1]:.0f} / {max(vals):.0f}"


def s7_cost(data: Data, ctx: Ctx) -> Section:
    sec = Section("cost", "7. Cost: LLM calls and seconds")
    units = data.main
    if not units:
        return sec.skip("no main-comparison records")
    configs = ordered_configs(units)
    metrics = ("llm_calls", "seconds")
    st, bases = matched(units, configs, metrics, ctx)
    every = every_measurement(units, configs, metrics, ctx)
    uneven = any(b.uneven for b in bases.values())

    def on(c: str, m: str, basis: Basis) -> list[float]:
        return [u[m] for u in units if u["config"] == c and u.get(m) is not None
                and mkey(u) in basis.keys]

    rows = [[c, *[cell(st[c][m], METRICS[m].digits) for m in metrics],
             *[_quantiles(on(c, m, bases[m])) for m in metrics]] for c in configs]
    sec.text("Per recovery (one record), mean over scenarios with scenario-bootstrap 95% "
             "interval; median / p90 / max over the same records. Seconds are wall time under "
             "64 concurrent workers.")
    sec.text(basis_text(bases) + (" The coverage-dependent numbers (each configuration on "
                                  "every measurement it has) are in main.json under "
                                  "all_measurements." if uneven else ""))
    sec.table(["configuration", *[METRICS[m].label for m in metrics],
               "LLM calls median / p90 / max", "seconds median / p90 / max"], rows)
    heals = [c for c in configs if c.split("[")[0].startswith("heal")]
    hm = ("claim_cf_replays", "ddmin_replays")
    hst, hbases = matched(units, heals, hm, ctx)
    hrows = []
    for c in heals:
        vals = {m: on(c, m, hbases[m]) for m in hm}
        hrows.append([c, len(vals[hm[0]]), *[cell(hst[c][m]) for m in hm],
                      *[f"{sum(v > 0 for v in vals[m])}/{len(vals[m])}" for m in hm],
                      *[_quantiles(vals[m]) for m in hm]])
    if heals:
        sec.text("Heal's claim-level replays per recovery (claim_cf_replays counts every replay "
                 "healing ran; ddmin_replays the part delta debugging spent). "
                 + basis_text(hbases, "heal configuration"))
        sec.table(["configuration", "records", "claim-level replays", "ddmin replays",
                   "records with claim-level replays", "records with ddmin replays",
                   "claim-level median / p90 / max", "ddmin median / p90 / max"], hrows)
    sec.numbers = {"configs": st, "basis": {m: b.as_dict() for m, b in bases.items()},
                   "uneven": uneven, "all_measurements": every, "heal_replays": hst,
                   "heal_replays_basis": {m: b.as_dict() for m, b in hbases.items()}}
    ok = {c: st[c]["llm_calls"] for c in configs if st[c]["llm_calls"]}
    if ok:
        hi = max(ok, key=lambda c: ok[c]["mean"])
        parts = [f"most LLM calls per recovery: {hi} {short(ok[hi], 'llm_calls')}"]
        for a, b in ((HEAL, INC), (HEAL_V, INC_V)):
            d = paired(units, "llm_calls", a, b, ctx)
            if d:
                parts.append(f"{a} - {b} {dshort(d, 'llm_calls')}")
        sec.takeaway = "; ".join(parts) + "."
    if ctx.figures:
        panels = [(METRICS[m].name, [{"stats": {c: st[c][m] for c in configs},
                                      "colour": lambda c: PALETTE[group_of(c)], "marker": marker_of}],
                   METRICS[m].label) for m in metrics]
        p = plot_series_rows(ctx.out / "main_cost.png", configs, panels,
                             basis_note(bases.values()) + "\nper recovery; seconds are wall time "
                             "under 64 concurrent workers; scenario-bootstrap 95% intervals",
                             legend=group_handles(configs))
        sec.figure(p, "cost per configuration")
        replaying = [c for c in heals if any(on(c, "claim_cf_replays", hbases["claim_cf_replays"]))]
        if replaying:
            p = plot_heal_replays(ctx.out / "main_heal_replays.png", units, replaying, hbases)
            sec.figure(p, "heal claim-level replays")
    return sec


def plot_heal_replays(path: Path, units: list[dict], heals: list[str],
                      bases: dict[str, Basis] | None = None) -> Path:
    """Empirical distribution (ECDF over records) of heal's replays per recovery, each
    metric on its matched basis among `heals` when `bases` is given."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes, ("claim_cf_replays", "ddmin_replays")):
        _style(ax)
        keep = on_basis(bases[m]) if bases and m in bases else (lambda u: True)
        for i, c in enumerate(heals[:4]):
            vals = np.sort([u[m] for u in units if u["config"] == c and u.get(m) is not None
                            and keep(u)])
            if not vals.size:
                continue
            y = np.arange(1, vals.size + 1) / vals.size * 100
            ax.step(np.concatenate([[0], vals]), np.concatenate([[0], y]), where="post",
                    color=PALETTE[i], lw=1.6, label=f"{c} ({vals.size} records)")
        ax.set_xlabel(f"{METRICS[m].name} per recovery", fontsize=7.5, color=INK2)
        ax.set_ylabel("records at or below (%)", fontsize=7.5, color=INK2)
        ax.set_ylim(-2, 102)
        ax.set_title(METRICS[m].name, fontsize=8, color=INK, loc="left")
    handles, names = axes[0].get_legend_handles_labels()
    note = ((basis_note(bases.values(), "heal configuration") + "\n") if bases else "") + (
        "empirical distribution over records (not scenarios); configurations with no "
        "claim-level replay are left out")
    return _save(fig, plt, path, _footer(fig, handles, names, ncol=2, note=note))


SUITE_CONFIGS = (NONE, INC, HEAL)


def s8_suites(data: Data, ctx: Ctx) -> Section:
    sec = Section("suites", "8. Cross-suite: heal vs incident")
    suites = [s for s in SUITE_ORDER if s in data.suites] + sorted(set(data.suites) - set(SUITE_ORDER))
    if not suites:
        return sec.skip("no suite records")
    metrics = ("asr", "bmp", "bcp")
    # Matched within each suite: the three configurations on the measurements all of
    # them have there (a suite's heal records may so far cover fewer runs than incident's).
    st, bases, every = {}, {}, {}
    for s in suites:
        st[s], bases[s] = matched(data.suites[s], SUITE_CONFIGS, metrics, ctx)
        every[s] = every_measurement(data.suites[s], SUITE_CONFIGS, metrics, ctx)
    uneven = any(b.uneven for bs in bases.values() for b in bs.values())
    diffs = {s: {f"{a} - {b}": {m: paired(data.suites[s], m, a, b, ctx) for m in metrics}
                 for a, b in ((HEAL, INC), (HEAL_V, INC_V))} for s in suites}
    sec.text("Per suite, mean over scenarios (dense evolving: over its runs and 3 replays; "
             "other suites: 1 replay per history run). The cross-model subset is split by the "
             "model each run.db records.")
    sec.text(f"Matched within each suite: {', '.join(SUITE_CONFIGS)} on the (run, replay) "
             "measurements all of them have there (per metric; a configuration the suite lacks "
             "is left out), so a row's columns compare the same runs; the differences are "
             "paired, each on what both of its configurations have. "
             + ("The coverage-dependent numbers (each configuration on everything it has) are in "
                "main.json under all_measurements." if uneven else "Every configuration a suite "
                "has covers the same measurements there."))

    def runs(s: str, c: str) -> int:
        return len({u["run"] for u in data.suites[s] if u["config"] == c})

    sec.table(["suite", *[f"{c} runs" for c in SUITE_CONFIGS],
               "matched ASR measurements / scenarios"],
              [[s, *[runs(s, c) for c in SUITE_CONFIGS],
                f"{len(bases[s]['asr'].keys)} / {bases[s]['asr'].scenarios}"] for s in suites])
    for m in metrics:
        sec.text(f"{METRICS[m].label}:")
        sec.table(["suite", *SUITE_CONFIGS, f"{HEAL} - {INC}", f"{HEAL_V} - {INC_V}"],
                  [[s, *[cell(st[s][c][m]) for c in SUITE_CONFIGS],
                    dcell(diffs[s][f"{HEAL} - {INC}"][m]), dcell(diffs[s][f"{HEAL_V} - {INC_V}"][m])]
                   for s in suites])
    sec.numbers = {"configs": st,
                   "basis": {s: {m: b.as_dict() for m, b in bs.items()} for s, bs in bases.items()},
                   "uneven": uneven, "all_measurements": every, "paired": diffs}
    measured = [s for s in suites if diffs[s][f"{HEAL} - {INC}"]["asr"]]
    lower = [s for s in measured if direction(diffs[s][f"{HEAL} - {INC}"]["asr"]) == -1]
    higher = [s for s in measured if direction(diffs[s][f"{HEAL} - {INC}"]["asr"]) == 1]
    sec.takeaway = (f"{HEAL} - {INC} ASR measured on {len(measured)}/{len(suites)} suites; "
                    f"interval below 0 on {len(lower)} ({', '.join(lower) or 'none'}), above 0 on "
                    f"{len(higher)} ({', '.join(higher) or 'none'}), includes 0 on the rest.")
    if ctx.figures:
        panels = []
        for m in ("asr", "bcp"):
            series = [{"name": c, "stats": {s: st[s][c][m] for s in suites}, "colour": PALETTE[i],
                       "marker": MARKERS[i]} for i, c in enumerate(SUITE_CONFIGS)]
            panels.append((METRICS[m].name, series, METRICS[m].label))
        how = (f"matched within each suite: {', '.join(SUITE_CONFIGS)} on the same (run, replay) "
               "measurements (counts in main_tables.md)" if uneven else
               "every configuration has the same measurements within each suite")
        p = plot_series_rows(ctx.out / "main_suites.png", suites, panels,
                             f"{how}\nper suite; scenario-bootstrap 95% intervals", xlim=(-3, 103),
                             sqrt_asr=True)
        sec.figure(p, "cross-suite")
    return sec


SECTIONS = (s1_tradeoff, s2_heal_vs_quarantine, s3_breakdown, s4_repetition, s5_clean,
            s6_reinfect, s7_cost, s8_suites)


# --- report ------------------------------------------------------------------------------

def inputs_md(data: Data) -> list[str]:
    rows = []
    for name, info in data.inputs.items():
        rows.append([name, info.get("path", "-"), info.get("state", "-"), info.get("records", 0),
                     info.get("part_files", 0), info.get("duplicates", 0), info.get("torn_lines", 0),
                     info.get("selected", 0), info.get("runs", 0), info.get("scenarios", 0),
                     ", ".join(f"{k}: {v}" for k, v in (info.get("dropped") or {}).items()) or "-"])
    out = ["## Inputs", "",
           measure.md_table(["input", "path", "state", "records", "part files", "duplicates",
                             "torn lines", "selected", "runs", "scenarios", "left out"], rows), ""]
    pres = data.preservation
    files = "; ".join(f"{f['path']} ({f['state']}, {f['rows']} rows)" for f in pres.get("files", []))
    out += [f"Benign claim preservation: {files or 'no file given'}.", ""]
    st = pres.get("status") or {}
    if st:
        out += [measure.md_table(["records file / suite", "BCP join status"],
                                 [[k, ", ".join(f"{s}: {n}" for s, n in v.items())]
                                  for k, v in st.items()]), ""]
    return out


def write_report(sections: list[Section], data: Data, ctx: Ctx, params: dict) -> dict:
    ctx.out.mkdir(parents=True, exist_ok=True)
    md = ["# Main comparison: final tables", "",
          f"Generated {params['generated']} by analyze_main.py from the files below; the batch "
          f"may still be running, so every cell gives its n (scenarios) and cells with fewer "
          f"than {SMALL} scenarios are flagged 'small sample'. Intervals: percentile bootstrap "
          f"over scenarios, {ctx.n_boot} resamples, seed {ctx.seed}.", "",
          "## Takeaways", ""]
    for s in sections:
        md.append(f"- **{s.title}**: " + (s.takeaway or (f"skipped ({s.skipped})" if s.skipped
                                                         else "no takeaway")))
    md += ["", *inputs_md(data)]
    for s in sections:
        md += [f"## {s.title}", ""]
        if s.skipped:
            md += [f"Skipped: {s.skipped}.", ""]
            continue
        if s.takeaway:
            md += [f"**Takeaway:** {s.takeaway}", ""]
        md += s.md
    (ctx.out / "main_tables.md").write_text("\n".join(md).rstrip() + "\n")
    result = {"params": params, "inputs": data.inputs, "preservation": data.preservation,
              "history_sample_anomalies": data.samples_note,
              "sections": {s.key: {"title": s.title, "takeaway": s.takeaway, "skipped": s.skipped,
                                   "figures": s.figures, "numbers": s.numbers} for s in sections}}
    result = _round(result)
    (ctx.out / "main.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, default in INPUTS.items():
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, default=ROOT / default)
    p.add_argument("--preservation", type=Path, nargs="+", default=[ROOT / PRESERVATION])
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verify-n", type=int, default=VERIFY_N)
    p.add_argument("--no-parts", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    t0 = time.time()
    ctx = Ctx(out=args.out, n_boot=args.n_boot, seed=args.seed, figures=not args.no_figures,
              verify_n=args.verify_n)
    ctx.out.mkdir(parents=True, exist_ok=True)
    paths = {name: getattr(args, name) for name in INPUTS}
    data = load(paths, args.preservation, args.runs, not args.no_parts, ctx)
    sections = []
    for fn in SECTIONS:
        try:
            sections.append(fn(data, ctx))
        except Exception as exc:              # noqa: BLE001 - one section must not cost the rest
            name = fn.__name__
            sec = Section(name, name).skip(f"failed: {type(exc).__name__}: {exc}")
            sec.md.append(traceback.format_exc())
            sections.append(sec)
    params = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": ctx.n_boot,
              "seed": ctx.seed, "verify_n": ctx.verify_n, "small_sample_below": SMALL,
              "parts": not args.no_parts, "sample1_runs": list(SAMPLE1)}
    result = write_report(sections, data, ctx, params)
    figs = sum(len(s.figures) for s in sections)
    print(f"wrote {ctx.out / 'main_tables.md'}, main.json and {figs} figures "
          f"in {time.time() - t0:.1f}s")
    for s in sections:
        print(f"  {s.title}: " + (f"skipped ({s.skipped})" if s.skipped else s.takeaway or ""))
    return result


if __name__ == "__main__":
    main()
