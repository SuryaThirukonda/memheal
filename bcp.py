"""Benign claim preservation (BCP) per recovery configuration (Phase 2, H.5).

memheal/preservation.py defines the metric; this script finds, for every recovery
record, the store it was recovered from and the copy recovery left, measures each copy,
and pools the results per configuration. Everything is read from the stores already on
disk: no model is called (the only model is the local MiniLM embedder, run offline), no
run folder is written to, and old records are measured exactly like new ones.

    uv run python bcp.py
    uv run python bcp.py --records results/recovery/records.jsonl results/claims/records.jsonl
    uv run python bcp.py --records results/ablations/k.jsonl --threshold 0.9 --workers 8
    uv run python bcp.py --records results/smoke/families.jsonl=smoke_families

Which copy a record measured. A record written after the heal work names its copy and
pre-target snapshot (`work_db`, `pre_db`, relative to the repo root). An older record
does not, so the path is rebuilt the way recover.py named it: the configuration's
work_tag (recover.config_tag plus the knob and repeat suffixes, rebuilt from the
record's fields, heal v2's @v<n>, @c<n> and @b<n> from verify_n, claim_n and
ddmin_budget included) inside the work subdirectory its sweep wrote to. That folder is not in
the record either, and it is decided once per records file, never per record and never
by trying one folder after another: a sweep that re-runs a configuration another
experiment also ran leaves a copy of the same name in each experiment's folder, usually
with the same counts, so a fallback would quietly measure the other experiment's trial.
In order: FILE=SUBDIR on --records (or --work-subdir, for every file); WORK_SUBDIRS,
what the sweeps behind the existing files used where the naming rule is wrong; else the
naming rule (results/<experiment>/records.jsonl -> <experiment>,
results/<group>/<name>.jsonl -> <name>), when that is a folder recover.py wrote in one
of the file's runs (it logs *.llm.jsonl there; a Mem0 run's own mem0/ folder holds
none). Failing all three, the file's records are reported 'no_work_subdir'.

Every copy, named or rebuilt, is then checked against its record: recovery_counts reads
n_quarantined, n_restored and n_reconstructed back from the stores, and a copy that
disagrees was written by another recovery (a later sweep over the same file name), so
it is reported 'copy_mismatch' and left out of every rate. On the existing sweeps every
record's own copy agrees.

A reinfection run (its configuration has later checkpoints in the same file) lives on
in its '@reinfect' copy, or in older sweeps in the plain one, and is measured as
recovery left it (preservation.after_recovery); any other record reads only the plain
copy. Only checkpoint-0 records are measured: the later checkpoints describe
reinfection, not recovery, and no copy holds their state.

The cosine threshold decides whether a reworded claim counts as kept, which favours
repairs that keep a claim's words over repairs that rebuild it, so every rate and paired
difference is also computed at each of --thresholds (--threshold is the one the main
tables and the figure use). preservation_pairs.jsonl lists the near misses: each
(pre-recovery claim, best same-label claim after recovery) pair a threshold of the
sweep decides, with its cosine and a blank `same` for a reader to fill in;
--calibration reads such a file back and reports each threshold's precision and recall
against those labels.

Outputs under --out (default results/preservation/): preservation.jsonl (one row per
record: counts, rates, paths, examples of lost benign and surviving attack claims, kept
counts per threshold), preservation_tables.md, preservation.json,
preservation_pairs.jsonl, and preservation.png (BCP vs ACR per configuration). Rates pool
claims over runs, with Wilson 95% intervals and, since claims of one run are not
independent, percentile-bootstrap 95% intervals over runs; paired differences against
--baseline (default selective[oracle]) resample runs too.
"""

from __future__ import annotations

import os

# Set before anything imports huggingface_hub, which reads them once at import: the
# embedder is the local MiniLM copy, and this script must never reach the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import threading                                                     # noqa: E402
import traceback                                                     # noqa: E402
from collections import Counter, defaultdict                         # noqa: E402
from concurrent.futures import ThreadPoolExecutor                    # noqa: E402
from collections.abc import Iterable                                 # noqa: E402
from dataclasses import dataclass, replace                           # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

import measure                                                       # noqa: E402
import recover                                                       # noqa: E402
from memheal import db, preservation                                 # noqa: E402
from memheal.scenarios import ALL_VARIANTS                           # noqa: E402

ROOT = Path(__file__).parent
DEFAULT_RECORDS = ("results/recovery/records.jsonl",)
DEFAULT_OUT = ROOT / "results" / "preservation"
RUNS = ROOT / "data" / "runs"
DEFAULT_BASELINE = "selective[oracle]"
SMALL = 10                  # runs below which a cell is labelled 'small sample' (analyze.py)
N_BOOT = 2000
DEFAULT_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
BAND_MARGIN = 0.05          # near misses this far outside the sweep are exported too
PAIRS_FILE = "preservation_pairs.jsonl"

# The --work-subdir of the sweeps behind the existing records files wherever the naming
# rule (subdir_by_name) gets it wrong, read off their recover lines in data/*_jobs.txt
# and the smoke pass: the history suites recovered in the default folder, two files
# share another file's folder, and the smoke pass wrote to 'smoke_<x>' (its 'families',
# 'k', ... folders belong to the full sweeps). Keyed by '<parent>/<file>', so a copy of
# results/ elsewhere resolves alike. Records from now on name their copy (work_db).
WORK_SUBDIRS = {
    **{f"{suite}/records.jsonl": recover.DEFAULT_SUBDIR
       for suite in ("adaptive", "external", "models", "mem0")},
    "claims/reinfect.jsonl": "claims",
    "ablations/groups_clean.jsonl": "groups",
    **{f"smoke/{x}.jsonl": f"smoke_{x}"
       for x in ("families", "k", "depth", "reretrieve", "groups", "claims", "flat")},
    # Phase 2 (data/main_b_jobs.txt, data/smoke_heal2_jobs.txt). Their records name their
    # copy, so these only matter for a record that lost its work_db; results/main/
    # records.jsonl and results/<suite>/main.jsonl follow the naming rule ('main').
    "main/clean_control.jsonl": "main",
    "main/reinfect.jsonl": "reinfect_heal",
    "smoke/heal2.jsonl": "smoke_heal2",
}

# Record fields copied into every output row, so the rows group (measure.setting_key)
# without the records file.
IDENTITY = ("run_dir", "variant", "domain", "category", "condition", "mode", "verdicts",
            "modifiers", "source", "checkpoint", "n_quarantined", "n_restored",
            "n_reconstructed")
# name -> (kept field, total field, counts removal). preservation.Preservation.as_dict's
# counts; ACR and the ambiguous rate count what is no longer kept.
METRICS = {
    "bcp": ("benign_kept", "benign", False),
    "bcp_exact": ("benign_kept_exact", "benign", False),
    "bcp_lineage": ("benign_lineage_kept", "benign_lineage", False),
    "acr": ("attack_kept", "attack", True),
    "acr_exact": ("attack_kept_exact", "attack", True),
    "ambiguous_removed": ("ambiguous_kept", "ambiguous", True),
}
PAIRED_METRICS = ("bcp", "bcp_exact", "bcp_lineage", "acr", "acr_exact")
# Metrics a threshold changes (the exact-only ones it does not), per group and paired.
SWEEP_METRICS = ("bcp", "bcp_lineage", "acr")
PAIRED_SWEEP = ("bcp", "acr")

# Figure: analyze.py's palette and ink; one colour per recovery strategy.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "D", "^")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
WIDTH = 6.5
STRATEGIES = (
    ("quarantine", "quarantine (root, ancestry, selective, incident, reset)",
     ("root", "ancestry", "selective", "incident", "reset")),
    ("repair", "repair (selective_recon, unmerge, selective_claims, heal)",
     ("selective_recon", "unmerge", "selective_claims", "heal", "heal_oracle")),
    ("none", "no recovery", ("none",)),
    ("other", "other (rollback)", ()),
)


# --- inputs ---------------------------------------------------------------------------

def read_records(path: Path, parts: bool = True) -> list[dict] | None:
    """`path` plus the part files a running batch has not merged yet, one record per
    measurement (as analyze.read_records); None when neither exists."""
    path = Path(path)
    files = [path] if path.is_file() else []
    part_files = sorted((path.parent / ".parts").glob(f"{path.stem}.*.jsonl")) if parts else []
    if not files and not part_files:
        return None
    rows = [r for f in files + part_files for r in measure.load_records(f)]
    last = {measure.record_key(r): i for i, r in enumerate(rows)}
    return [r for i, r in enumerate(rows) if last[measure.record_key(r)] == i]


def experiment_name(path: Path) -> str:
    """'recovery' for results/recovery/records.jsonl, 'ablations/k' for results/ablations/k.jsonl."""
    path = Path(path)
    return path.parent.name if path.stem == "records" else f"{path.parent.name}/{path.stem}"


def subdir_by_name(path: Path) -> str:
    """The naming rule: results/<experiment>/records.jsonl -> '<experiment>',
    results/<group>/<name>.jsonl -> '<name>' (results/ablations/k.jsonl -> 'k')."""
    path = Path(path)
    return path.parent.name if path.stem == "records" else path.stem


def is_work_folder(folder: Path) -> bool:
    """Did recover.py write this folder? It logs one <tag>.llm.jsonl per configuration
    beside the copy; a Mem0 run's own mem0/ folder (history and vectors) holds none."""
    return Path(folder).is_dir() and next(Path(folder).glob("*.llm.jsonl"), None) is not None


def work_subdir(path: Path, run_dirs: Iterable[Path], given: str | None = None) -> tuple[str | None, str]:
    """(the one work subdirectory of a records file's rebuilt copies, where it came from:
    'given', 'known', 'name' or 'none'). See the module docstring for why there is no
    fallback past it."""
    if given:
        return given, "given"
    path = Path(path)
    known = WORK_SUBDIRS.get(f"{path.parent.name}/{path.name}")
    if known:
        return known, "known"
    name = subdir_by_name(path)
    if recover.SUBDIR_RE.fullmatch(name) and any(is_work_folder(Path(r) / name) for r in run_dirs):
        return name, "name"
    return None, "none"


def records_spec(value: str) -> tuple[str, str | None]:
    """'FILE' or 'FILE=SUBDIR' (--records)."""
    path, sep, subdir = value.partition("=")
    return path, (subdir if sep else None)


def config_name(rec: dict) -> str:
    """measure's configuration label, the replay pages only when not 'patched', then knobs."""
    label = measure.config_label(measure.config_key(rec), False)
    if (rec.get("source") or "patched") != "patched":
        label += f" ({rec['source']})"
    return label + measure.knob_label(measure.knobs(rec))


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _repo_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


# --- which stores a record measured ----------------------------------------------------

def _stated(rec: dict, name: str, default):
    """The record's value for a knob, or recover.py's default when it states none (None
    where the knob does not apply, or a record from before the field). Not `or`: a
    --ddmin-budget of 0 is a setting of its own."""
    value = rec.get(name)
    return default if value is None else value


def record_options(rec: dict, subdir: str) -> list[recover.Options]:
    """The recover.Options that would have named this record's copy, likeliest first.

    k is the one knob a record cannot always give back: it records the k every replay
    used, which is the run's own unless --k was passed. Without --depth an observe
    'fresh' record had --k; with --depth either is possible, so both are tried. The heal
    v2 knobs (@v<n>, @c<n>, @b<n>) come back from verify_n, claim_n and ddmin_budget;
    recover.config_tag itself decides where each applies (+verify, the claim-level heal
    modes), so a value a record carries where it does not apply adds no suffix.
    """
    verdicts = rec.get("verdicts") if rec.get("verdicts") in ("oracle", "cf") else "oracle"
    defaults = recover.Options()
    base = replace(
        defaults, verdicts=verdicts, modifiers=tuple(rec.get("modifiers") or ()),
        source=rec.get("source") or "patched", work_subdir=subdir,
        family=rec.get("family") or "recorded", depth=rec.get("depth") or 0,
        repeat=rec.get("repeat"), cf_workers=rec.get("cf_workers") or 1,
        ablation=rec.get("cf_method") or "frozen", cf_groups=bool(rec.get("cf_groups")),
        stop_after_action=bool(measure.knob_value(rec, "stop_after_action")),
        verify_n=_stated(rec, "verify_n", defaults.verify_n),
        claim_n=_stated(rec, "claim_n", defaults.claim_n),
        ddmin_budget=_stated(rec, "ddmin_budget", defaults.ddmin_budget))
    ks: list[int | None] = [None]
    if rec.get("observe") == "fresh" and rec.get("k") is not None:
        ks = [None, rec["k"]] if base.depth else [rec["k"]]
    return [replace(base, k=k) for k in ks]


def resolve_copy(rec: dict, run_dir: Path, subdir: str | None,
                 reinfected: bool = False) -> tuple[Path | None, str]:
    """(the recovered copy, how it was found: 'record', 'rebuilt' or 'no_subdir'). The
    record's own work_db wins; otherwise the copy recover.py would have written in
    `subdir`: for a reinfection run its '@reinfect' copy, then the plain one (older
    sweeps reused one file); for any other record only the plain one, since an
    '@reinfect' copy is another run's."""
    if rec.get("work_db"):
        p = _repo_path(rec["work_db"])
        return (p if p.is_file() else None), "record"
    if subdir is None:
        return None, "no_subdir"
    for opts in record_options(rec, subdir):
        for reinfect in ((True, False) if reinfected else (False,)):
            tag = recover.work_tag(rec["mode"], replace(opts, reinfect=reinfect))
            p = run_dir / subdir / f"{tag}.db"
            if p.is_file():
                return p, "rebuilt"
    return None, "rebuilt"


def disagreement(rec: dict, copy_counts: dict[str, int]) -> list[str]:
    """The counts a record states that its copy does not hold (fields the record lacks
    are not compared)."""
    return [k for k, v in copy_counts.items() if rec.get(k) is not None and rec[k] != v]


def resolve_pre(rec: dict, run_dir: Path, target_session: int) -> Path | None:
    """The pre-target snapshot: the record's pre_db, else recover.py's own choice."""
    if rec.get("pre_db"):
        p = _repo_path(rec["pre_db"])
        return p if p.is_file() else None
    return recover.snapshot_before(run_dir, target_session)


def _read(path: Path) -> preservation.Store:
    conn = db.connect_readonly(path)
    try:
        return preservation.read_store(conn)
    finally:
        conn.close()


# --- measuring ------------------------------------------------------------------------

@dataclass
class Job:
    """One record to measure, with the file it came from, that file's work subdirectory,
    and whether the record's configuration went on into reinfection."""
    rec: dict
    experiment: str
    records: Path
    subdir: str | None
    reinfected: bool = False


@dataclass
class Settings:
    runs_dir: Path
    threshold: float
    scope: str
    thresholds: tuple[float, ...] = ()
    band: tuple[float, float] | None = None


def base_row(job: Job) -> dict:
    """The record's identity and knobs, so a row groups (measure.setting_key) without the
    records file. The heal v2 knobs are copied only from a record that states them: a
    missing one reads as None either way, and rows of older records stay as they were."""
    rec = job.rec
    return {"experiment": job.experiment, "records": _rel(job.records),
            **{k: rec.get(k) for k in IDENTITY},
            **{k: measure.knob_value(rec, k) for k in measure.KNOBS
               if k in rec or k not in measure.HEAL_V2_KNOBS},
            "config": config_name(rec)}


def measure_record(job: Job, cache: preservation.EmbeddingCache, settings: Settings,
                   pres: dict[Path, preservation.Store]) -> dict:
    """One output row: identity, the stores used, status, and the counts when 'ok'.
    `pres` caches pre-target snapshots across the records of one run."""
    row = base_row(job)
    rec = job.rec
    scenario = ALL_VARIANTS.get(rec.get("variant") or "")
    run_dir = settings.runs_dir / rec["run_dir"]
    if scenario is None:
        return {**row, "status": "no_scenario"}
    if not run_dir.is_dir():
        return {**row, "status": "missing_run"}
    copy_path, found = resolve_copy(rec, run_dir, job.subdir, job.reinfected)
    pre_path = resolve_pre(rec, run_dir, scenario.target_session)
    row.update(work_db=_rel(copy_path), pre_db=_rel(pre_path), copy_from=found)
    if copy_path is None:
        return {**row, "status": "no_work_subdir" if found == "no_subdir" else "missing_copy"}
    if pre_path is None:
        return {**row, "status": "missing_pre"}
    if pre_path not in pres:
        pres[pre_path] = _read(pre_path)
    pre, copy = pres[pre_path], _read(copy_path)
    target, deepened = scenario.target_session, bool(rec.get("depth"))
    start, after, _ = preservation.recovery_view(pre, copy, target_session=target,
                                                 deepened=deepened)
    row["copy_counts"] = preservation.recovery_counts(start, after, copy)
    off = disagreement(rec, row["copy_counts"])
    if off:
        return {**row, "status": "copy_mismatch", "disagrees": off}
    spec, is_root = preservation.oracle_inputs(scenario, rec.get("condition") or "poisoned")
    result = preservation.measure_copy(
        pre, copy, spec, is_root, cache, target_session=target, deepened=deepened,
        threshold=settings.threshold, thresholds=settings.thresholds, scope=settings.scope,
        band=settings.band)
    # 'near' feeds preservation_pairs.jsonl and is dropped from the row before writing.
    return {**row, "status": "ok", **result.as_dict(), "near": result.near}


def measure_run(jobs: list[Job], embedder, lock: threading.Lock, settings: Settings) -> list[dict]:
    """Every record of one run, sharing its snapshot and claim vectors. A record that
    fails is reported, not fatal: one unreadable copy must not cost the sweep."""
    cache = preservation.EmbeddingCache(embedder, lock)
    pres: dict[Path, preservation.Store] = {}
    out = []
    for job in jobs:
        try:
            out.append(measure_record(job, cache, settings, pres))
        except Exception as e:                               # noqa: BLE001
            out.append({**base_row(job), "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(limit=3)})
    return out


def _run_config(rec: dict) -> tuple:
    """A run and configuration, every checkpoint of it (measure.record_key without the
    checkpoint)."""
    return (rec.get("run_dir"), measure.config_key(rec), measure.knobs(rec),
            rec.get("condition", "poisoned"))


def collect_jobs(specs: list[tuple[Path, str | None]], parts: bool, given: str | None = None,
                 runs_dir: Path = RUNS) -> tuple[list[Job], list[dict]]:
    """Checkpoint-0 records of every (file, its FILE=SUBDIR or None), and what each file
    held. The work subdirectory is decided here, once per file; `given` is --work-subdir."""
    jobs: list[Job] = []
    inputs = []
    for path, subdir in specs:
        recs = read_records(path, parts)
        at0 = [r for r in recs or () if r.get("checkpoint", 0) == 0]
        runs = sorted({Path(runs_dir) / r["run_dir"] for r in at0 if r.get("run_dir")})
        sub, how = work_subdir(path, runs, subdir or given)
        reinfected = {_run_config(r) for r in recs or () if r.get("checkpoint", 0) > 0}
        exp = experiment_name(path)
        inputs.append({"records": _rel(path), "experiment": exp,
                       "state": "missing" if recs is None else "ok",
                       "read": len(recs or ()), "checkpoint0": len(at0),
                       "later_checkpoints": len(recs or ()) - len(at0),
                       "subdir": sub, "subdir_from": how})
        jobs += [Job(r, exp, path, sub, _run_config(r) in reinfected) for r in at0]
    return jobs, inputs


def measure_all(jobs: list[Job], embedder, settings: Settings, workers: int) -> list[dict]:
    """Rows in job order. Runs are the unit of work: a run's records share a snapshot
    and most of their claims, so its vectors are encoded once."""
    by_run: dict[str, list[int]] = defaultdict(list)
    for i, job in enumerate(jobs):
        by_run[job.rec["run_dir"]].append(i)
    lock = threading.Lock()
    rows: list[dict | None] = [None] * len(jobs)

    def work(idx: list[int]) -> None:
        for i, row in zip(idx, measure_run([jobs[i] for i in idx], embedder, lock, settings)):
            rows[i] = row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, by_run.values()))
    return rows                                                    # type: ignore[return-value]


# --- statistics -----------------------------------------------------------------------

def counts(row: dict, metric: str, threshold: float | None = None) -> tuple[int, int]:
    """(numerator, denominator) of one metric for one row, at the row's own threshold or
    at one of the sweep's (the denominators no threshold changes)."""
    kept, total, removal = METRICS[metric]
    d = row.get(total) or 0
    src = row if threshold is None else row["sweep"][preservation.threshold_key(threshold)]
    k = src.get(kept) or 0
    return (d - k if removal else k), d


def _boot_index(n_runs: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_runs, size=(n_boot, n_runs))


def _percentile(values: np.ndarray) -> tuple[float, float] | None:
    """measure.paired_differences' percentile rule on the finite values."""
    v = np.sort(values[np.isfinite(values)])
    if not len(v):
        return None
    return float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])


def _pooled(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, num[idx].sum(axis=1) / np.where(d > 0, d, 1), np.nan)


def ratio(pairs: list[tuple[int, int]], n_boot: int = N_BOOT, seed: int = 0) -> dict | None:
    """A rate pooled over runs: Wilson 95% (claims as if independent) and a percentile
    bootstrap 95% over runs (claims of one run move together)."""
    n, d = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    if not d:
        return None
    lo, hi = measure.wilson(n, d)
    num, den = np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float)
    boot = _percentile(_pooled(num, den, _boot_index(len(pairs), n_boot, seed)))
    return {"n": n, "d": d, "rate": n / d, "wilson": [lo, hi], "boot": list(boot) if boot else None}


def diff(a: list[tuple[int, int]], b: list[tuple[int, int]], n_boot: int = N_BOOT,
         seed: int = 0) -> list[float] | None:
    """Pooled rate of A minus pooled rate of B over the same runs (a[i] and b[i] are one
    run), with a percentile bootstrap 95% interval from resampling runs."""
    na, da = (np.array(x, float) for x in zip(*a))
    nb, db_ = (np.array(x, float) for x in zip(*b))
    if not da.sum() or not db_.sum():
        return None
    obs = na.sum() / da.sum() - nb.sum() / db_.sum()
    idx = _boot_index(len(a), n_boot, seed)
    ci = _percentile(_pooled(na, da, idx) - _pooled(nb, db_, idx))
    return None if ci is None else [float(obs), *ci]


def _sort_key(key: tuple) -> tuple:
    exp, cond, setting = key
    return (exp, cond != "poisoned", cond, measure._config_sort(setting))


def summarize(rows: list[dict], n_boot: int = N_BOOT, seed: int = 0,
              thresholds: Iterable[float] = ()) -> list[dict]:
    """One entry per experiment x condition x configuration (measure.setting_key), with
    the threshold-dependent rates again at every threshold of the sweep."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r.get("condition") or "poisoned", measure.setting_key(r))].append(r)
    out = []
    for key in sorted(groups, key=_sort_key):
        rs = groups[key]
        ok = [r for r in rs if r["status"] == "ok"]
        entry = {"experiment": key[0], "condition": key[1], "config": rs[0]["config"],
                 "mode": rs[0]["mode"], "source": rs[0].get("source") or "patched",
                 "knobs": [list(p) for p in measure.knobs(rs[0])],
                 "records": len(rs), "runs": len({r["run_dir"] for r in ok}),
                 "status": dict(Counter(r["status"] for r in rs))}
        for m in METRICS:
            entry[m] = ratio([counts(r, m) for r in ok], n_boot, seed)
        entry["sweep"] = {preservation.threshold_key(t): {
            m: ratio([counts(r, m, t) for r in ok], n_boot, seed) for m in SWEEP_METRICS}
            for t in thresholds}
        out.append(entry)
    return out


def paired(rows: list[dict], baseline: str = DEFAULT_BASELINE, n_boot: int = N_BOOT,
           seed: int = 0, thresholds: Iterable[float] = ()) -> list[dict]:
    """Each configuration minus `baseline` on the runs both measured, within one
    experiment, condition, replay-page source and knob setting (a configuration with
    one-sided knobs, the counterfactual's, the claim-level replays' or +verify's, is set
    against the baseline's setting without them, as in measure.paired_differences); BCP
    and ACR again at every threshold of the sweep."""
    thresholds = tuple(thresholds)
    blocks: dict[tuple, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["status"] == "ok":
            key = (r["experiment"], r.get("condition") or "poisoned",
                   r.get("source") or "patched", measure.knobs(r))
            blocks[key][measure.config_label(measure.config_key(r), False)][r["run_dir"]] = r
    out = []
    for (exp, cond, src, kn), by_label in blocks.items():
        base_kn = next((k for k in measure.fallback_settings(kn)
                        if baseline in blocks.get((exp, cond, src, k), {})), None)
        base = blocks.get((exp, cond, src, base_kn), {}).get(baseline)
        if not base:
            continue
        for label, by_run in by_label.items():
            if label == baseline and base_kn == kn:
                continue
            runs = sorted(set(by_run) & set(base))
            if not runs:
                continue
            entry = {"experiment": exp, "condition": cond, "source": src,
                     "a": label + measure.knob_label(kn), "b": baseline + measure.knob_label(base_kn),
                     "runs": len(runs)}
            for m in PAIRED_METRICS:
                entry[m] = diff([counts(by_run[rd], m) for rd in runs],
                                [counts(base[rd], m) for rd in runs], n_boot, seed)
            entry["sweep"] = {preservation.threshold_key(t): {
                m: diff([counts(by_run[rd], m, t) for rd in runs],
                        [counts(base[rd], m, t) for rd in runs], n_boot, seed)
                for m in PAIRED_SWEEP} for t in thresholds}
            out.append((_sort_key((exp, cond, measure.setting_key(by_run[runs[0]]))), entry))
    return [e for _, e in sorted(out, key=lambda pair: pair[0])]


# --- the threshold: near misses and calibration ----------------------------------------

def near_pairs(rows: list[dict]) -> list[dict]:
    """Every near miss of the measured rows once per (label, pre, post), with how many
    records met it and a few of their configurations, highest cosine first, and a blank
    `same` for a reader: does `post` still say what `pre` said?"""
    out: dict[tuple, dict] = {}
    for r in rows:
        for n in r.get("near") or ():
            e = out.setdefault((n["label"], n["pre"], n["post"]),
                               {**n, "records": 0, "configs": [], "same": None})
            e["records"] += 1
            if r["config"] not in e["configs"] and len(e["configs"]) < 5:
                e["configs"].append(r["config"])
    return sorted(out.values(), key=lambda e: (-e["cosine"], e["label"], e["pre"]))


def read_labels(path: Path) -> list[dict]:
    """A hand-labelled pairs file (preservation_pairs.jsonl with `same` filled in)."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# --- tables ---------------------------------------------------------------------------

def runs_cell(n: int) -> str:
    return f"{n} (small sample)" if n < SMALL else str(n)


def rate_cell(s: dict | None) -> str:
    """'62% (31/50) [48, 74]': pooled rate, counts, Wilson 95%."""
    if s is None:
        return "-"
    lo, hi = s["wilson"]
    return f"{100 * s['rate']:.0f}% ({s['n']}/{s['d']}) [{100 * lo:.0f}, {100 * hi:.0f}]"


def boot_cell(s: dict | None) -> str:
    if s is None or s.get("boot") is None:
        return "-"
    lo, hi = s["boot"]
    return f"[{100 * lo:.0f}, {100 * hi:.0f}]"


def diff_cell(d: list[float] | None) -> str:
    return "-" if d is None else measure._ci(tuple(d), 100)


def sweep_cell(s: dict | None) -> str:
    """'36% [28, 45]': pooled rate and run-bootstrap 95%."""
    if s is None:
        return "-"
    return f"{100 * s['rate']:.0f}% {boot_cell(s)}"


def share(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


COUNTED = ("ok", "missing_copy", "copy_mismatch", "no_work_subdir", "missing_pre")


def subdir_cell(inp: dict) -> str:
    if inp["subdir"] is None:
        return "none (pass FILE=SUBDIR)"
    return f"{inp['subdir']} ({inp['subdir_from']})"


def coverage_rows(inputs: list[dict], rows: list[dict]) -> list[list]:
    out = []
    for inp in inputs:
        rs = [r for r in rows if r["records"] == inp["records"]]
        st = Counter(r["status"] for r in rs)
        ok = [r for r in rs if r["status"] == "ok"]
        other = sum(v for k, v in st.items() if k not in COUNTED)
        out.append([inp["records"], inp["state"], subdir_cell(inp), inp["read"],
                    inp["checkpoint0"], inp["later_checkpoints"], st.get("ok", 0),
                    st.get("missing_copy", 0), st.get("copy_mismatch", 0),
                    st.get("no_work_subdir", 0), st.get("missing_pre", 0), other,
                    sum((r.get("reinfection_dropped") or 0) > 0 for r in ok),
                    sum((r.get("deepened") or 0) > 0 for r in ok),
                    sum(r.get("copy_from") == "record" for r in ok)])
    return out


COVERAGE_HEADER = ["records file", "state", "work subdir", "records", "checkpoint 0",
                   "later checkpoints (not measured)", "measured", "missing copy",
                   "copy disagrees with record", "no work subdir", "missing snapshot",
                   "other failures", "reinfected copies undone", "--depth stores rebuilt",
                   "copy named by record"]
GROUP_HEADER = ["configuration", "runs", "BCP (Wilson 95%)", "BCP run-bootstrap 95%",
                "BCP exact only", "BCP lineage claims only", "ACR (Wilson 95%)",
                "ACR run-bootstrap 95%", "ACR exact only", "ambiguous claims removed", "not measured"]
PAIRED_HEADER = ["A", "B", "runs", "BCP A-B (pp)", "BCP exact-only A-B (pp)",
                 "BCP lineage-only A-B (pp)", "ACR A-B (pp)", "ACR exact-only A-B (pp)"]
CALIBRATION_HEADER = ["cosine >=", "labelled pairs", "kept by both", "kept by cosine only",
                      "kept by reader only", "precision", "recall", "F1"]


def sweep_header(first: list[str], thresholds: tuple[float, ...]) -> list[str]:
    return [*first, "exact only", *(f"cosine >= {t:.2f}" for t in thresholds)]


def sweep_rows(groups: list[dict], metric: str, thresholds: tuple[float, ...]) -> list[list]:
    """One configuration per row: the rate exact-only, then at each threshold."""
    exact = {"bcp": "bcp_exact", "acr": "acr_exact"}[metric]
    return [[g["config"], runs_cell(g["runs"]), sweep_cell(g[exact]),
             *(sweep_cell(g["sweep"][preservation.threshold_key(t)][metric]) for t in thresholds)]
            for g in groups]


def paired_sweep_rows(pairs: list[dict], metric: str, thresholds: tuple[float, ...]) -> list[list]:
    return [[_pair_name(p), p["b"], runs_cell(p["runs"]), diff_cell(p[f"{metric}_exact"]),
             *(diff_cell(p["sweep"][preservation.threshold_key(t)][metric]) for t in thresholds)]
            for p in pairs]


def _pair_name(p: dict) -> str:
    return p["a"] + ("" if p["source"] == "patched" else f" ({p['source']})")


def calibration_section(params: dict, calibration: list[dict]) -> list[str]:
    head = "## The threshold against hand labels\n"
    if not calibration:
        return [head, "No labels given. Fill in `same` (true/false) on a sample of "
                f"`{PAIRS_FILE}` (does the claim after recovery still say what the claim "
                "before it said?) and pass the file with `--calibration`; this section then "
                "reports each threshold's precision and recall against those labels.\n"]
    rows = [[f"{c['threshold']:.2f}", c["pairs"], c["tp"], c["fp"], c["fn"],
             share(c["precision"]), share(c["recall"]),
             "-" if c["f1"] is None else f"{c['f1']:.2f}"] for c in calibration]
    return [head, f"Labels: `{params['calibration']}`. A pair is kept by cosine when its "
            "cosine reaches the threshold; recall below 100% is a reworded benign claim "
            "the metric counts lost.\n", measure.md_table(CALIBRATION_HEADER, rows), ""]


def group_cells(g: dict) -> list:
    missing = sum(v for k, v in g["status"].items() if k != "ok")
    return [g["config"], runs_cell(g["runs"]), rate_cell(g["bcp"]), boot_cell(g["bcp"]),
            rate_cell(g["bcp_exact"]), rate_cell(g["bcp_lineage"]), rate_cell(g["acr"]),
            boot_cell(g["acr"]), rate_cell(g["acr_exact"]), rate_cell(g["ambiguous_removed"]),
            missing or ""]


def tables(params: dict, inputs: list[dict], rows: list[dict], groups: list[dict],
           pairs: list[dict], calibration: list[dict] = ()) -> str:
    thresholds = tuple(params.get("thresholds") or ())
    parts = [
        "# Benign claim preservation (BCP) and attack claim removal (ACR)\n",
        "Computed post hoc from the stores by `bcp.py` (metric: `memheal/preservation.py`); "
        "no model calls. For each record, the affected memories are the oracle poison "
        "lineage of the pre-target snapshot plus every memory the recovery left less "
        "retrievable than it found it (quarantined, absorbed into a rebuild, or not "
        "re-created by a rollback; read from the stores, not the recovery's log), counted "
        f"when {params['scope']!r} before recovery. Their claims (deterministic "
        "sentence/clause split, deduplicated) are labelled with the scenario's AttackSpec: "
        "attack (names the attacker), ambiguous (undermines the legitimate system without "
        "naming a replacement; in neither rate) or benign. A claim is kept if its words "
        "appear in an active memory after recovery, or if an active claim with the same "
        f"label has cosine >= {params['threshold']} ({params['embedder']}). BCP = kept benign "
        "/ benign; ACR = attack claims no longer kept / attack claims. 'Exact only' drops "
        "the cosine route: stricter for BCP, more lenient for ACR. 'Lineage claims only' "
        "counts benign claims of lineage memories alone, the same claims for every "
        "configuration of a run, so it compares modes on one denominator.\n",
        "Rates pool claims over runs. Wilson intervals treat claims as independent; the "
        "run-bootstrap intervals resample runs and are the honest ones (claims of one run "
        f"move together). Cells over fewer than {SMALL} runs say 'small sample'.\n",
        "## Inputs and coverage\n",
        measure.md_table(COVERAGE_HEADER, coverage_rows(inputs, rows)),
        "\nWork subdir: where the copies of records that do not name their own were "
        "looked for, decided once per file (given on the command line, known for the "
        "existing sweeps, or by name), never by trying another experiment's folder. A copy "
        "whose quarantined, restored or reconstructed count disagrees with its record's "
        "was written by another recovery and is not measured.\n",
        "The threshold decides whether a reworded claim counts as kept; a repair that keeps "
        "a claim's words passes at any threshold, one that rewords it does not. The "
        "'by threshold' tables give every rate and paired difference again at each "
        f"threshold of the sweep, and `{PAIRS_FILE}` lists the near misses for labelling.\n",
        *calibration_section(params, list(calibration)),
    ]
    for exp in dict.fromkeys(g["experiment"] for g in groups):
        for cond in dict.fromkeys(g["condition"] for g in groups if g["experiment"] == exp):
            gs = [g for g in groups if g["experiment"] == exp and g["condition"] == cond]
            title = f"## {exp}: {cond} runs"
            if cond == "clean":
                title += " (false-alarm control: nothing is attack, every loss is collateral)"
            parts += [title + "\n", measure.md_table(GROUP_HEADER, [group_cells(g) for g in gs]), ""]
            if thresholds:
                for metric, name in (("bcp", "BCP"), ("acr", "ACR")):
                    parts += [f"### {exp}: {cond} runs, {name} by threshold\n",
                              measure.md_table(sweep_header(["configuration", "runs"], thresholds),
                                               sweep_rows(gs, metric, thresholds)), ""]
            ps = [p for p in pairs if p["experiment"] == exp and p["condition"] == cond]
            if ps:
                parts += [f"### {exp}: {cond} runs, paired differences against {params['baseline']}\n",
                          measure.md_table(PAIRED_HEADER, [
                              [_pair_name(p), p["b"], runs_cell(p["runs"]), diff_cell(p["bcp"]),
                               diff_cell(p["bcp_exact"]), diff_cell(p["bcp_lineage"]),
                               diff_cell(p["acr"]), diff_cell(p["acr_exact"])] for p in ps]),
                          "\nA minus B on the runs both measured, percentile bootstrap 95% "
                          "over runs.\n"]
                if thresholds:
                    for metric, name in (("bcp", "BCP"), ("acr", "ACR")):
                        parts += [f"### {exp}: {cond} runs, paired {name} A-B (pp) by threshold\n",
                                  measure.md_table(sweep_header(["A", "B", "runs"], thresholds),
                                                   paired_sweep_rows(ps, metric, thresholds)), ""]
    return "\n".join(parts) + "\n"


# --- figure ---------------------------------------------------------------------------

def strategy(mode: str) -> int:
    """Index into STRATEGIES: quarantine, repair, no recovery, or other."""
    for i, (_, _, modes) in enumerate(STRATEGIES):
        if mode in modes or (i == 1 and mode.startswith("heal")):
            return i
    return len(STRATEGIES) - 1


def _point(g: dict, metric: str) -> tuple[float, float, float] | None:
    s = g[metric]
    if s is None:
        return None
    lo, hi = s["boot"] or s["wilson"]
    return 100 * s["rate"], 100 * lo, 100 * hi


def _clusters(points: list[dict], tol: float = 2.0) -> list[list[dict]]:
    """Points within `tol` percentage points of each other on both axes share one label,
    so identical configurations (selective and selective+immunize) do not overprint."""
    out: list[list[dict]] = []
    for p in sorted(points, key=lambda p: (-p["y"], p["x"])):
        for c in out:
            if abs(c[0]["x"] - p["x"]) <= tol and abs(c[0]["y"] - p["y"]) <= tol:
                c.append(p)
                break
        else:
            out.append([p])
    return out


def plot(groups: list[dict], path: Path) -> Path | None:
    """BCP against ACR, one point per configuration of the poisoned runs (patched replay
    pages), one panel per experiment; crosshairs are run-bootstrap 95% intervals."""
    panels: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        x, y = _point(g, "acr"), _point(g, "bcp")
        if g["condition"] == "poisoned" and g["source"] == "patched" and x and y:
            panels[g["experiment"]].append({"g": g, "x": x[0], "xlo": x[1], "xhi": x[2],
                                            "y": y[0], "ylo": y[1], "yhi": y[2]})
    if not panels:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ncols = min(2, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, 3.4 * nrows + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    used, small = set(), False
    for ax, (exp, pts) in zip(axes.flat, panels.items()):
        _style(ax)
        for p in pts:
            slot = strategy(p["g"]["mode"])
            used.add(slot)
            colour, hollow = PALETTE[slot], p["g"]["runs"] < SMALL
            small |= hollow
            ax.plot([p["xlo"], p["xhi"]], [p["y"]] * 2, color=colour, lw=1, alpha=0.6, zorder=2)
            ax.plot([p["x"]] * 2, [p["ylo"], p["yhi"]], color=colour, lw=1, alpha=0.6, zorder=2)
            ax.plot(p["x"], p["y"], marker=MARKERS[slot], ms=6, ls="none", color=colour,
                    mfc=SURFACE if hollow else colour, mec=colour, mew=1.4, zorder=3)
        runs = max(p["g"]["runs"] for p in pts)
        ax.set_title(f"{exp} (up to {runs} runs)", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("attack claims removed, ACR (%)", fontsize=8, color=INK2)
        ax.set_ylabel("benign claims kept, BCP (%)", fontsize=8, color=INK2)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker=MARKERS[i], ls="none", color=PALETTE[i], ms=6)
               for i in sorted(used)]
    names = [STRATEGIES[i][1] for i in sorted(used)]
    if small:
        handles.append(Line2D([], [], marker="o", ls="none", color=MUTED, mfc=SURFACE, ms=6))
        names.append(f"hollow: fewer than {SMALL} runs")
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=7,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11 if nrows == 1 else 0.06, 1, 1))
    # Labels go in last, once the axes have their final size, so their measured extents
    # are the ones that get saved.
    renderer = fig.canvas.get_renderer()
    for ax, pts in zip(axes.flat, panels.values()):
        _label(ax, pts, renderer)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# Where a label may sit relative to its point (offset in points, alignment), tried in
# order until one overlaps no other label, no marker and stays inside the axes.
_SPOTS = [(dx, dy, ha, va) for dy, va in ((0, "center"), (7, "bottom"), (-7, "top"),
                                           (14, "bottom"), (-14, "top"))
          for dx, ha in ((6, "left"), (-6, "right"))]


def _label(ax, pts: list[dict], renderer) -> None:
    """Name every configuration next to its point without overprinting: points that
    coincide share one label (_clusters), and each label takes the first spot in
    _SPOTS that collides with nothing already drawn."""
    frame = ax.get_window_extent(renderer)
    marks = [ax.transData.transform((p["x"], p["y"])) for p in pts]
    placed = []

    def draw(c: list[dict], spot: tuple):
        dx, dy, ha, va = spot
        t = ax.annotate("\n".join(p["g"]["config"] for p in c), (c[0]["x"], c[0]["y"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=5.5, color=INK2)
        return t, t.get_window_extent(renderer).padded(1.5)

    def fits(box) -> bool:
        return (frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0
                and box.y1 <= frame.y1 and not any(box.overlaps(b) for b in placed)
                and not any(box.contains(*m) for m in marks))

    for c in _clusters(pts):
        for spot in _SPOTS:
            t, box = draw(c, spot)
            if fits(box):
                break
            t.remove()
        else:
            t, box = draw(c, _SPOTS[0])            # nothing fits: the first spot, as is
        placed.append(box)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


# --- main -----------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", default=list(DEFAULT_RECORDS), metavar="JSONL",
                   help="recover.py records files, each optionally FILE=SUBDIR to name the "
                        "work subdirectory of its copies (default results/recovery/records.jsonl)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--threshold", type=float, default=preservation.DEFAULT_THRESHOLD,
                   help="cosine at or above which a same-label claim counts as kept, for the "
                        "main tables and the figure")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                   metavar="T", help="the sweep: every rate and paired difference again at "
                                     "each (--threshold is always added)")
    p.add_argument("--calibration", type=Path, default=None, metavar="JSONL",
                   help=f"hand-labelled pairs ({PAIRS_FILE} with `same` filled in): report "
                        "each threshold's precision and recall against them")
    p.add_argument("--workers", type=int, default=4, help="threads (runs are the unit)")
    p.add_argument("--scope", choices=sorted(preservation.SCOPES), default="retrievable",
                   help="which pre-recovery memories hold claims: active ones (retrievable), "
                        "or active and consolidated (stored)")
    p.add_argument("--baseline", default=DEFAULT_BASELINE,
                   help="configuration the paired differences are taken against")
    p.add_argument("--runs-dir", type=Path, default=RUNS)
    p.add_argument("--work-subdir", default=None, metavar="NAME",
                   help="work subdirectory of every file's copies (default: per file, see "
                        "the module docstring)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-parts", action="store_true",
                   help="ignore records a running batch has not merged yet")
    args = p.parse_args(argv)
    subdirs = [args.work_subdir] + [records_spec(r)[1] for r in args.records]
    for sub in subdirs:
        if sub is not None and not recover.SUBDIR_RE.fullmatch(sub):
            p.error(f"a work subdirectory must be one folder name, got {sub!r}")
    if not all(0 < t <= 1 for t in (args.threshold, *args.thresholds)):
        p.error("--threshold and --thresholds must be in (0, 1]")
    if args.calibration is not None and not args.calibration.is_file():
        p.error(f"--calibration: no file {args.calibration}")
    return args


def main(argv=None, embedder=None) -> dict:
    """Measure, aggregate, write. `embedder` is for tests; by default the local MiniLM."""
    args = parse_args(argv)
    specs = [(_repo_path(path), sub) for path, sub in map(records_spec, args.records)]
    jobs, inputs = collect_jobs(specs, not args.no_parts, args.work_subdir, Path(args.runs_dir))
    if embedder is None:
        from memheal.memory import get_embedder
        embedder = get_embedder()
    thresholds = tuple(sorted({*args.thresholds, args.threshold}))
    band = (thresholds[0] - BAND_MARGIN, min(1.0, thresholds[-1] + BAND_MARGIN))
    settings = Settings(Path(args.runs_dir), args.threshold, args.scope, thresholds, band)
    rows = measure_all(jobs, embedder, settings, args.workers)
    near = near_pairs(rows)
    for r in rows:
        r.pop("near", None)
    groups = summarize(rows, args.n_boot, args.seed, thresholds)
    pairs = paired(rows, args.baseline, args.n_boot, args.seed, thresholds)
    calibration = (preservation.calibrate(read_labels(args.calibration),
                                          preservation.EmbeddingCache(embedder), thresholds)
                   if args.calibration else [])
    params = {"threshold": args.threshold, "thresholds": list(thresholds), "band": list(band),
              "scope": args.scope, "baseline": args.baseline,
              "embedder": getattr(embedder, "name", type(embedder).__name__),
              "n_boot": args.n_boot, "seed": args.seed, "parts": not args.no_parts,
              "calibration": _rel(args.calibration) if args.calibration else None}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preservation.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out / PAIRS_FILE, "w") as f:
        f.writelines(json.dumps(n) + "\n" for n in near)
    (out / "preservation_tables.md").write_text(
        tables(params, inputs, rows, groups, pairs, calibration))
    status = Counter(r["status"] for r in rows)
    summary = {"params": params, "inputs": inputs, "status": dict(status),
               "groups": groups, "paired": pairs, "calibration": calibration,
               "near_pairs": len(near),
               "not_measured": [{k: r.get(k) for k in ("records", "run_dir", "config", "status",
                                                       "work_db", "disagrees", "error")}
                                for r in rows if r["status"] != "ok"]}
    (out / "preservation.json").write_text(json.dumps(summary, indent=1))
    figure = plot(groups, out / "preservation.png")
    print(f"{len(rows)} records: " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    for name in ("preservation.jsonl", "preservation_tables.md", "preservation.json", PAIRS_FILE):
        print(f"wrote {out / name}")
    if figure:
        print(f"wrote {figure}")
    return summary


if __name__ == "__main__":
    main()
