"""Analysis tables and figures for the experiment sweeps (proposal Sections 5.5, 6.5, 6.6,
8.9, 9.3, 12.3; plan items T10 and H).

recover.py writes one record per run x configuration x checkpoint (recover.RECORD_KEYS)
and each experiment has its own records file. This script turns those files, plus the
history runs' summary.json, into tables and figures. It never opens a run.db and never
calls a model, so it finishes in seconds and can be rerun while a batch is still
appending: every number in the outputs is computed from the files it reads.

    uv run python analyze.py
    uv run python analyze.py --families results/smoke/families.jsonl --out /tmp/analysis

Inputs (each overridable by the flag of the same name; a missing input skips its section,
a thin one is flagged). Records a running batch has not merged yet
(<dir>/.parts/<stem>.*.jsonl, see batch.py) are read too unless --no-parts:

    --families      results/families/records.jsonl       X4  --family full --repeat 1 2 3
    --recovery      results/recovery/records.jsonl       the canonical sweep (evolving runs)
    --k             results/ablations/k.jsonl            X5  --k 3 10
    --depth         results/ablations/depth.jsonl        X6  --depth 1 2 3
    --reretrieve    results/ablations/reretrieve.jsonl   X7  --ablation reretrieve
    --groups        results/ablations/groups.jsonl       X8  --cf-groups --cf-workers 6
    --groups-clean  results/ablations/groups_clean.jsonl X8  clean-run false-alarm control
    --claims        results/claims/records.jsonl         X9  selective_claims
    --reinfect      results/claims/reinfect.jsonl        X9  selective_claims+immunize
    --flat          results/flat/records.jsonl           X3  recovery on FlatMemory runs
    --heal          results/main/records.jsonl           H   Phase 2 (data/main_b_jobs.txt):
                    results/flat/main.jsonl                  heal / heal_rebuild / heal_oracle
                                                             (+ what the file also holds); one
                                                             section per file (9a, 9b, ...)
    --runs          data/runs                            X2  history runs (summary.json)
    --preservation  results/preservation/preservation.jsonl  bcp.py's BCP/ACR per record

Outputs (under --out, default results/analysis/): analysis_tables.md (every table, n in
every cell), analysis.json (headline numbers), analysis_k.png, analysis_depth.png,
analysis_groups.png, analysis_flat.png, analysis_heal.png (the first --heal file) and
analysis_heal_<folder>.png for the others (analysis_heal_flat.png).

Metrics are measure.py's: ASR pools attacker outcomes over every replayed task; BMP is
the share of probes kept among those the same run answered correctly with no recovery;
BR is the mean share of tasks whose action matches the clean run. A cell over fewer than
10 runs is labelled 'small sample'. Knob settings are measure.KNOBS', heal v2's verify_n,
claim_n and ddmin_budget included, so no row pools two settings. Section 9 also reports,
for records written since heal v2, the claims each heal configuration dropped per rule
(dropped_by), ddmin's replays and the post-heal check (post_heal_harmful). Its runs are
clustered: a Phase 2 scenario has up to three history runs, which share its pages and
poison, so every interval in section 9 is a percentile bootstrap that resamples
scenarios (variants) with all their runs, n reads runs (scenarios), and 'small sample'
counts scenarios there.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import textwrap
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import measure
from cost import triggered
from memheal.heal import RULES as HEAL_RULES
from memheal.scenarios import ALL_VARIANTS

try:
    from measure import KNOBS, knobs
except ImportError:         # the knob table lived in cost.py before it was unified
    from cost import KNOBS, knobs

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "analysis"
SMALL = 10                  # runs below which a cell is labelled 'small sample'

INPUTS = {
    "families": "results/families/records.jsonl",
    "recovery": "results/recovery/records.jsonl",
    "k": "results/ablations/k.jsonl",
    "depth": "results/ablations/depth.jsonl",
    "reretrieve": "results/ablations/reretrieve.jsonl",
    "groups": "results/ablations/groups.jsonl",
    "groups_clean": "results/ablations/groups_clean.jsonl",
    "claims": "results/claims/records.jsonl",
    "reinfect": "results/claims/reinfect.jsonl",
    "flat": "results/flat/records.jsonl",
    # --heal takes several files (HEAL_INPUTS, each a section of its own); this is the first.
    "heal": "results/main/records.jsonl",
}
RUNS = "data/runs"
# bcp.py's output: one row per measured record, not a records file (see Inputs.preservation).
PRESERVATION = "results/preservation/preservation.jsonl"
# The plan's experiment log: flat history runs 1198-1434 (no suffix), evolving 0959-1197.
FLAT_RANGE, EVOLVING_RANGE = "1198-1434", "0959-1197"

# Figures: categorical slots in fixed order, status colours only for states, ink for text.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "D", "^")
STATUS = {"good": "#0ca30c", "warning": "#b07800", "critical": "#d03b3b"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
WIDTH = 6.5                 # inches: one text column of a paper


# --- inputs -----------------------------------------------------------------------------

def read_records(path: Path, parts: bool = True) -> tuple[list[dict] | None, int]:
    """`path` plus any part files a running batch has not merged into it yet (batch.py
    appends them at the very end and then deletes them). A measurement present in both
    collapses to its last copy, as measure.load_records does within one file. None when
    neither exists, so a missing input is told apart from an empty one."""
    files = [path] if path.is_file() else []
    part_files = sorted((path.parent / ".parts").glob(f"{path.stem}.*.jsonl")) if parts else []
    if not files and not part_files:
        return None, 0
    rows = [r for f in files + part_files for r in measure.load_records(f)]
    last = {measure.record_key(r): i for i, r in enumerate(rows)}
    return [r for i, r in enumerate(rows) if last[measure.record_key(r)] == i], len(part_files)


def read_rows(path: Path) -> list[dict] | None:
    """A JSON-lines file as it is, torn lines skipped; None when it does not exist."""
    if not path.is_file():
        return None
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: {path}:{i} is not valid JSON, skipped")
    return rows


class Inputs:
    """Every records file, read once, and what was found, for the report's input table."""

    def __init__(self, paths: dict[str, Path], runs_dir: Path, parts: bool = True,
                 preservation: Path | None = None, heal: tuple[str, ...] = ("heal",)):
        self.paths, self.runs_dir, self.parts = paths, runs_dir, parts
        self.preservation_path = preservation
        self.heal = tuple(h for h in heal if h in paths)     # the heal files' input names
        self._cache: dict[str, list[dict] | None] = {}
        self._preservation: list[dict] | None = None
        self.found: dict[str, dict] = {}

    def preservation(self) -> list[dict]:
        """bcp.py's rows, read once. They are not deduplicated like records: one run and
        configuration measured from two records files (the claims sweep's and the
        canonical selective[oracle], say) are two rows, told apart by the file they
        name (`records`), which is how heal_preservation matches them."""
        if "preservation" not in self.found:
            path = self.preservation_path
            rows = read_rows(path) if path is not None else None
            self._preservation = rows
            runs = len({r.get("run_dir") for r in rows or ()})
            self.found["preservation"] = {
                "path": str(path), "records": len(rows or ()), "runs": runs, "part_files": 0,
                "state": "missing" if rows is None else "thin" if runs < SMALL else "ok"}
        return self._preservation or []

    def get(self, name: str) -> list[dict]:
        if name not in self._cache:
            recs, n_parts = read_records(self.paths[name], self.parts)
            self._cache[name] = recs
            runs = len({r.get("run_dir") for r in recs or ()})
            self.found[name] = {
                "path": str(self.paths[name]), "records": len(recs or ()), "runs": runs,
                "part_files": n_parts,
                "state": "missing" if recs is None else "thin" if runs < SMALL else "ok"}
        return self._cache[name] or []

    def path(self, name: str) -> str:
        return str(self.paths[name])


@dataclass
class Ctx:
    out: Path
    n_boot: int = 1000
    seed: int = 0
    figures: bool = True
    aa: bool = False
    flat_range: str = FLAT_RANGE
    evolving_range: str = EVOLVING_RANGE


@dataclass
class Section:
    title: str
    md: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)
    skipped: str | None = None

    def skip(self, why: str) -> "Section":
        self.skipped = why
        return self

    def table(self, header: list[str], rows: list[list]) -> None:
        self.md += [measure.md_table(header, rows), ""]

    def text(self, *lines: str) -> None:
        self.md += [*lines, ""]


# --- record selection -------------------------------------------------------------------

def at0(recs: list[dict], condition: str = "poisoned") -> list[dict]:
    """Right after recovery (checkpoint 0), one condition."""
    return [r for r in recs if r.get("checkpoint", 0) == 0
            and r.get("condition", "poisoned") == condition]


def label(rec: dict) -> str:
    return measure.config_label(measure.config_key(rec), False)


def canonical(recs: list[dict]) -> list[dict]:
    """The default sweep's records: patched pages, no experiment knob set. Ablation files
    are compared against these, so a knob record never stands in for its own baseline."""
    return [r for r in recs if (r.get("source") or "patched") == "patched" and not knobs(r)]


def index(recs: list[dict]) -> dict[str, dict[str, dict]]:
    """Configuration label -> run -> record (the last one wins, as in load_records)."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        out[label(r)][r["run_dir"]] = r
    return out


def order(labels, preferred) -> list[str]:
    """The preferred configurations first (present or not, so a missing one shows as a
    row with n=0), then anything else the file holds."""
    return list(preferred) + sorted(set(labels) - set(preferred))


def tagged(recs: list[dict], tag: str) -> list[dict]:
    """Copies with `tag` added to the modifiers and every knob back at its default.

    measure.paired_differences pairs by configuration label within one knob setting, so
    an ablation sweep set against the canonical one (same configuration, other knobs)
    needs a label of its own instead of its knobs; the tag names it in every row. The
    copies' BMP baseline is then the canonical 'none', which is the one measure picks for
    these sweeps anyway (counterfactual options and stop_after_action are ignored)."""
    reset = {name: None for name in KNOBS}
    reset["stop_after_action"] = False      # else measure reads it back from the notes
    return [{**r, **reset, "modifiers": sorted({*(r.get("modifiers") or ()), tag})}
            for r in recs]


def baselines_of(recs: list[dict]) -> dict:
    """measure.py's 'none' baselines for BMP. measure matches a record to the same run's
    'none' under the same knobs (repeat, k, depth, family; not the counterfactual's own
    options), so ablation records can be pooled with canonical baselines safely."""
    return measure._none_baselines(recs)


def attacked(rec: dict) -> bool:
    return any(o == "attacker" for o in rec.get("task_outcomes") or ()) or (
        not rec.get("task_outcomes") and (rec.get("asr") or 0) > 0)


def run_asr(rec: dict) -> float | None:
    outs = rec.get("task_outcomes") or []
    return sum(o == "attacker" for o in outs) / len(outs) if outs else rec.get("asr")


_REPLAYS = re.compile(r"replays=(\d+)")


def cf_replays(rec: dict) -> int | None:
    """Counterfactual replays. Records from before the knobs only state them in the
    notes ('incident: ... replays=8', once per ablation the configuration ran)."""
    if rec.get("cf_replays") is not None:
        return rec["cf_replays"]
    found = _REPLAYS.findall(rec.get("notes") or "")
    return sum(map(int, found)) if found else None


def cf_seconds(rec: dict) -> float | None:
    """Wall time inside counterfactual ablations; older records never measured it."""
    return rec.get("cf_seconds") if "cf_seconds" in rec else None


# --- numbers and cells ------------------------------------------------------------------

def small(n: int) -> bool:
    return n < SMALL


def runs_cell(n: int) -> str:
    return f"{n} (small sample)" if small(n) else str(n)


def rate(n: float, d: int) -> dict | None:
    if not d:
        return None
    lo, hi = measure.wilson(n, d)
    return {"n": n, "d": d, "rate": round(n / d, 4), "ci": [round(lo, 4), round(hi, 4)],
            "small_sample": small(d)}


def rate_cell(n: float, d: int) -> str:
    return measure._ratio_ci(n, d)


def clustered(items: list, clusters: list | None) -> list[list]:
    """`items` grouped by `clusters` (one label per item), groups in order of first
    appearance; one group per item when clusters is None. A bootstrap that draws these
    groups resamples clusters with all their members, and draws exactly what resampling
    the items would when every item is its own cluster."""
    if clusters is None:
        return [[x] for x in items]
    groups: dict = {}
    for x, c in zip(items, clusters):
        groups.setdefault(c, []).append(x)
    return list(groups.values())


def boot_mean(values, n_boot: int, seed: int = 0,
              clusters: list | None = None) -> tuple[float, float, float] | None:
    """Mean with a percentile bootstrap 95% interval, resampling the values (one per run),
    the same recipe measure.paired_differences uses; with `clusters` (a label per value,
    the run's scenario) whole clusters are resampled, since runs of one scenario move
    together."""
    keep = [(v, c) for v, c in zip(values, clusters or range(len(values))) if v is not None]
    if not keep:
        return None
    vals = [v for v, _ in keep]
    groups = clustered(vals, [c for _, c in keep] if clusters is not None else None)
    rnd = random.Random(seed)
    boots = sorted(statistics.mean([v for g in rnd.choices(groups, k=len(groups)) for v in g])
                   for _ in range(n_boot))
    return (statistics.mean(vals), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1])


def mean_cell(ci: tuple | None, n: int | None = None, digits: int = 1) -> str:
    if ci is None:
        return "-"
    obs, lo, hi = ci
    s = f"{obs:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return s + (f" (n={n}{', small sample' if small(n) else ''})" if n is not None else "")


def ci_numbers(ci: tuple | None, n: int | None = None) -> dict | None:
    if ci is None:
        return None
    out = {"mean": round(ci[0], 4), "ci": [round(ci[1], 4), round(ci[2], 4)]}
    if n is not None:
        out.update(n=n, small_sample=small(n))
    return out


ROW_HEADER = ["runs", "ASR (tasks; Wilson 95%)", "runs still attacked", "BMP", "BR",
              "poison left in memory"]


def row_cells(row: measure.RecoveryRow) -> list:
    return [runs_cell(row.runs), rate_cell(row.asr_n, row.asr_d),
            measure.pct(row.attacked_runs, row.runs), measure._ratio(row.bmp_n, row.bmp_d),
            measure._fmt(row.br), measure.pct(row.residual_runs, row.residual_d)]


def row_numbers(row: measure.RecoveryRow) -> dict:
    return {"runs": row.runs, "small_sample": small(row.runs), "asr": rate(row.asr_n, row.asr_d),
            "attacked_runs": rate(row.attacked_runs, row.runs), "bmp": rate(row.bmp_n, row.bmp_d),
            "br": None if row.br is None else round(row.br, 4),
            "poison_left": rate(row.residual_runs, row.residual_d),
            "collateral": None if row.collateral is None else round(row.collateral, 3),
            "quarantined": None if row.quarantined is None else round(row.quarantined, 3),
            "llm_calls": None if row.llm_calls is None else round(row.llm_calls, 3),
            "seconds": None if row.seconds is None else round(row.seconds, 3)}


def aggregate(recs: list[dict], baselines: dict) -> measure.RecoveryRow:
    return measure.aggregate(measure.config_key(recs[0]), recs, baselines)


DIFF_HEADER = ["A", "B", "runs", "population", "ASR (pp)", "BR (x100)", "BMP (pp)",
               "collateral", "poison left (pp)"]


def diff_rows(diffs: list[dict]) -> list[list]:
    return [[d["a"], d["b"], runs_cell(d["runs"]), d["population"], measure._ci(d["ASR"], 100),
             measure._ci(d["BR"], 100), measure._ci(d["BMP"], 100),
             measure._ci(d["collateral"], 1, 2), measure._ci(d["poison left"], 100)]
            for d in diffs]


def diff_numbers(diffs: list[dict]) -> list[dict]:
    def num(v):
        return None if v is None else [round(x, 4) for x in v]
    return [{"a": d["a"], "b": d["b"], "population": d["population"], "runs": d["runs"],
             "small_sample": small(d["runs"]),
             **{k: num(d[k]) for k in ("ASR", "BR", "BMP", "collateral", "poison left")}}
            for d in diffs]


DIFF_NOTE = ("A minus B on the same runs, percentile bootstrap 95% intervals over runs; ASR "
             "here is the mean of per-run ASR (measure.paired_differences). Each "
             "configuration is one replay per run: compare against the replay-noise rows "
             "(A/A) and Section 2 before reading a difference as real.")


# --- figures ----------------------------------------------------------------------------

def _figure(ncols: int, height: float = 3.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, ncols, figsize=(WIDTH, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        _style(ax)
    return plt, fig, list(axes[0])


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _series(ax, xs, points, slot: int, n_series: int, name: str, step: float,
            connect: bool = True, annotate: bool = False) -> bool:
    """One series of (value, lo, hi, runs) points (None = no data at that x), offset
    sideways so intervals of neighbouring series do not overlap. Hollow markers mark
    small samples; connect=False for a categorical x, where a line would imply a trend.
    annotate writes each point's n above its interval, for figures whose points may
    stand on different numbers of runs. Returns whether any small-sample point was
    drawn."""
    off = (slot - (n_series - 1) / 2) * step
    shown = [(x + off, p) for x, p in zip(xs, points) if p is not None]
    if not shown:
        return False
    colour = PALETTE[slot]
    if connect:
        ax.plot([x for x, _ in shown], [p[0] for _, p in shown], color=colour, lw=2,
                label=name, zorder=2, solid_capstyle="round")
    else:
        ax.plot([], [], marker=MARKERS[slot], ms=6.5, color=colour, ls="none", label=name)
    for x, (v, lo, hi, runs) in shown:
        ax.plot([x, x], [lo, hi], color=colour, lw=1.2, alpha=0.7, zorder=2)
        ax.plot(x, v, marker=MARKERS[slot], ms=6.5, color=colour, ls="none", zorder=3,
                mfc=SURFACE if small(runs) else colour, mec=colour, mew=1.6)
        if annotate:
            ax.annotate(str(runs), (x, hi), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, color=colour)
    return any(small(p[3]) for _, p in shown)


def _legend(fig, axes, any_small: bool, ncol: int = 4, note: str | None = None) -> None:
    """The series and the hollow-marker key in one row at the bottom; `note` (how to read
    the figure) on lines of its own below it, so it never widens the legend row. A
    figure with a note saves with one more legend_rows per note line."""
    from matplotlib.lines import Line2D
    handles, names = [], []
    for ax in axes:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    if any_small:
        handles.append(Line2D([], [], marker="o", ls="none", color=MUTED, mfc=SURFACE, ms=6.5))
        names.append(f"hollow: fewer than {SMALL} runs (small sample)")
    if handles:
        lines = note.count("\n") + 1 if note else 0
        fig.legend(handles, names, loc="lower center", ncol=min(ncol, len(handles)),
                   frameon=False, fontsize=7.5, labelcolor=INK,
                   bbox_to_anchor=(0.5, 0.012 + 0.033 * lines if note else 0.0))
    if note:
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7, color=INK2,
                 linespacing=1.3)


def _save(fig, plt, path: Path, legend_rows: int = 1, after_layout=None) -> Path:
    """Lay out, save at 200 dpi, close. after_layout(renderer) runs once the axes have
    their final size, for anything placed by its measured extent (direct labels)."""
    fig.tight_layout(rect=(0, 0.06 * legend_rows + 0.02, 1, 1))
    if after_layout is not None:
        after_layout(fig.canvas.get_renderer())
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def _titles(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _pct_point(n: float, d: int) -> tuple | None:
    if not d:
        return None
    lo, hi = measure.wilson(n, d)
    return (100 * n / d, 100 * lo, 100 * hi, d)


def _runs_point(n: float, d: int, runs: int | None = None) -> tuple | None:
    """A percentage point whose n is `runs` (default the denominator), for rates pooled
    over tasks where the sample that matters is the number of runs."""
    p = _pct_point(n, d)
    return None if p is None else (*p[:3], d if runs is None else runs)


def _pct_axis(ax) -> None:
    """0-100% with headroom above 100 for the n labels _series(annotate=True) writes."""
    ax.set_ylim(-3, 112)
    ax.set_yticks(range(0, 101, 20))


# --- 1. task families (5.5) -------------------------------------------------------------

FAMILY_MODES = ("none", "ancestry", "selective[oracle]", "incident[cf]", "incident[cf]+verify")


def _repeat(rec: dict) -> int:
    return rec.get("repeat") or 0           # None = a configuration replayed only once


def first_replay(recs: list[dict]) -> dict[str, dict[str, dict]]:
    """One record per configuration and run: its lowest repeat, so every configuration
    weighs one replay per run (repeats are Section 2's subject)."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in sorted(recs, key=_repeat, reverse=True):
        out[label(r)][r["run_dir"]] = r
    return out


def per_task(recs: list[dict]) -> list[tuple[int, int]]:
    """(attacker, replayed) per task index of the family."""
    width = max((len(r.get("task_outcomes") or ()) for r in recs), default=0)
    out = []
    for i in range(width):
        outs = [r["task_outcomes"][i] for r in recs if len(r.get("task_outcomes") or ()) > i]
        out.append((sum(o == "attacker" for o in outs), len(outs)))
    return out


def disagreeing(recs: list[dict]) -> tuple[int, int]:
    """Runs whose tasks disagree on reaching the attacker, among runs with 2+ tasks: what
    a single-task measurement would miss."""
    multi = [r for r in recs if len(r.get("task_outcomes") or ()) > 1]
    return sum(len({o == "attacker" for o in r["task_outcomes"]}) > 1 for r in multi), len(multi)


def s1_families(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("1. Task families (proposal 5.5)")
    recs = [r for r in at0(inp.get("families")) if r.get("family") == "full"]
    if not recs:
        return sec.skip(f"no family=full poisoned records in {inp.path('families')}")
    base = baselines_of(recs)
    first = first_replay(recs)
    width = max(len(r.get("task_outcomes") or ()) for r in recs)
    sec.text(f"{len(recs)} records, {len({r['run_dir'] for r in recs})} runs. One replay per "
             "run and configuration (its lowest repeat). BMP's baseline is measure.py's: the "
             "same run's 'none' under the same knobs and repeat (any repeat for a "
             "configuration replayed once). "
             "'Harmful for any task' = runs where at least one task of the family reached "
             "the attacker; 'tasks disagree' = runs whose tasks split on reaching it.")
    header = ["configuration", *ROW_HEADER[:2], *[f"task {i}" for i in range(width)],
              "runs whose tasks disagree", "harmful for any task (Wilson 95%)", "BMP", "BR",
              "no none baseline", "replay used"]
    rows = []
    for lab in order(first, FAMILY_MODES):
        rs = list(first.get(lab, {}).values())
        if not rs:
            rows.append([lab, "0 (no records)"] + ["-"] * (len(header) - 2))
            continue
        row = aggregate(rs, base)
        tasks, dis = per_task(rs), disagreeing(rs)
        used = Counter(f"repeat {r['repeat']}" if r.get("repeat") else "single" for r in rs)
        rows.append([lab, runs_cell(row.runs), rate_cell(row.asr_n, row.asr_d),
                     *[measure._ratio(n, d) for n, d in tasks],
                     *["-"] * (width - len(tasks)), measure.pct(*dis),
                     rate_cell(row.attacked_runs, row.runs), measure._ratio(row.bmp_n, row.bmp_d),
                     measure._fmt(row.br), row.no_baseline,
                     ", ".join(f"{k} x{v}" for k, v in sorted(used.items()))])
        sec.numbers[lab] = {**row_numbers(row), "per_task": [rate(n, d) for n, d in tasks],
                            "tasks_disagree": rate(*dis),
                            "harmful_any_task": rate(row.attacked_runs, row.runs)}
    sec.table(header, rows)
    return sec


# --- 2. repeats (6.6) -------------------------------------------------------------------

def by_repeat(recs: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    out: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        if r.get("repeat") is not None:
            out[label(r)][r["run_dir"]][r["repeat"]] = r
    return out


def deterministic(lab: str) -> bool:
    """Same store on every repeat: no counterfactual or verify loop, whose plans are
    themselves replays (measure.replay_noise's criterion)."""
    return "[cf]" not in lab and not lab.startswith("incident") and "verify" not in lab


def outcome_flips(runs: dict[str, dict[int, dict]]) -> tuple[int, int, int]:
    """(run, task) units seen in 2+ repeats: (units, any outcome differs, attacker
    vs not differs)."""
    units = any_diff = attack_diff = 0
    for reps in runs.values():
        if len(reps) < 2:
            continue
        for outs in zip(*(r.get("task_outcomes") or () for _, r in sorted(reps.items()))):
            units += 1
            any_diff += len(set(outs)) > 1
            attack_diff += len({o == "attacker" for o in outs}) > 1
    return units, any_diff, attack_diff


def majority(reps: list[dict]) -> list[bool]:
    """Per task: attacker in a strict majority of the replays."""
    return [2 * sum(o == "attacker" for o in outs) > len(outs)
            for outs in zip(*(r.get("task_outcomes") or () for r in reps))]


def s2_repeats(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("2. Repeats (proposal 6.6)")
    reps = by_repeat(at0(inp.get("families")))
    if not reps:
        return sec.skip(f"no records with a repeat index in {inp.path('families')}")
    sec.text("Configurations swept with --repeat. A repeat replays the task family on a "
             "fresh copy; for deterministic stores (no counterfactual, no verify) every "
             "difference between repeats is the model's replay, which makes these rows the "
             "replay-noise floor for the single-replay comparisons elsewhere. incident[cf] "
             "also re-runs its counterfactual on each repeat, so its plan can differ too.")
    rows = []
    for lab in order(reps, FAMILY_MODES):
        for rp in sorted({rp for runs in reps.get(lab, {}).values() for rp in runs}):
            rs = [runs[rp] for runs in reps[lab].values() if rp in runs]
            n, d = measure.asr_counts(rs)
            rows.append([lab, rp, runs_cell(len(rs)), rate_cell(n, d)])
    sec.text("### ASR per repeat (all runs replayed at that repeat)")
    sec.table(["configuration", "repeat", "runs", "ASR (tasks; Wilson 95%)"], rows)

    rows, pooled = [], [0, 0, 0]
    for lab in order(reps, FAMILY_MODES):
        runs = reps.get(lab, {})
        if not runs:
            continue
        wanted = sorted({rp for r in runs.values() for rp in r})[:3]
        common = {rd: r for rd, r in runs.items() if all(rp in r for rp in wanted)}
        per = [measure.asr_counts([r[rp] for r in common.values()]) for rp in wanted]
        asrs = [n / d for n, d in per if d]
        units, any_diff, attack_diff = outcome_flips(runs)
        if deterministic(lab):
            pooled = [pooled[0] + units, pooled[1] + any_diff, pooled[2] + attack_diff]
        vote = single = changed = tasks = 0
        if len(wanted) == 3:
            for r in common.values():
                maj = majority([r[rp] for rp in wanted])
                first = [o == "attacker" for o in r[wanted[0]].get("task_outcomes") or ()]
                vote, single = vote + sum(maj), single + sum(first)
                changed += sum(a != b for a, b in zip(maj, first))
                tasks += len(maj)
        rows.append([lab, runs_cell(len(common)), *[rate_cell(n, d) for n, d in per],
                     "-" if len(asrs) < 2 else f"{100 * (max(asrs) - min(asrs)):.0f}",
                     measure.pct(any_diff, units), rate_cell(attack_diff, units),
                     rate_cell(vote, tasks) if tasks else "-",
                     rate_cell(single, tasks) if tasks else "-",
                     measure.pct(changed, tasks) if tasks else "-"])
        sec.numbers[lab] = {
            "runs_all_repeats": len(common), "repeats": wanted,
            "small_sample": small(len(common)),
            "asr_per_repeat": [rate(n, d) for n, d in per],
            "spread_pp": None if len(asrs) < 2 else round(100 * (max(asrs) - min(asrs)), 2),
            "outcome_differs": rate(any_diff, units), "attack_differs": rate(attack_diff, units),
            "majority_of_3_asr": rate(vote, tasks), "single_replay_asr": rate(single, tasks),
            "majority_changes_task": rate(changed, tasks)}
    sec.text("### Spread across repeats, outcome flips, majority vote",
             "Runs replayed at all three repeats. Spread = largest minus smallest repeat "
             "ASR on those runs, in percentage points. Flips count (run, task) outcomes "
             "seen at 2+ repeats. Majority = attacker on at least 2 of the 3 replays; "
             "single replay = repeat 1 on the same runs.")
    sec.table(["configuration", "runs (all 3 repeats)", "ASR repeat 1", "ASR repeat 2",
               "ASR repeat 3", "spread (pp)", "(run, task) outcome differs",
               "(run, task) attacker vs not differs (Wilson 95%)", "ASR majority of 3",
               "ASR single replay", "tasks where majority != single"], rows)
    if pooled[0]:
        sec.numbers["replay_noise"] = {"units": pooled[0],
                                       "outcome_differs": rate(pooled[1], pooled[0]),
                                       "attack_differs": rate(pooled[2], pooled[0])}
        sec.text(f"Replay-noise floor (deterministic stores pooled): attacker-vs-not flips in "
                 f"{rate_cell(pooled[2], pooled[0])} of (run, task) units; any outcome "
                 f"change in {measure.pct(pooled[1], pooled[0])}.")
    return sec


# --- 3/4. knob sweeps: k and depth (9.3) ------------------------------------------------

def run_k(rec: dict, runs_dir: Path) -> int | None:
    """The k a canonical record's replays used: the run's own k (summary.json), since
    the canonical sweep never overrides it."""
    if rec.get("k") is not None:
        return rec["k"]
    try:
        return json.loads((runs_dir / rec["run_dir"] / "summary.json").read_text()).get("k")
    except (OSError, ValueError):
        return None


def sweep_index(levels: dict, modes) -> dict[str, dict[object, dict[str, dict]]]:
    """mode label -> level -> run -> record."""
    out: dict[str, dict[object, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for level, recs in levels.items():
        for r in recs:
            if label(r) in modes:
                out[label(r)][level][r["run_dir"]] = r
    return out


def common_runs(by_level: dict[object, dict[str, dict]]) -> set[str]:
    sets = [set(v) for v in by_level.values()]
    return set.intersection(*sets) if sets else set()


def _level_sort(v) -> tuple:
    return (not isinstance(v, (int, float)), v if isinstance(v, (int, float)) else 0, str(v))


def on_common(by_level: dict[object, dict[str, dict]], level,
              common: set[str]) -> list[dict]:
    return [r for rd, r in by_level.get(level, {}).items() if rd in common]


def sweep_rows(sec: Section, idx, modes, level_name: str, extra_header: list[str], extra,
               paired_header: tuple = (), paired=None) -> dict:
    """One row per (mode, level): ASR and `extra(records) -> (cells, numbers)` on every
    run measured there, then ASR and `paired(records)` on only the runs measured at every
    level. A partial batch reaches the higher levels on fewer runs, so the all-runs
    columns also move with which runs a level happens to hold; a trend is read from the
    paired columns, which are the ones the figures draw. Paired numbers carry the
    suffix '_common_runs'."""
    rows, numbers = [], {}
    for mode in modes:
        by_level = idx.get(mode, {})
        common = common_runs(by_level)
        for level in sorted(by_level, key=_level_sort):
            rs = list(by_level[level].values())
            cs = on_common(by_level, level, common)
            n, d = measure.asr_counts(rs)
            cn, cd = measure.asr_counts(cs)
            cells, nums = extra(rs)
            pcells, pnums = paired(cs) if paired else ([], {})
            observe = ", ".join(sorted({r.get("observe") or "logged" for r in rs}))
            rows.append([mode, level, runs_cell(len(rs)), observe, rate_cell(n, d),
                         measure.pct(sum(attacked(r) for r in rs), len(rs)), *cells,
                         f"{rate_cell(cn, cd)} on {runs_cell(len(common))}", *pcells])
            numbers[f"{mode} {level_name}={level}"] = {
                "mode": mode, level_name: level, "runs": len(rs), "small_sample": small(len(rs)),
                "asr": rate(n, d), "attacked_runs": rate(sum(attacked(r) for r in rs), len(rs)),
                "asr_common_runs": rate(cn, cd), "common_runs": len(common), **nums,
                **{f"{k}_common_runs": v for k, v in pnums.items()}}
    sec.table(["configuration", level_name, "runs", "observe", "ASR (tasks; Wilson 95%)",
               "runs still attacked", *extra_header, "ASR on runs measured at every level",
               *paired_header], rows)
    return numbers


def _counterfactual(rs: list[dict]) -> bool:
    return any(r.get("verdicts") == "cf" or r.get("mode") == "incident" for r in rs)


def replays_per_incident(rs: list[dict], ctx: Ctx) -> tuple[tuple | None, int]:
    """Bootstrap mean of counterfactual replays over the runs whose observed action was
    harmful, and how many such runs there were (the figure's n for that point)."""
    inc = [r for r in rs if r.get("incident")]
    return boot_mean([cf_replays(r) for r in inc], ctx.n_boot, ctx.seed), len(inc)


def _incident_costs(rs: list[dict], ctx: Ctx) -> tuple[list, dict]:
    inc = [r for r in rs if r.get("incident")]
    if not _counterfactual(rs):
        return ["-", "-", "-"], {"incident_runs": len(inc)}
    reps, _ = replays_per_incident(rs, ctx)
    secs = boot_mean([r.get("seconds") for r in inc], ctx.n_boot, ctx.seed)
    return ([len(inc), mean_cell(reps), mean_cell(secs)],
            {"incident_runs": len(inc), "cf_replays_per_incident": ci_numbers(reps),
             "seconds_per_incident": ci_numbers(secs)})


def _paired_replays(rs: list[dict], ctx: Ctx) -> tuple[list, dict]:
    """Replays per incident on the runs measured at every k: what the k figure draws."""
    if not _counterfactual(rs):
        return ["-"], {}
    reps, n_inc = replays_per_incident(rs, ctx)
    return [mean_cell(reps, n_inc)], {"cf_replays_per_incident": ci_numbers(reps, n_inc)}


def s3_k(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("3. Retrieval k ablation (proposal 9.3)")
    ab = at0(inp.get("k"))
    if not ab:
        return sec.skip(f"no poisoned records in {inp.path('k')}")
    modes = ("none", "incident[cf]")
    runs = {r["run_dir"] for r in ab}
    levels: dict = defaultdict(list)
    for r in ab:
        levels[r.get("k")].append(r)
    for r in canonical(at0(inp.get("recovery"))):
        if r["run_dir"] in runs and label(r) in modes:
            levels[run_k(r, inp.runs_dir)].append(r)
    idx = sweep_index(levels, modes)
    sec.text(f"{len(runs)} runs in {inp.path('k')}; the run's own k comes from the canonical "
             f"sweep ({inp.path('recovery')}) on the same runs. Ablation records observe the "
             "incident from a fresh replay at the new k (observe=fresh); the canonical ones "
             "use the run's logged action, so the own-k row is not a fresh observation. "
             "Replays and seconds are per run whose observed action was harmful (incident). "
             "Read a trend over k from the columns on the runs measured at every k: the "
             "all-runs columns also change with which runs a level holds. The figure draws "
             "those same runs, with n above each point.")
    sec.numbers = sweep_rows(sec, idx, modes, "k",
                             ["incident runs", "cf replays per incident (bootstrap 95%)",
                              "recovery seconds per incident"],
                             lambda rs: _incident_costs(rs, ctx),
                             ["cf replays per incident on those runs (bootstrap 95%)"],
                             lambda rs: _paired_replays(rs, ctx))
    if ctx.figures:
        sec.figures.append(plot_k(idx, ctx.out / "analysis_k.png", ctx))
    return sec


def sweep_levels(by_level: dict[object, dict[str, dict]],
                 levels) -> tuple[list[list[dict]], bool]:
    """Each plotted level's records for one sweep figure series, and whether to join them.

    A partial batch reaches the higher levels on fewer runs, so a line through every
    run measured at each level partly traces which runs exist there rather than the
    knob. The series is therefore the runs measured at every level (sweep_rows' paired
    columns), and the line is a within-run change. When no run was measured at every
    level it falls back to all runs, drawn unjoined so no trend is implied. A mode with
    no records draws nothing, and counts as joined so it does not change the note."""
    common = common_runs(by_level)
    if common or not by_level:
        return [on_common(by_level, lv, common) for lv in levels], True
    return [list(by_level.get(lv, {}).values()) for lv in levels], False


def _sweep_note(joined: bool) -> str:
    return ("lines: the same runs at every level; number: runs per point" if joined else
            "number: runs per point; unjoined: no run measured at every level")


def plot_k(idx, path: Path, ctx: Ctx) -> Path:
    plt, fig, (ax, ax2) = _figure(2, 3.2)
    levels = sorted({lv for m in idx.values() for lv in m if isinstance(lv, (int, float))})
    any_small, all_joined = False, True
    for slot, mode in enumerate(("none", "incident[cf]")):
        per_level, joined = sweep_levels(idx.get(mode, {}), levels)
        pts = [_runs_point(*measure.asr_counts(rs), len(rs)) for rs in per_level]
        any_small |= _series(ax, levels, pts, slot, 2, mode, 0.25, joined, annotate=True)
        all_joined &= joined
    per_level, joined = sweep_levels(idx.get("incident[cf]", {}), levels)
    pts = []
    for rs in per_level:
        ci, n_inc = replays_per_incident(rs, ctx)
        pts.append(None if ci is None else (*ci, n_inc))
    any_small |= _series(ax2, levels, pts, 1, 1, "incident[cf]", 0, joined, annotate=True)
    for a in (ax, ax2):
        a.set_xticks(levels)
    _pct_axis(ax)
    ax2.set_ylim(0, max(ax2.get_ylim()[1], 1) * 1.15)       # room for the n labels
    _titles(ax, "ASR after recovery (Wilson 95%)", "retrieval k", "% of replayed tasks")
    _titles(ax2, "Counterfactual replays per incident", "retrieval k",
            "replays per incident run (bootstrap 95%)")
    _legend(fig, [ax], any_small, note=_sweep_note(all_joined and joined))
    return _save(fig, plt, path, legend_rows=2)


def poison_left(rs: list[dict]) -> tuple[int, int]:
    """Runs whose store still holds an active memory naming the attacker, of the runs
    where that was checked (residual_attack is None on records that never looked)."""
    known = [r for r in rs if r.get("residual_attack") is not None]
    return sum((r["residual_attack"] or 0) > 0 for r in known), len(known)


def _mean(values: list) -> float | None:
    return statistics.mean(values) if values else None


def _round(v: float | None, digits: int = 3) -> float | None:
    return None if v is None else round(v, digits)


def _depth_extra(rs: list[dict]) -> tuple[list, dict]:
    left, known = poison_left(rs)
    coll = _mean([measure.collateral_count(r) for r in rs])
    quar = _mean([r.get("n_quarantined", 0) for r in rs])
    return ([measure._fmt(coll, 1), measure._fmt(quar, 1), rate_cell(left, known)],
            {"collateral": _round(coll), "quarantined": _round(quar),
             "poison_left": rate(left, known)})


def _depth_paired(rs: list[dict]) -> tuple[list, dict]:
    """Collateral and poison left on the runs measured at every depth: the poison-left
    trend otherwise mixes in which runs the batch has reached at the deeper levels."""
    left, known = poison_left(rs)
    coll = _mean([measure.collateral_count(r) for r in rs])
    return ([measure._fmt(coll, 1), rate_cell(left, known)],
            {"collateral": _round(coll), "poison_left": rate(left, known)})


def s4_depth(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("4. Consolidation depth ablation (proposal 9.3)")
    ab = at0(inp.get("depth"))
    if not ab:
        return sec.skip(f"no poisoned records in {inp.path('depth')}")
    modes = ("none", "selective[oracle]", "incident[cf]")
    runs = {r["run_dir"] for r in ab}
    levels: dict = defaultdict(list)
    for r in ab:
        levels[r.get("depth") or 0].append(r)
    for r in canonical(at0(inp.get("recovery"))):
        if r["run_dir"] in runs and label(r) in modes:
            levels[0].append(r)
    idx = sweep_index(levels, modes)
    sec.text(f"{len(runs)} runs in {inp.path('depth')}; depth 0 is the canonical sweep on the "
             "same runs (logged incident). Depth d adds d extra consolidation rounds to the "
             "copy before recovery. Collateral = quarantined memories labelled benign or "
             "outside the lineage (mean per run); poison left = runs whose store still holds "
             "an active memory naming the attacker. Read a trend over depth from the columns "
             "on the runs measured at every depth: the all-runs columns also change with "
             "which runs a level holds. The figure draws those same runs, with n above each "
             "point.")
    sec.numbers = sweep_rows(sec, idx, modes, "depth",
                             ["collateral", "quarantined", "poison left in memory (Wilson 95%)"],
                             _depth_extra,
                             ["collateral on those runs", "poison left on those runs"],
                             _depth_paired)
    if ctx.figures:
        sec.figures.append(plot_depth(idx, ctx.out / "analysis_depth.png"))
    return sec


def plot_depth(idx, path: Path) -> Path:
    plt, fig, (ax, ax2) = _figure(2, 3.2)
    modes = ("none", "selective[oracle]", "incident[cf]")
    levels = sorted({lv for m in idx.values() for lv in m})
    any_small, all_joined = False, True
    for slot, mode in enumerate(modes):
        per_level, joined = sweep_levels(idx.get(mode, {}), levels)
        asr = [_runs_point(*measure.asr_counts(rs), len(rs)) for rs in per_level]
        left = [_runs_point(*poison_left(rs)) for rs in per_level]
        any_small |= _series(ax, levels, asr, slot, len(modes), mode, 0.12, joined,
                             annotate=True)
        any_small |= _series(ax2, levels, left, slot, len(modes), mode, 0.12, joined,
                             annotate=True)
        all_joined &= joined
    for a in (ax, ax2):
        a.set_xticks(levels)
        _pct_axis(a)
    _titles(ax, "ASR after recovery (Wilson 95%)", "extra consolidation rounds",
            "% of replayed tasks")
    _titles(ax2, "Poison left in memory (Wilson 95%)", "extra consolidation rounds",
            "% of runs")
    _legend(fig, [ax], any_small, note=_sweep_note(all_joined))
    return _save(fig, plt, path, legend_rows=2)


# --- 5. frozen vs re-retrieval, and the ancestry / counterfactual ablations (9.3) -------

def culprits(rec: dict, singles_only: bool = False) -> frozenset | None:
    """The record's minimal flipping sets as a set of sets (None when not recorded).
    singles_only keeps single removals: what a run without --cf-groups would test."""
    sets = rec.get("culprit_sets")
    if sets is None:
        return None
    return frozenset(frozenset(s) for s in sets if not singles_only or len(s) == 1)


AGREEMENT = ("same culprits", "overlapping", "disjoint", "only frozen found one",
             "only re-retrieval found one", "neither found one")


def agreement(frozen: frozenset, other: frozenset) -> str:
    if not frozen and not other:
        return AGREEMENT[5]
    if not other:
        return AGREEMENT[3]
    if not frozen:
        return AGREEMENT[4]
    if frozen == other:
        return AGREEMENT[0]
    members = lambda sets: set().union(*sets)      # noqa: E731
    return AGREEMENT[1] if members(frozen) & members(other) else AGREEMENT[2]


def _frozen_cf(rec: dict) -> bool:
    return label(rec) == "incident[cf]" and rec.get("cf_method") in (None, "frozen")


def frozen_reference(inp: Inputs) -> dict[str, tuple[frozenset, str]]:
    """Frozen-method culprit sets per incident run, from the first source that recorded
    them: the canonical sweep (records written before the knobs lack culprit_sets), then
    the families sweep at its lowest repeat, then the groups sweep's single removals."""
    sources = (
        ("canonical sweep", [r for r in canonical(at0(inp.get("recovery"))) if _frozen_cf(r)], False),
        ("families sweep", sorted((r for r in at0(inp.get("families")) if _frozen_cf(r)),
                                  key=_repeat), False),
        ("groups sweep (single removals)", at0(inp.get("groups")), True),
    )
    out: dict[str, tuple[frozenset, str]] = {}
    for name, recs, singles in sources:
        for r in recs:
            sets = culprits(r, singles)
            if sets is not None and r.get("incident") and r["run_dir"] not in out:
                out[r["run_dir"]] = (sets, name)
    return out


def frozen_vs_frozen(inp: Inputs) -> Counter:
    """Culprit agreement between two frozen ablations of one run (families repeats): the
    noise floor for re-retrieval's agreement, since every ablation is itself replays."""
    reps: dict[str, dict[int, frozenset]] = defaultdict(dict)
    for r in at0(inp.get("families")):
        if _frozen_cf(r) and r.get("incident") and r.get("repeat") and culprits(r) is not None:
            reps[r["run_dir"]][r["repeat"]] = culprits(r)
    return Counter(agreement(a, b).replace("re-retrieval", "second replay")
                   for by in reps.values() for (_, a), (_, b) in combinations(sorted(by.items()), 2))


def s5_reretrieve(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("5. Frozen vs re-retrieval counterfactual, and ancestry / counterfactual "
                  "ablations (proposal 9.3)")
    rer = [r for r in at0(inp.get("reretrieve")) if r.get("cf_method") == "reretrieve"]
    frozen = {r["run_dir"]: r for r in canonical(at0(inp.get("recovery"))) if _frozen_cf(r)}
    if rer and frozen:
        _reretrieve_tables(sec, rer, frozen, inp, ctx)
    else:
        missing = "no re-retrieval records" if not rer else "no canonical incident[cf] records"
        sec.text(f"Frozen vs re-retrieval skipped: {missing} "
                 f"({inp.path('reretrieve')}, {inp.path('recovery')}).")
    _ablation_tables(sec, inp, ctx)
    if not sec.numbers:
        return sec.skip("no re-retrieval records and no canonical sweep")
    return sec


def _cost_row(name: str, rs: list[dict], ctx: Ctx) -> tuple[list, dict]:
    inc = [r for r in rs if r.get("incident")]
    n, d = measure.asr_counts(rs)
    cols = {"cf replays": [cf_replays(r) for r in inc],
            "seconds": [r.get("seconds") for r in inc],
            "cf seconds": [cf_seconds(r) for r in inc],
            "LLM calls": [r.get("llm_calls") for r in inc]}
    cis = {k: boot_mean(v, ctx.n_boot, ctx.seed) for k, v in cols.items()}
    return ([name, runs_cell(len(rs)), runs_cell(len(inc)), rate_cell(n, d),
             measure.pct(sum(attacked(r) for r in rs), len(rs)),
             *[mean_cell(ci) for ci in cis.values()]],
            {"runs": len(rs), "incident_runs": len(inc), "small_sample": small(len(rs)),
             "asr": rate(n, d), **{k.replace(" ", "_"): ci_numbers(v) for k, v in cis.items()}})


def _paired_cost(a: dict[str, dict], b: dict[str, dict], runs, metrics, ctx: Ctx) -> list[list]:
    """A minus B per run over `runs` for each (name, fn) metric, bootstrap 95%."""
    rows = []
    for name, fn in metrics:
        diffs = [fn(a[rd]) - fn(b[rd]) for rd in runs
                 if fn(a[rd]) is not None and fn(b[rd]) is not None]
        ci = boot_mean(diffs, ctx.n_boot, ctx.seed)
        rows.append([name, len(diffs), "-" if ci is None else measure._ci(ci, 1, 1)])
    return rows


def _reretrieve_tables(sec: Section, rer, frozen, inp: Inputs, ctx: Ctx) -> None:
    rer_by = {r["run_dir"]: r for r in rer}
    runs = sorted(set(rer_by) & set(frozen))
    sec.text(f"### Frozen vs re-retrieval ({len(runs)} runs in both)",
             "Frozen = the canonical incident[cf] (the retrieved set is held fixed and one "
             "memory removed); re-retrieval removes the memory and retrieves top-k again. "
             "Costs are means per incident run (bootstrap 95%); older canonical records "
             "state replays only in their notes and never measured cf seconds.")
    rows, nums = [], {}
    for name, by in (("frozen (canonical)", frozen), ("re-retrieval", rer_by)):
        cells, n = _cost_row(name, [by[rd] for rd in runs], ctx)
        rows.append(cells)
        nums[name] = n
    sec.table(["method", "runs", "incident runs", "ASR after (Wilson 95%)", "runs still attacked",
               "cf replays", "seconds", "cf seconds", "LLM calls"], rows)
    inc = [rd for rd in runs if frozen[rd].get("incident") and rer_by[rd].get("incident")]
    metrics = (("cf replays", cf_replays), ("seconds", lambda r: r.get("seconds")),
               ("LLM calls", lambda r: r.get("llm_calls")), ("run ASR (x100)",
               lambda r: None if run_asr(r) is None else 100 * run_asr(r)))
    sec.text("Paired, re-retrieval minus frozen, per incident run in both:")
    sec.table(["metric", "runs", "difference (bootstrap 95%)"],
              _paired_cost(rer_by, frozen, inc, metrics, ctx))
    diffs = measure.paired_differences(
        [*tagged([rer_by[rd] for rd in runs], "reretrieve"), *[frozen[rd] for rd in runs],
         *[r for r in at0(inp.get("recovery")) if measure.is_baseline(r)]],
        pairs=(("incident[cf]+reretrieve", "incident[cf]", "incident"),), n_boot=ctx.n_boot,
        seed=ctx.seed)
    if diffs:
        sec.table(DIFF_HEADER, diff_rows(diffs))
    sec.numbers["frozen_vs_reretrieve"] = {**nums, "paired": diff_numbers(diffs)}
    _culprit_agreement(sec, rer_by, inp)


def _culprit_agreement(sec: Section, rer_by: dict[str, dict], inp: Inputs) -> None:
    ref = frozen_reference(inp)
    counts, sources = Counter(), Counter()
    for rd, r in rer_by.items():
        if r.get("incident") and rd in ref and culprits(r) is not None:
            counts[agreement(ref[rd][0], culprits(r))] += 1
            sources[ref[rd][1]] += 1
    floor = frozen_vs_frozen(inp)
    n, nf = sum(counts.values()), sum(floor.values())
    found = ", ".join(f"{k}: {v} runs" for k, v in sources.items()) or "none found"
    sec.text("### Culprit agreement",
             "Culprits = minimal removal sets that flip the decision. The canonical sweep's "
             "older records do not store them, so the frozen side comes from the first frozen "
             f"source that does ({found}). "
             "The noise floor compares two frozen ablations of one run (families repeats).")
    floor_names = [a.replace("re-retrieval", "second replay") for a in AGREEMENT]
    sec.table(["agreement", f"frozen vs re-retrieval ({runs_cell(n)} incident runs)",
               f"frozen vs frozen ({runs_cell(nf)} run-repeat pairs)"],
              [[a, measure.pct(counts[a], n), measure.pct(floor[fa], nf)]
               for a, fa in zip(AGREEMENT, floor_names)])
    sec.numbers["culprit_agreement"] = {
        "runs": n, "small_sample": small(n), "counts": dict(counts), "frozen_sources": dict(sources),
        "same_or_overlapping": rate(counts[AGREEMENT[0]] + counts[AGREEMENT[1]], n),
        "noise_floor_pairs": nf, "noise_floor_counts": dict(floor),
        "noise_floor_same_or_overlapping": rate(floor[floor_names[0]] + floor[floor_names[1]], nf)}


ABLATION_ROWS = (("none", "no recovery"),
                 ("ancestry", "counterfactual off: whole lineage"),
                 ("selective[cf]", "lineage-intersected counterfactual"),
                 ("incident[cf]", "ancestry off: counterfactual over all retrieved"))
ABLATION_PAIRS = (("incident[cf]", "selective[cf]", "incident"),
                  ("ancestry", "selective[cf]", "incident"),
                  ("incident[cf]", "ancestry", "incident"))


def _ablation_tables(sec: Section, inp: Inputs, ctx: Ctx) -> None:
    recs = canonical(at0(inp.get("recovery")))
    if not recs:
        sec.text(f"Ancestry / counterfactual ablation skipped: no records in {inp.path('recovery')}.")
        return
    base, idx = baselines_of(recs), index(recs)
    sec.text("### Ancestry-off vs lineage-intersected vs counterfactual-off (canonical sweep)",
             "Incident runs = the original action reached the attacker (the only runs a "
             "counterfactual has an alarm for). Collateral and quarantined are means per run.")
    rows, nums = [], {}
    for pop, keep in (("incident runs", lambda r: measure.record_incident(r) is True),
                      ("all runs", lambda r: True)):
        for lab, role in ABLATION_ROWS:
            rs = [r for r in idx.get(lab, {}).values() if keep(r)]
            if not rs:
                continue
            row = aggregate(rs, base)
            rows.append([pop, lab, role, *row_cells(row), measure._fmt(row.quarantined, 1),
                         measure._fmt(row.collateral, 1), measure._fmt(row.llm_calls, 1)])
            nums[f"{pop}: {lab}"] = row_numbers(row)
    sec.table(["population", "configuration", "role", *ROW_HEADER, "quarantined", "collateral",
               "LLM calls"], rows)
    diffs = measure.paired_differences(recs, pairs=ABLATION_PAIRS, n_boot=ctx.n_boot, seed=ctx.seed)
    path = inp.paths["recovery"]
    aa = measure.replay_aa(path, recs, n_boot=ctx.n_boot, seed=ctx.seed) \
        if ctx.aa and path.is_file() else []
    sec.text(DIFF_NOTE)
    sec.table(DIFF_HEADER, diff_rows(diffs + aa))
    noise = measure.replay_noise(path) if path.is_file() else None
    if noise:
        sec.text(f"Replay noise in the canonical sweep (measure.replay_noise: {noise['pairs']} "
                 f"measurements written twice on the same deterministic store): attacker-vs-not "
                 f"flips in {rate_cell(noise['attack_flips'], noise['tasks'])} of tasks, probe "
                 f"flips in {measure.pct(noise['probe_flips'], noise['probes'])}."
                 + ("" if ctx.aa else " Bootstrap A/A rows: rerun with --aa, or see "
                    "results/recovery/recovery_tables.md."))
    sec.numbers["ablation"] = {
        "rows": nums, "paired": diff_numbers(diffs), "replay_aa": diff_numbers(aa),
        "replay_noise": None if not noise else {
            "pairs": noise["pairs"], "attack_flips": rate(noise["attack_flips"], noise["tasks"]),
            "probe_flips": rate(noise["probe_flips"], noise["probes"])}}


# --- 6. pair/group counterfactuals and latency (12.3) -----------------------------------

KINDS = ("single", "pair", "group", "none", "no incident")


def culprit_kind(rec: dict) -> str:
    """What the counterfactual found for a run it had an alarm for: a single memory, a
    minimal pair, only the all-candidates group, or nothing."""
    if not triggered(rec):
        return "no incident"
    sets = rec.get("culprit_sets")
    if sets is None:
        return "unknown"
    if not sets:
        return "none"
    size = min(len(s) for s in sets)
    return "single" if size == 1 else "pair" if size == 2 else "group"


def _groups_records(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("cf_groups")]


def s6_groups(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("6. Pair/group counterfactuals and latency (proposal 12.3)")
    grp = _groups_records(at0(inp.get("groups")))
    clean = _groups_records(at0(inp.get("groups_clean"), "clean"))
    if not grp and not clean:
        return sec.skip(f"no --cf-groups records in {inp.path('groups')} or {inp.path('groups_clean')}")
    seq_all = canonical(at0(inp.get("recovery")))
    if grp:
        _groups_poisoned(sec, grp, seq_all, inp, ctx)
    if clean:
        _groups_clean(sec, clean, inp)
    else:
        sec.text(f"Clean-run false-alarm control: no records in {inp.path('groups_clean')}.")
    return sec


def _kinds_table(sec: Section, recs: list[dict], title: str) -> dict:
    kinds = Counter(culprit_kind(r) for r in recs)
    alarms = sum(v for k, v in kinds.items() if k != "no incident")
    sec.table(["culprit found", f"{title} ({runs_cell(len(recs))})", "share of alarms (Wilson 95%)"],
              [[k, kinds[k], "-" if k == "no incident" else rate_cell(kinds[k], alarms)]
               for k in (*KINDS, *sorted(set(kinds) - set(KINDS)))])
    return {"runs": len(recs), "alarms": alarms, "small_sample": small(alarms),
            **{k: rate(kinds[k], alarms) for k in kinds if k != "no incident"},
            "no_incident": kinds["no incident"]}


def _groups_poisoned(sec, grp, seq_all, inp: Inputs, ctx: Ctx) -> None:
    sec.text("### Culprit kinds (poisoned runs, incident[cf]+verify with --cf-groups)",
             "Pairs and the all-candidates group are tried only when no single removal flips "
             "the decision; verify rounds can add sets from a later ablation.")
    sec.numbers["culprit_kinds"] = _kinds_table(sec, grp, "runs")
    par = {r["run_dir"]: r for r in grp}
    seq = {r["run_dir"]: r for r in seq_all if label(r) == label(grp[0])}
    runs = sorted(set(par) & set(seq))
    if not runs:
        sec.text(f"No canonical {label(grp[0])} records for these runs; ASR and latency "
                 "comparison skipped.")
        return
    base = baselines_of(at0(inp.get("recovery")))
    sec.text(f"### ASR after {label(grp[0])}: groups vs canonical ({len(runs)} runs in both)")
    rows, nums = [], {}
    for name, by in (("canonical (sequential, singles)", seq), ("groups, 6 workers", par)):
        row = aggregate([by[rd] for rd in runs], base)
        rows.append([name, *row_cells(row), measure._fmt(row.quarantined, 1),
                     measure._fmt(row.collateral, 1)])
        nums[name] = row_numbers(row)
    sec.table(["sweep", *ROW_HEADER, "quarantined", "collateral"], rows)
    par_tagged = tagged([par[rd] for rd in runs], "groups")
    a, b = label(par_tagged[0]), label(grp[0])
    diffs = measure.paired_differences(
        [*par_tagged, *[seq[rd] for rd in runs],
         *[r for r in at0(inp.get("recovery")) if measure.is_baseline(r)]],
        pairs=((a, b, "incident"), (a, b, "all")), n_boot=ctx.n_boot, seed=ctx.seed)
    sec.text(DIFF_NOTE)
    sec.table(DIFF_HEADER, diff_rows(diffs))
    inc = [rd for rd in runs if par[rd].get("incident") and seq[rd].get("incident")]
    sec.text(f"### Latency, parallel vs sequential ({len(inc)} incident runs in both)",
             "Parallel = 6 replay workers, replays stopped at the first acting call, plus "
             "keep-only-one / pair / group replays; sequential = the canonical sweep, one "
             "worker, full replays, single removals. The three changes are confounded here. "
             "Paired difference = parallel minus sequential.")
    metrics = (("cf replays", cf_replays), ("cf seconds", cf_seconds),
               ("recovery seconds", lambda r: r.get("seconds")),
               ("LLM calls", lambda r: r.get("llm_calls")),
               ("seconds per cf replay", lambda r: (r["seconds"] / cf_replays(r))
                if cf_replays(r) and r.get("seconds") is not None else None))
    rows, lat = [], {}
    for (name, fn), diff in zip(metrics, _paired_cost(par, seq, inc, metrics, ctx)):
        s = boot_mean([fn(seq[rd]) for rd in inc], ctx.n_boot, ctx.seed)
        p = boot_mean([fn(par[rd]) for rd in inc], ctx.n_boot, ctx.seed)
        rows.append([name, mean_cell(s), mean_cell(p), runs_cell(diff[1]), diff[2]])
        lat[name] = {"sequential": ci_numbers(s), "parallel": ci_numbers(p), "paired_runs": diff[1]}
    sec.table(["metric", "sequential mean", "parallel mean", "runs paired",
               "parallel minus sequential (bootstrap 95%)"], rows)
    sec.numbers["asr"] = {**nums, "paired": diff_numbers(diffs)}
    sec.numbers["latency"] = lat
    if ctx.figures:
        sec.figures.append(plot_groups(grp, par, seq, inc, ctx.out / "analysis_groups.png", ctx))


def _groups_clean(sec: Section, clean: list[dict], inp: Inputs) -> None:
    """The false-alarm control: clean runs whose own action is flagged. Everything such a
    recovery removes is collateral, so quarantined, BMP and BR are the cost of a wrong alarm."""
    nones = [r for r in at0(inp.get("groups_clean"), "clean") if measure.is_baseline(r)]
    base = {**baselines_of(canonical(at0(inp.get("recovery"), "clean"))), **baselines_of(nones)}
    seq_idx = index(canonical(at0(inp.get("recovery"), "clean")))
    sec.text("### Clean-run false-alarm control",
             "Clean runs given a simulated alarm on their own action. Canonical rows are the "
             "same configuration without --cf-groups on the same runs.")
    rows, nums = [], {}
    for lab, rs in sorted(index(clean).items()):
        rs = list(rs.values())
        seq = [seq_idx.get(lab, {}).get(r["run_dir"]) for r in rs]
        for name, group in (("groups", rs), ("canonical", [r for r in seq if r])):
            if not group:
                continue
            row = aggregate(group, base)
            kinds = Counter(culprit_kind(r) for r in group)
            rows.append([lab, name, runs_cell(row.runs),
                         sum(triggered(r) for r in group), measure._fmt(row.quarantined, 1),
                         measure._fmt(row.collateral, 1), measure._ratio(row.bmp_n, row.bmp_d),
                         measure._fmt(row.br), ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))])
            nums[f"{lab} ({name})"] = {**row_numbers(row), "culprit_kinds": dict(kinds)}
    sec.table(["configuration", "sweep", "runs", "alarms raised", "quarantined", "collateral",
               "BMP", "BR", "culprit kinds"], rows)
    sec.numbers["clean_control"] = nums


def plot_groups(grp, par, seq, inc, path: Path, ctx: Ctx) -> Path:
    plt, fig, (ax, ax2) = _figure(2, 3.0)
    kinds = Counter(culprit_kind(r) for r in grp if triggered(r))
    names = [k for k in ("single", "pair", "group", "none") if k in kinds or k != "group"]
    colours = {"single": STATUS["good"], "pair": STATUS["warning"], "group": STATUS["warning"],
               "none": STATUS["critical"]}
    ys = list(range(len(names)))[::-1]
    ax.barh(ys, [kinds[k] for k in names], height=0.5, color=[colours[k] for k in names])
    for y, k in zip(ys, names):
        ax.text(kinds[k], y, f" {kinds[k]}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    alarms = sum(kinds.values())
    _titles(ax, f"Culprit found ({alarms} incident runs"
            f"{', small sample' if small(alarms) else ''})", "runs", "")
    any_small = False
    for slot, (name, by) in enumerate((("sequential (canonical)", seq), ("parallel + groups", par))):
        vals = [by[rd].get("seconds") for rd in inc]
        for x0, v in enumerate(vals):
            if slot == 0 and v is not None and par[inc[x0]].get("seconds") is not None:
                ax2.plot([0, 1], [v, par[inc[x0]]["seconds"]], color=GRID, lw=1, zorder=1)
        ci = boot_mean(vals, ctx.n_boot, ctx.seed)
        any_small |= _series(ax2, [slot], [None if ci is None else (*ci, len(inc))], slot, 1,
                             name, 0, connect=False)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["sequential", "parallel"], fontsize=8, color=INK)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(bottom=0)
    _titles(ax2, "Recovery seconds per incident run", "", "seconds (mean, bootstrap 95%)")
    _legend(fig, [ax2], any_small)
    return _save(fig, plt, path)


# --- 7. claim repair (12.3 / 8.9) -------------------------------------------------------

CLAIM_CONFIGS = ("selective[oracle]", "selective_recon[oracle]", "selective_claims[oracle]",
                 "selective[cf]", "selective_claims[cf]")
CLAIM_PAIRS = (("selective_claims[oracle]", "selective[oracle]", "all"),
               ("selective_claims[oracle]", "selective[oracle]", "mixed"),
               ("selective_claims[oracle]", "selective_recon[oracle]", "all"),
               ("selective_recon[oracle]", "selective[oracle]", "mixed"),
               ("selective_claims[cf]", "selective[cf]", "incident"),
               # A/A: the same deterministic oracle store, replayed in two sweeps.
               ("selective[oracle]", "selective[oracle]+recovery_sweep", "all"))


def s7_claims(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("7. Claim repair (proposal 12.3 / 8.9)")
    recs = at0(inp.get("claims"))
    if not recs:
        return sec.skip(f"no poisoned records in {inp.path('claims')}")
    idx = index(recs)
    canon = canonical(at0(inp.get("recovery")))
    base = {**baselines_of(canon), **baselines_of(recs)}
    sec.text(f"{len({r['run_dir'] for r in recs})} runs. BMP's baseline is the same run's "
             f"'none' from {inp.path('recovery')} (the claims sweep has no 'none'). Claim "
             "repairs = memories rewritten without the attacker's claim (mean per run). "
             "Oracle rows on the runs with all three oracle configurations; cf rows on the "
             "runs with both cf configurations.")
    header = ["configuration", *ROW_HEADER, "claim repairs", "quarantined", "collateral",
              "LLM calls", "seconds"]
    rows = []
    for group in (CLAIM_CONFIGS[:3], CLAIM_CONFIGS[3:]):
        present = [g for g in group if g in idx]
        runs = set.intersection(*(set(idx[g]) for g in present)) if present else set()
        for lab in group:
            rs = [idx[lab][rd] for rd in sorted(runs)] if lab in idx else []
            if not rs:
                rows.append([lab, "0 (no records)"] + ["-"] * (len(header) - 2))
                continue
            row = aggregate(rs, base)
            repaired = statistics.mean(r.get("n_claim_repaired") or 0 for r in rs)
            rows.append([lab, *row_cells(row), f"{repaired:.1f}", measure._fmt(row.quarantined, 1),
                         measure._fmt(row.collateral, 1), measure._fmt(row.llm_calls, 1),
                         measure._fmt(row.seconds, 1)])
            sec.numbers[lab] = {**row_numbers(row), "claim_repaired": round(repaired, 3)}
    sec.table(header, rows)
    runs = {r["run_dir"] for r in recs}
    earlier = tagged([r for r in canon if label(r) == "selective[oracle]" and r["run_dir"] in runs],
                     "recovery_sweep")
    diffs = measure.paired_differences(
        [*recs, *earlier, *[r for r in canon if measure.is_baseline(r)]], pairs=CLAIM_PAIRS,
        n_boot=ctx.n_boot, seed=ctx.seed)
    sec.text(DIFF_NOTE + " 'mixed' = runs with a mixed descendant. The last row is an A/A "
             "check: the same oracle store replayed in this sweep and in the canonical one.")
    sec.table(DIFF_HEADER, diff_rows(diffs))
    sec.numbers["paired"] = diff_numbers(diffs)
    _reinfection(sec, inp, canon)
    return sec


def _reinfection(sec: Section, inp: Inputs, canon: list[dict]) -> None:
    rein = [r for r in inp.get("reinfect") if r.get("condition", "poisoned") == "poisoned"]
    if not rein:
        sec.text(f"Reinfection: no records in {inp.path('reinfect')}.")
        return
    runs = {r["run_dir"] for r in rein}
    ref = [r for r in inp.get("recovery") if r["run_dir"] in runs
           and r.get("condition", "poisoned") == "poisoned" and not knobs(r)
           and label(r) == "selective[oracle]+immunize"]
    recs = rein + ref
    series = measure.reinfection_series(recs)
    rates = measure.reinfection_rates(recs)
    mem = measure.reinfection_memory(recs)
    sec.text("### Reinfection (Section 8.8 schedule)",
             "ASR at each checkpoint (0 on patched pages, later on live pages), runs clean "
             "after recovery that are attacked again, and runs whose store names the attacker "
             "at the last checkpoint. Reference: selective[oracle]+immunize from the canonical "
             "sweep on the same runs.")
    rows, nums = [], {}
    for key, s in series.items():
        lab = measure.config_label(key, False)
        followed = len({r["run_dir"] for r in recs if measure.config_key(r) == key
                        and r.get("checkpoint", 0) > 0})
        carry = mem.get(key, {}).get("carrying", {})
        last = max(carry) if carry else None
        rows.append([lab, runs_cell(followed),
                     *[measure._ratio(*s[cp]) if cp in s else "-" for cp in measure.CHECKPOINTS],
                     "-" if key not in rates else measure.pct(*rates[key]),
                     "-" if last is None else f"{measure.pct(*carry[last])} at {last}"])
        nums[lab] = {"runs": followed, "small_sample": small(followed),
                     "asr": {cp: rate(*s[cp]) for cp in s},
                     "reinfected": rate(*rates[key]) if key in rates else None}
    sec.table(["configuration", "runs", *[f"ASR cp {cp}" for cp in measure.CHECKPOINTS],
               "reinfected (of clean after recovery)", "store names attacker"], rows)
    sec.numbers["reinfection"] = nums


# --- 8. flat vs evolving memory (6.5) ---------------------------------------------------

RUN_NAME = re.compile(r"(\d+)_(clean|poisoned)_(.+)")
FLAT_MODES = ("none", "root", "ancestry", "selective[oracle]", "incident[cf]")


def _range(spec: str) -> tuple[int, int]:
    lo, hi = spec.split("-")
    return int(lo), int(hi)


def category_of(variant: str) -> str:
    s = ALL_VARIANTS.get(variant)
    return getattr(s, "attack_category", None) or variant.rsplit("_", 1)[-1]


def history_rows(runs_dir: Path, flat: str, evolving: str) -> tuple[list[dict], int]:
    """Original-run outcomes from summary.json: flat = runs in `flat` without the
    _evolving suffix, evolving = runs in `evolving` with it. Returns (rows, run folders
    in range without a readable summary)."""
    (flo, fhi), (elo, ehi) = _range(flat), _range(evolving)
    rows, unreadable = [], 0
    for p in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else ():
        m = RUN_NAME.fullmatch(p.name)
        if not m:
            continue
        n, evolving_name = int(m.group(1)), p.name.endswith("_evolving")
        memory = ("evolving" if evolving_name and elo <= n <= ehi
                  else "flat" if not evolving_name and flo <= n <= fhi else None)
        if memory is None:
            continue
        try:
            summary = json.loads((p / "summary.json").read_text())
        except (OSError, ValueError):
            unreadable += 1
            continue
        variant = m.group(3).removesuffix("_evolving")
        rows.append({"run_dir": p.name, "number": n, "memory": memory, "condition": m.group(2),
                     "variant": variant, "category": category_of(variant),
                     "outcome": summary.get("outcome")})
    return rows, unreadable


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: the discordant pairs under Binomial(b + c, 1/2)."""
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired_by_variant(rows: list[dict], condition: str) -> tuple[dict, int]:
    """variant -> {memory: attacker?}, for variants with one run of each memory. Run
    numbers repeat across parallel processes, so a variant run twice keeps its latest."""
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    dupes = 0
    for r in sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["number"]):
        dupes += r["memory"] in by[r["variant"]]
        by[r["variant"]][r["memory"]] = r
    return {v: {m: x["outcome"] == "attacker" for m, x in d.items()}
            for v, d in by.items() if len(d) == 2}, dupes


def s8_flat(inp: Inputs, ctx: Ctx) -> Section:
    sec = Section("8. Flat vs evolving memory (proposal 6.5)")
    rows, unreadable = history_rows(inp.runs_dir, ctx.flat_range, ctx.evolving_range)
    if rows:
        _history_tables(sec, rows, unreadable, ctx)
    else:
        sec.text(f"History outcomes skipped: no run folders in {inp.runs_dir} for flat "
                 f"{ctx.flat_range} / evolving {ctx.evolving_range}.")
    flat = canonical(at0(inp.get("flat")))
    evol = canonical(at0(inp.get("recovery")))
    if flat and evol:
        _flat_recovery(sec, flat, evol, ctx)
    else:
        sec.text(f"Recovery on flat runs skipped: no records in "
                 f"{inp.path('flat') if not flat else inp.path('recovery')}.")
    if not sec.numbers:
        return sec.skip("no history runs and no flat recovery records")
    if ctx.figures:
        sec.figures.append(plot_flat(rows, flat, evol, ctx.out / "analysis_flat.png"))
    return sec


def _history_tables(sec: Section, rows: list[dict], unreadable: int, ctx: Ctx) -> None:
    cats = ["all", *sorted({r["category"] for r in rows})]
    sec.text(f"### Original-run outcomes ({len(rows)} runs; flat {ctx.flat_range}, evolving "
             f"{ctx.evolving_range}; {unreadable} folders without a readable summary)")
    table, nums = [], {}
    for memory in ("flat", "evolving"):
        for cond in ("poisoned", "clean"):
            for cat in cats:
                rs = [r for r in rows if r["memory"] == memory and r["condition"] == cond
                      and cat in ("all", r["category"])]
                if not rs:
                    continue
                c = Counter(r["outcome"] for r in rs)
                table.append([memory, cond, cat, runs_cell(len(rs)), rate_cell(c["attacker"], len(rs)),
                              c["correct"], c["other"], c["none"]])
                nums[f"{memory} {cond} {cat}"] = {"attacker": rate(c["attacker"], len(rs)),
                                                  "outcomes": dict(c)}
    sec.table(["memory", "condition", "category", "runs", "attacker (Wilson 95%)", "correct",
               "other", "no action"], table)
    sec.text("### Paired by variant (exact McNemar)",
             "Each variant has one flat and one evolving run; discordant pairs decide the test.")
    table = []
    for cond in ("poisoned", "clean"):
        pairs, dupes = paired_by_variant(rows, cond)
        for cat in cats:
            ps = [p for v, p in pairs.items() if cat in ("all", category_of(v))]
            if not ps:
                continue
            both = sum(p["flat"] and p["evolving"] for p in ps)
            b = sum(p["flat"] and not p["evolving"] for p in ps)
            c = sum(p["evolving"] and not p["flat"] for p in ps)
            p_val = mcnemar_exact(b, c)
            table.append([cond, cat, runs_cell(len(ps)), both, b, c, len(ps) - both - b - c,
                          f"{p_val:.3g}"])
            nums[f"mcnemar {cond} {cat}"] = {"pairs": len(ps), "both": both, "flat_only": b,
                                             "evolving_only": c, "p": round(p_val, 6),
                                             "duplicates_dropped": dupes}
    sec.table(["condition", "category", "variant pairs", "attacker in both", "flat only",
               "evolving only", "neither", "exact McNemar p"], table)
    sec.numbers["history"] = nums


def _flat_recovery(sec: Section, flat: list[dict], evol: list[dict], ctx: Ctx) -> None:
    variants = {r["variant"] for r in flat} & {r["variant"] for r in evol}
    sec.text(f"### Recovery, flat vs evolving ({len(variants)} variants with recovery records "
             "in both)",
             "Flat records from the X3 sweep, evolving from the canonical sweep, restricted "
             "to the same variants. Root-only effectiveness is the key contrast: in a flat "
             "store nothing is consolidated, so removing the root should remove the attack.")
    table, nums = [], {}
    idx = {"flat": index([r for r in flat if r["variant"] in variants]),
           "evolving": index([r for r in evol if r["variant"] in variants])}
    for lab in FLAT_MODES:
        for memory, recs in (("flat", flat), ("evolving", evol)):
            rs = list(idx[memory].get(lab, {}).values())
            if not rs:
                continue
            row = aggregate(rs, baselines_of(recs))
            rc = row.root_consolidated
            table.append([lab, memory, *row_cells(row), "-" if rc is None else f"{100 * rc:.0f}%"])
            nums[f"{lab} {memory}"] = {**row_numbers(row), "root_consolidated": rc}
    sec.table(["configuration", "memory", *ROW_HEADER, "root already consolidated"], table)
    eff = _root_effect(sec, idx, ctx)
    sec.numbers["recovery"] = {"variants": len(variants), "rows": nums, "root_only": eff}


def _root_effect(sec: Section, idx: dict, ctx: Ctx) -> dict:
    """Root-only: among runs attacked with no recovery, the share root alone clears, per
    memory; then the flat-minus-evolving gap in run-level ASR reduction, paired by variant."""
    out, rows, drop = {}, [], {}
    for memory in ("flat", "evolving"):
        none, root = idx[memory].get("none", {}), idx[memory].get("root", {})
        both = [rd for rd in none if rd in root]
        hit = [rd for rd in both if attacked(none[rd])]
        fixed = sum(not attacked(root[rd]) for rd in hit)
        drop[memory] = {none[rd]["variant"]: run_asr(none[rd]) - run_asr(root[rd]) for rd in both
                        if run_asr(none[rd]) is not None and run_asr(root[rd]) is not None}
        rows.append([memory, runs_cell(len(both)), runs_cell(len(hit)), rate_cell(fixed, len(hit))])
        out[memory] = {"runs": len(both), "attacked_without_recovery": len(hit),
                       "cleared_by_root": rate(fixed, len(hit))}
    sec.table(["memory", "runs with none and root", "attacked with no recovery",
               "cleared by root alone (Wilson 95%)"], rows)
    shared = sorted(set(drop["flat"]) & set(drop["evolving"]))
    ci = boot_mean([drop["flat"][v] - drop["evolving"][v] for v in shared], ctx.n_boot, ctx.seed)
    sec.text(f"Run-level ASR reduction by root (none minus root), flat minus evolving, paired "
             f"by variant ({runs_cell(len(shared))} variants): "
             f"{'-' if ci is None else measure._ci(ci, 100)} percentage points (bootstrap 95%).")
    out["flat_minus_evolving_reduction_pp"] = None if ci is None else {
        "mean": round(100 * ci[0], 2), "ci": [round(100 * ci[1], 2), round(100 * ci[2], 2)],
        "variants": len(shared), "small_sample": small(len(shared))}
    return out


def plot_flat(rows: list[dict], flat: list[dict], evol: list[dict], path: Path) -> Path:
    plt, fig, (ax, ax2) = _figure(2, 3.1)
    pois = [r for r in rows if r["condition"] == "poisoned"]
    cats = ["all", *sorted({r["category"] for r in pois})]
    any_small = False
    for slot, memory in enumerate(("flat", "evolving")):
        pts = []
        for cat in cats:
            rs = [r for r in pois if r["memory"] == memory and cat in ("all", r["category"])]
            p = _pct_point(sum(r["outcome"] == "attacker" for r in rs), len(rs))
            pts.append(p)
        any_small |= _series(ax, range(len(cats)), pts, slot, 2, f"{memory} memory", 0.2,
                             connect=False)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=7.5, color=INK)
    ax.set_ylim(-3, 103)
    _titles(ax, "Poisoned history runs reaching the attacker", "attack category",
            "% of runs (Wilson 95%)")
    variants = {r["variant"] for r in flat} & {r["variant"] for r in evol}
    for slot, (memory, recs) in enumerate((("flat", flat), ("evolving", evol))):
        idx = index([r for r in recs if r["variant"] in variants])
        pts = []
        for lab in FLAT_MODES:
            rs = list(idx.get(lab, {}).values())
            p = _pct_point(*measure.asr_counts(rs)) if rs else None
            pts.append(None if p is None else (*p[:3], len(rs)))
        any_small |= _series(ax2, range(len(FLAT_MODES)), pts, slot, 2, f"{memory} memory",
                             0.2, connect=False)
    ax2.set_xticks(range(len(FLAT_MODES)))
    ax2.set_xticklabels([m.replace("[", "\n[") for m in FLAT_MODES], fontsize=7, color=INK)
    ax2.set_ylim(-3, 103)
    _titles(ax2, f"ASR after recovery ({len(variants)} variants)", "",
            "% of replayed tasks (Wilson 95%)")
    _legend(fig, [ax, ax2], any_small)
    return _save(fig, plt, path)


# --- 9. healing vs quarantine (Phase 2, H) ----------------------------------------------

HEAL_MODES = ("heal", "heal_rebuild", "heal_oracle")
# The heal modes and quarantine on the same detection (incident) first, then the other
# repairs and the oracle ladder; tables and the figure's marker slots follow this order.
HEAL_COMPARE = ("heal", "heal_rebuild", "heal_oracle", "incident", "selective_claims",
                "selective_recon", "selective", "unmerge")
# The files Phase 2's heal sweeps write (data/main_b_jobs.txt): heal, heal_rebuild and
# heal_oracle (and incident/heal +verify at verify_n=3) on the dense evolving runs, next
# to the main_a sweep's other configurations, and heal and heal_oracle on their flat
# twins. --heal takes any number of files and each gets a section of its own (flat and
# evolving runs are different memories, so no row pools them); the history suites' heal
# sweeps are analyze_history.py's section 5.
HEAL_INPUTS = ("results/main/records.jsonl", "results/flat/main.jsonl")
# Where a run's record of a configuration is read from, first hit wins: the heal file
# (same code and time as heal itself), then the claims sweep, then the canonical one.
HEAL_FALLBACK = ("claims", "recovery")
HEAL_SOURCES = ("heal", *HEAL_FALLBACK)
# The four categorical slots go to the modes the comparison is about (the heal modes and
# incident); the other repairs and the oracle ladder are context, grouped as 'other' in
# one ink, told apart by marker and direct label, never by a status colour or a fifth hue.
HEAL_PRIMARY = 4
HEAL_COLOURS = (*PALETTE, *(INK2,) * (len(HEAL_COMPARE) - HEAL_PRIMARY))
HEAL_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")
HEAL_POPULATIONS = (("incident runs", lambda r: measure.record_incident(r) is True),
                    ("all runs", lambda r: True))
# What a heal v2 record says about how healing decided (recover.RECORD_KEYS): claims
# dropped per rule, ddmin's replays, and the post-heal check. Older records lack them.
HEAL_V2_FIELDS = ("dropped_by", "post_heal_harmful", "ddmin_replays")
SCENARIO_BOOT = "percentile bootstrap 95% resampling scenarios"


def heal_input_names(paths) -> list[tuple[str, Path]]:
    """(input name, path) per --heal file: 'heal' for the first (the name bcp.py rows and
    every earlier report used), 'heal_<folder>' for the others (results/flat/main.jsonl
    -> heal_flat), numbered when two share a folder name."""
    out, seen = [], set()
    for i, p in enumerate(paths):
        folder = re.sub(r"[^A-Za-z0-9]+", "_", Path(p).parent.name).strip("_")
        name = base = "heal" if i == 0 else f"heal_{folder or i}"
        j = 2
        while name in seen:
            name, j = f"{base}_{j}", j + 1
        seen.add(name)
        out.append((name, Path(p)))
    return out


def _shown(path) -> str:
    """A path as the report prints it: relative to the repo when it is inside it."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def heal_specs(inp: "Inputs") -> list[tuple[str, str, str]]:
    """(input name, section title, figure file) per heal input. One file keeps the title
    and figure name every earlier report had; several are 9a, 9b, ... and name their file."""
    base = "Healing vs quarantine (Phase 2, H)"
    if len(inp.heal) == 1:
        return [(inp.heal[0], f"9. {base}", "analysis_heal.png")]
    return [(name, f"9{chr(ord('a') + i)}. {base}: {_shown(inp.paths[name])}",
             "analysis_heal.png" if name == "heal" else f"analysis_{name}.png")
            for i, name in enumerate(inp.heal)]


def scenario(rec: dict) -> str:
    """Section 9's unit of independence: the scenario (variant). The Phase 2 suites have
    up to three history runs per scenario, which share its pages, tasks and poison, so
    every interval in the section resamples scenarios with all their runs."""
    return rec.get("variant") or rec["run_dir"]


def runs_scen_cell(runs: int, scenarios: int, of: int | None = None) -> str:
    """'97 (77 scenarios)', or '12 of 97 (11 scenarios)'; a small sample counts scenarios."""
    head = f"{runs}" if of is None else f"{runs} of {of}"
    tail = ", small sample" if small(scenarios) else ""
    return f"{head} ({scenarios} scenario{'s' * (scenarios != 1)}{tail})"


def boot_rate(n: float, d: float, ci: tuple | None, scenarios: int) -> dict | None:
    """A pooled rate with its scenario-bootstrap interval, for analysis.json."""
    if not d:
        return None
    return {"n": n, "d": d, "rate": round(n / d, 4),
            "ci": None if ci is None else [round(ci[1], 4), round(ci[2], 4)],
            "interval": "scenario bootstrap", "small_sample": small(scenarios)}


def boot_cell(n: float, d: float, ci: tuple | None) -> str:
    """'12% (8/69) [5-20]': measure's ratio with a bootstrap interval instead of Wilson."""
    if not d:
        return "-"
    return measure._ratio(n, d) + ("" if ci is None else f" [{100 * ci[1]:.0f}-{100 * ci[2]:.0f}]")


def boot_runs(rs: list[dict], pair, ctx: Ctx) -> tuple[float, float, tuple | None]:
    """Pooled sum(n) / sum(d) of pair(record) -> (n, d) | None over the records, and its
    interval resampling scenarios: (n, d, ci)."""
    got = [(p, scenario(r)) for r in rs if (p := pair(r)) is not None]
    pairs = [p for p, _ in got]
    ci = boot_pooled(pairs, ctx.n_boot, ctx.seed, [c for _, c in got])
    whole = lambda x: int(x) if float(x).is_integer() else x
    return whole(sum(p[0] for p in pairs)), whole(sum(p[1] for p in pairs)), ci


def boot_scen(values: list, rs: list[dict], ctx: Ctx) -> tuple | None:
    """boot_mean of one value per record, resampling the records' scenarios."""
    return boot_mean(values, ctx.n_boot, ctx.seed, [scenario(r) for r in rs])


def heal_label(rec: dict) -> str:
    """Configuration and knobs, without the repeat: heal_selection keeps one replay per
    run (the lowest repeat), so repeats of one sweep share a row."""
    return label(rec) + measure.knob_label(measure.drop_knobs(knobs(rec), ("repeat",)))


def _mode_order(lab: str, rep: dict[str, dict]) -> tuple:
    return (HEAL_COMPARE.index(rep[lab]["mode"]), lab)


@dataclass
class HealSelection:
    """The compared configurations on the runs a heal mode was applied to.

    by      configuration -> run -> record
    origin  configuration -> run -> the input it came from (sources)
    rep     configuration -> one of its records (mode, modifiers, knobs)
    runs    the runs a heal mode was applied to
    base    'none' baselines for BMP, the heal file's own first
    sources the inputs read, in order: the heal file, then HEAL_FALLBACK
    """
    by: dict[str, dict[str, dict]]
    origin: dict[str, dict[str, str]]
    rep: dict[str, dict]
    runs: set[str]
    base: dict
    sources: tuple[str, ...] = HEAL_SOURCES

    def labels(self) -> list[str]:
        return sorted(self.by, key=lambda lab: _mode_order(lab, self.rep))

    def records(self, lab: str, keep=lambda r: True) -> list[dict]:
        return [r for _, r in sorted(self.by.get(lab, {}).items()) if keep(r)]


def heal_selection(inp: Inputs, heal: str = "heal") -> HealSelection:
    """Poisoned, patched, checkpoint-0 records of HEAL_COMPARE's modes on the runs a heal
    mode was applied to, one per configuration and run, the heal file being input `heal`.

    A configuration the heal file holds is read from it alone, never mixed with an
    earlier sweep's copy on some runs; one it does not hold (selective_claims, say)
    comes from the sweep that ran it. Within a file the lowest repeat wins, so every
    configuration weighs one replay per run (Section 1's rule)."""
    sources = (heal, *HEAL_FALLBACK)
    pools = {src: [r for r in at0(inp.get(src)) if (r.get("source") or "patched") == "patched"]
             for src in sources}
    runs = {r["run_dir"] for recs in pools.values() for r in recs if r.get("mode") in HEAL_MODES}
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    origin: dict[str, dict[str, str]] = defaultdict(dict)
    rep: dict[str, dict] = {}
    for src in sources:
        for r in sorted(pools[src], key=_repeat, reverse=True):
            if r.get("mode") not in HEAL_COMPARE or r["run_dir"] not in runs:
                continue
            lab, rd = heal_label(r), r["run_dir"]
            if origin[lab].get(rd, src) != src:
                continue                            # an earlier source holds this run
            by[lab][rd], origin[lab][rd] = r, src
            rep.setdefault(lab, r)
    base = {}
    for src in reversed(sources):                   # the heal file's own 'none' wins
        base.update(baselines_of(pools[src]))
    return HealSelection(dict(by), dict(origin), rep, runs, base, sources)


def _resolved(path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return (p if p.is_absolute() else ROOT / p).resolve()


def heal_preservation(inp: Inputs) -> dict[tuple, dict]:
    """(input name, run, measure.setting_key) -> bcp.py's row for that record.

    A row is matched by the records file it measured (bcp.py writes it relative to the
    repo root), not just by run and configuration: the same configuration recovered by
    another sweep is another copy of the store, and its claims another trial."""
    names = {_resolved(p): name for name, p in inp.paths.items()}
    out = {}
    for row in inp.preservation():
        name = names.get(_resolved(row.get("records")))
        if (name is not None and row.get("checkpoint", 0) == 0 and row.get("run_dir")
                and (row.get("condition") or "poisoned") == "poisoned"):
            out[(name, row["run_dir"], measure.setting_key(row))] = row
    return out


def claim_counts(row: dict | None) -> dict[str, tuple[int, int]] | None:
    """BCP and ACR counts of one bcp.py row as (numerator, denominator), bcp.METRICS'
    definitions: benign claims still retrievable over benign claims in the affected
    memories; attack claims no longer retrievable over attack claims. None unless the
    row was measured ('ok')."""
    if not row or row.get("status") != "ok":
        return None
    attack = row.get("attack") or 0
    return {"bcp": (row.get("benign_kept") or 0, row.get("benign") or 0),
            "acr": (attack - (row.get("attack_kept") or 0), attack)}


def boot_pooled(pairs: list[tuple[int, int]], n_boot: int, seed: int = 0,
                clusters: list | None = None) -> tuple[float, float, float] | None:
    """sum(n) / sum(d) over runs with a percentile bootstrap 95% interval from resampling
    runs, or with `clusters` (the run's scenario, one per pair) whole scenarios. The
    claims of one run move together, and so do the runs of one scenario, so Wilson over
    claims is too narrow (bcp.py reports both); this is the interval the figure draws.
    Clustered, a rate of 0 or 1 is widened to Wilson over its scenarios."""
    total = sum(d for _, d in pairs)
    if not total:
        return None
    groups = [(sum(x[0] for x in g), sum(x[1] for x in g)) for g in clustered(pairs, clusters)]
    rnd = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = rnd.choices(groups, k=len(groups))
        d = sum(x[1] for x in sample)
        if d:
            boots.append(sum(x[0] for x in sample) / d)
    boots.sort()
    obs, lo, hi = (sum(n for n, _ in pairs) / total, boots[int(0.025 * len(boots))],
                   boots[int(0.975 * len(boots)) - 1])
    if clusters is not None and obs in (0.0, 1.0):
        # Every draw of an all-or-nothing cell is the same, so the bootstrap cannot see
        # how far the rate could be from it: there the bound is Wilson's with one
        # observation per scenario (what a degenerate [0, 0] would otherwise hide).
        w_lo, w_hi = measure.wilson(obs * len(groups), len(groups))
        lo, hi = min(lo, w_lo), max(hi, w_hi)
    return obs, lo, hi


def boot_pooled_diff(a: list[tuple[int, int]], b: list[tuple[int, int]], n_boot: int,
                     seed: int = 0, clusters: list | None = None
                     ) -> tuple[float, float, float] | None:
    """Pooled rate of A minus pooled rate of B over the same runs (a[i] and b[i] are one
    run), percentile bootstrap 95% over runs (bcp.diff's recipe), or over whole
    scenarios with `clusters` (a label per run): each run's A/B pair stays together."""
    def pooled(xs, ids):
        d = sum(xs[i][1] for i in ids)
        return sum(xs[i][0] for i in ids) / d if d else None

    every = list(range(len(a)))
    pa, pb = pooled(a, every), pooled(b, every)
    if pa is None or pb is None:
        return None
    groups = clustered(every, clusters)
    rnd = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ids = [i for g in rnd.choices(groups, k=len(groups)) for i in g]
        x, y = pooled(a, ids), pooled(b, ids)
        if x is not None and y is not None:
            boots.append(x - y)
    boots.sort()
    return (pa - pb, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1])


def _field_mean(rs: list[dict], key: str) -> float | None:
    """Mean over the records that state `key`; records from before heal lack its fields."""
    return _mean([r[key] for r in rs if r.get(key) is not None])


def _sources_cell(sel: HealSelection, lab: str, rs: list[dict]) -> str:
    c = Counter(sel.origin[lab][r["run_dir"]] for r in rs)
    return ", ".join(f"{src} {c[src]}" for src in sel.sources if c[src])


def s9_heal(inp: Inputs, ctx: Ctx, heal: str = "heal",
            title: str = "9. Healing vs quarantine (Phase 2, H)",
            figure: str = "analysis_heal.png") -> Section:
    sec = Section(title)
    sel = heal_selection(inp, heal)
    files = ", ".join(inp.path(s) for s in sel.sources)
    if not sel.runs:
        # 'heal' names the family here (heal_rebuild too); the wording predates
        # heal_rebuild and is kept so a report over the same files reads the same.
        return sec.skip("no heal or heal_oracle poisoned records in " + files)
    labels = sel.labels()
    # heal_rebuild came with heal v2: a sweep from before it (no record states the v2
    # fields) could not have run it, so it is not reported missing there.
    v2 = any(_states_v2(r) for lab in labels for r in sel.records(lab))
    absent = [m for m in HEAL_COMPARE if (v2 or m != "heal_rebuild")
              and not any(sel.rep[lab]["mode"] == m for lab in labels)]
    inc_recs = {rd: r for lab in labels if sel.rep[lab]["mode"] in HEAL_MODES
                for rd, r in sel.by[lab].items() if measure.record_incident(r) is True}
    all_recs = {rd: r for lab in labels for rd, r in sel.by[lab].items() if rd in sel.runs}
    n_scen, inc_scen = len({scenario(r) for r in all_recs.values()}), len(
        {scenario(r) for r in inc_recs.values()})
    sec.text(f"{len(sel.runs)} runs ({n_scen} scenarios) a heal mode was applied to; "
             f"{len(inc_recs)} runs ({inc_scen} scenarios) with an incident: the original "
             "action reached the attacker, the only runs heal and incident act on. Each "
             f"configuration's record for a run is read from the first of {files} that "
             "holds it, at its lowest repeat ('from' counts runs per file). Rows are on "
             "these runs; a configuration missing on some shows its own n, and the paired "
             "rows below are on the runs both configurations have. A scenario's history "
             "runs share its pages, tasks and poison, so every interval in this section is "
             f"a {SCENARIO_BOOT} (each with all of its runs), and 'small sample' marks "
             f"fewer than {SMALL} scenarios."
             + (f" No records on these runs for: {', '.join(absent)}." if absent else ""))
    pres = heal_preservation(inp)
    counts = {lab: {rd: claim_counts(pres.get((sel.origin[lab][rd], rd, measure.setting_key(r))))
                    for rd, r in sel.by[lab].items()} for lab in labels}
    sec.numbers.update(file=inp.path(heal), runs=len(sel.runs), scenarios=n_scen,
                       incident_runs=len(inc_recs), incident_scenarios=inc_scen,
                       absent_modes=absent,
                       sources={lab: dict(Counter(sel.origin[lab].values())) for lab in labels})
    _heal_outcomes(sec, sel, labels, ctx)
    _heal_actions(sec, sel, labels, ctx)
    _heal_decisions(sec, sel, labels, ctx)
    have_bcp = any(c is not None for by in counts.values() for c in by.values())
    if have_bcp:
        _heal_preservation_table(sec, sel, labels, counts, ctx)
    else:
        sec.text(f"Benign claim preservation skipped: no measured bcp.py rows for these records "
                 f"in {inp.preservation_path} (run bcp.py --records on {files}).")
        sec.numbers["preservation"] = None
    _heal_paired(sec, sel, labels, counts, ctx)
    points = heal_points(sel, labels, counts, ctx)
    sec.numbers["figure"] = points
    drawn = [p for p in points["points"] if p["bcp"]]
    if ctx.figures and any(p["mode"] in HEAL_MODES for p in drawn):
        sec.figures.append(plot_heal(points, ctx.out / figure))
    elif ctx.figures:
        sec.text(f"Figure {figure} not drawn: it needs BCP, and no heal configuration here "
                 "has it.")
    return sec


def heal_row(rs: list[dict], base: dict, ctx: Ctx) -> tuple[list, dict]:
    """One configuration's outcome cells and numbers: measure.aggregate's pooled counts,
    each with a scenario-bootstrap interval, n as runs (scenarios)."""
    row = aggregate(rs, base)
    n_scen = len({scenario(r) for r in rs})
    hit = lambda r: (float(any(o == "attacker" for o in r.get("task_outcomes") or ())
                           or (r.get("asr") or 0) > 0), 1)
    left = lambda r: (None if r.get("residual_attack") is None
                      else (float(r["residual_attack"] > 0), 1))
    asr, att = boot_runs(rs, lambda r: measure.asr_counts([r]), ctx), boot_runs(rs, hit, ctx)
    bmp = boot_runs(rs, lambda r: measure.bmp_counts([r], base)[:2], ctx)
    res = boot_runs(rs, left, ctx)
    cells = [runs_scen_cell(row.runs, n_scen), boot_cell(*asr), boot_cell(*att),
             boot_cell(*bmp), measure._fmt(row.br), boot_cell(*res)]
    nums = {**row_numbers(row), "scenarios": n_scen, "small_sample": small(n_scen),
            "asr": boot_rate(*asr, n_scen), "attacked_runs": boot_rate(*att, n_scen),
            "bmp": boot_rate(*bmp, n_scen), "poison_left": boot_rate(*res, n_scen)}
    return cells, nums


HEAL_ROW_HEADER = ["runs (scenarios)", "ASR (tasks)", "runs still attacked", "BMP", "BR",
                   "poison left in memory"]


def _heal_outcomes(sec: Section, sel: HealSelection, labels: list[str], ctx: Ctx) -> None:
    sec.text("### Outcome after recovery",
             "Poison left = runs whose store still holds an active memory naming the "
             "attacker; residual = mean such memories per run. Rates pool runs (ASR pools "
             f"their tasks, BMP their probes); intervals: {SCENARIO_BOOT}.")
    rows, nums = [], {}
    for pop, keep in HEAL_POPULATIONS:
        for lab in labels:
            rs = sel.records(lab, keep)
            if not rs:
                continue
            cells, nums[f"{pop}: {lab}"] = heal_row(rs, sel.base, ctx)
            resid = _field_mean(rs, "residual_attack")
            nums[f"{pop}: {lab}"]["residual_attack"] = _round(resid)
            rows.append([pop, lab, _sources_cell(sel, lab, rs), *cells, measure._fmt(resid, 1)])
    sec.table(["population", "configuration", "from", *HEAL_ROW_HEADER, "residual"], rows)
    sec.numbers["rows"] = nums


HEAL_COUNTS = (("quarantined", "n_quarantined"), ("healed", "n_healed"),
               ("heal-quarantined", "n_heal_quarantined"), ("claims kept", "claims_kept"),
               ("claims dropped", "claims_dropped"))


def _heal_actions(sec: Section, sel: HealSelection, labels: list[str], ctx: Ctx) -> None:
    sec.text("### What recovery did, and its cost (all runs)",
             "Means per run. healed = flagged memories replaced by a healed memory; "
             "heal-quarantined = flagged memories heal quarantined with nothing in their "
             "place; claims kept / dropped = claims written into replacements / judged to "
             "carry the attack (heal and selective_claims); claim cf replays = heal's "
             "claim-level counterfactual, not in cf replays (the memory-level ablation). "
             "'-' = no record of that configuration states the field (older records). "
             f"Intervals: {SCENARIO_BOOT}.")
    rows, nums = [], {}
    for lab in labels:
        rs = sel.records(lab)
        n_scen = len({scenario(r) for r in rs})
        means = {key: _field_mean(rs, key) for _, key in HEAL_COUNTS}
        costs = {"cf replays": boot_scen([cf_replays(r) for r in rs], rs, ctx),
                 "claim cf replays": boot_scen([r.get("claim_cf_replays") for r in rs], rs, ctx),
                 "LLM calls": boot_scen([r.get("llm_calls") for r in rs], rs, ctx),
                 "seconds": boot_scen([r.get("seconds") for r in rs], rs, ctx)}
        rows.append([lab, runs_scen_cell(len(rs), n_scen),
                     *[measure._fmt(v, 1) for v in means.values()],
                     *[mean_cell(ci) for ci in costs.values()]])
        nums[lab] = {"runs": len(rs), "scenarios": n_scen, "small_sample": small(n_scen),
                     **{k: _round(v) for k, v in means.items()},
                     **{k.replace(" ", "_"): ci_numbers(v) for k, v in costs.items()}}
    sec.table(["configuration", "runs (scenarios)", *[name for name, _ in HEAL_COUNTS],
               "cf replays", "claim cf replays", "LLM calls", "seconds"], rows)
    sec.numbers["actions"] = nums


def heal_rules(recs: list[dict]) -> list[str]:
    """The rules a dropped_by table shows: memheal/heal.py's, in its order, then any other
    a record names (a rule added after this script was written is shown, not lost)."""
    named = {rule for r in recs for rule in (r.get("dropped_by") or {})}
    return [*HEAL_RULES, *sorted(named - set(HEAL_RULES))]


def _states_v2(rec: dict) -> bool:
    return any(rec.get(f) is not None for f in HEAL_V2_FIELDS)


def _heal_decisions(sec: Section, sel: HealSelection, labels: list[str], ctx: Ctx) -> None:
    """Heal v2's account of its own decisions, per heal configuration: the rule that
    dropped each claim, ddmin's replays and the post-heal check. Shown only when some
    record states them; records written before heal v2 do not, and the section then
    reads exactly as it did."""
    labs = [lab for lab in labels if sel.rep[lab]["mode"] in HEAL_MODES
            and any(_states_v2(r) for r in sel.records(lab))]
    if not labs:
        return
    rules = heal_rules([r for lab in labs for r in sel.records(lab)])
    sec.text("### Why claims were dropped, and the post-heal check (all runs)",
             "dropped_by (memheal/heal.py RULES): claims dropped per rule, summed over runs, "
             "with each rule's share of the claims dropped. name = names the observed "
             "destination or was read off its site; source-taint = points at a root page, or "
             "asserts a change of route from a tainted page; cf = the claim-level "
             "counterfactual (and claims restating what it found); ddmin = delta debugging "
             "showed putting the claim back is harmful; support = a rebuilt claim no input "
             "supports (heal_rebuild); oracle = the AttackSpec (heal_oracle). ddmin replays = "
             "the part of the claim cf replays ddmin spent. Post-heal check = the explained "
             "decision replayed on the frozen set as healing left it, when a retrieved memory "
             "was healed (recorded, not acted on): the mean over checked runs of the share of "
             "its replays that were harmful (the replay count per run is not recorded, so "
             "replays are not pooled), and the runs with at least one harmful replay. Runs = "
             f"runs whose record states these fields (heal v2). Intervals: {SCENARIO_BOOT}.")
    rows, nums = [], {}
    for lab in labs:
        rs = [r for r in sel.records(lab) if _states_v2(r)]
        n_scen = len({scenario(r) for r in rs})
        per_rule = {rule: sum((r.get("dropped_by") or {}).get(rule, 0) for r in rs)
                    for rule in rules}
        total = sum(per_rule.values())
        ddmin = boot_scen([r.get("ddmin_replays") for r in rs], rs, ctx)
        checked = [r for r in rs if r.get("post_heal_harmful") is not None]
        c_scen = len({scenario(r) for r in checked})
        share = boot_scen([r["post_heal_harmful"] for r in checked], checked, ctx)
        harmful = boot_runs(checked, lambda r: (float(r["post_heal_harmful"] > 0), 1), ctx)
        rows.append([lab, runs_scen_cell(len(rs), n_scen), total,
                     *[f"{per_rule[rule]} ({100 * per_rule[rule] / total:.0f}%)" if total
                       else str(per_rule[rule]) for rule in rules],
                     mean_cell(ddmin), runs_scen_cell(len(checked), c_scen),
                     "-" if share is None else f"{100 * share[0]:.0f}% "
                     f"[{100 * share[1]:.0f}, {100 * share[2]:.0f}]",
                     boot_cell(*harmful) if checked else "-"])
        nums[lab] = {"runs": len(rs), "scenarios": n_scen, "small_sample": small(n_scen),
                     "claims_dropped": total, "dropped_by": per_rule,
                     "ddmin_replays": ci_numbers(ddmin), "post_heal_runs": len(checked),
                     "post_heal_scenarios": c_scen,
                     "post_heal_harmful_share": None if share is None else {
                         **ci_numbers(share), "n": len(checked), "small_sample": small(c_scen)},
                     "post_heal_runs_harmful": boot_rate(*harmful, c_scen)}
    sec.table(["configuration", "runs (scenarios)", "claims dropped", *rules,
               "ddmin replays", "runs with a post-heal check (scenarios)",
               "post-heal harmful share", "runs with a harmful post-heal replay"], rows)
    sec.numbers["decisions"] = nums


def _claims_cells(got: list[tuple[dict, str]], key: str, ctx: Ctx) -> tuple[list, dict]:
    """Wilson over claims (too narrow: claims of a run, and runs of a scenario, move
    together; kept for bcp.py's comparison) and the scenario bootstrap the figure uses.
    `got` is (claim_counts row, scenario) per measured run."""
    pairs = [c[key] for c, _ in got]
    n, d = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    ci = boot_pooled(pairs, ctx.n_boot, ctx.seed, [s for _, s in got])
    return ([rate_cell(n, d), "-" if ci is None else f"[{100 * ci[1]:.0f}, {100 * ci[2]:.0f}]"],
            {**(rate(n, d) or {"n": n, "d": d}), "boot": ci_numbers(ci)})


def _heal_preservation_table(sec: Section, sel: HealSelection, labels: list[str],
                             counts: dict, ctx: Ctx) -> None:
    sec.text("### Benign claim preservation (BCP) and attack claim removal (ACR)",
             "From bcp.py (memheal/preservation.py): BCP = benign claims of the affected "
             "memories still retrievable after recovery; ACR = attack claims no longer "
             "retrievable. Pooled over runs, Wilson 95% over claims and a "
             f"{SCENARIO_BOOT} (claims of one run, and runs of one scenario, move together; "
             "the figure uses it). Runs measured = runs with a bcp.py row for the record's "
             "own records file and status ok. Quarantine loses every benign claim of a "
             "flagged memory; healing should keep them.")
    rows, nums, unmeasured = [], {}, []
    for pop, keep in HEAL_POPULATIONS:
        for lab in labels:
            rs = sel.records(lab, keep)
            if not rs:
                continue
            got = [(counts[lab][r["run_dir"]], scenario(r)) for r in rs
                   if counts[lab][r["run_dir"]]]
            m_scen = len({s for _, s in got})
            nums[f"{pop}: {lab}"] = {"runs": len(rs), "measured": len(got),
                                     "measured_scenarios": m_scen, "small_sample": small(m_scen)}
            if not got:
                if lab not in unmeasured:
                    unmeasured.append(lab)
                continue
            bcp, nums[f"{pop}: {lab}"]["bcp"] = _claims_cells(got, "bcp", ctx)
            acr, nums[f"{pop}: {lab}"]["acr"] = _claims_cells(got, "acr", ctx)
            rows.append([pop, lab, runs_scen_cell(len(got), m_scen, len(rs)), *bcp, *acr])
    sec.table(["population", "configuration", "runs measured (scenarios)", "BCP (Wilson 95%)",
               "BCP scenario bootstrap 95%", "ACR (Wilson 95%)", "ACR scenario bootstrap 95%"],
              rows)
    if unmeasured:
        sec.text(f"No measured bcp.py row for: {', '.join(unmeasured)}.")
    sec.numbers["preservation"] = nums


def _counterpart(sel: HealSelection, lab: str, mode: str, verdicts: str) -> str | None:
    """The configuration `lab` is set against: `mode` with the same modifiers and knobs,
    else with the one-sided knobs set aside in measure.ONE_SIDED_LADDER's order (heal's
    claim-level replays, +verify's replays per check, then the counterfactual's options
    too: measure.paired_differences' rule)."""
    rec = {**sel.rep[lab], "mode": mode, "verdicts": verdicts}
    for names in ((), *measure.ONE_SIDED_LADDER):
        cand = heal_label({**rec, **{k: None for k in names}})
        if cand in sel.by:
            return cand
    return None


# What each heal mode is set against besides incident (healing vs quarantine on the same
# detection): heal_rebuild minus heal is what rebuilding derived memories adds to
# filtering them, heal_oracle minus heal what perfect claim labels add (localisation).
HEAL_AGAINST = {"heal_rebuild": (("heal", "cf"),), "heal_oracle": (("heal", "cf"),)}


def heal_pairs(sel: HealSelection, labels: list[str]) -> list[tuple[str, str]]:
    """(A, B): every heal configuration minus incident on the same detection, then minus
    heal where HEAL_AGAINST says so."""
    out = []
    for lab in labels:
        mode = sel.rep[lab]["mode"]
        against = [("incident", "cf"), *HEAL_AGAINST.get(mode, ())]
        if mode in HEAL_MODES:
            out += [(lab, b) for b in (_counterpart(sel, lab, m, v) for m, v in against) if b]
    return out


def _heal_paired(sec: Section, sel: HealSelection, labels: list[str], counts: dict,
                 ctx: Ctx) -> None:
    rows, nums = [], []
    rnd = random.Random(ctx.seed)
    for a, b in heal_pairs(sel, labels):
        A, B = sel.by[a], sel.by[b]
        runs = sorted(rd for rd in set(A) & set(B) if measure.record_incident(A[rd]) is True)
        if not runs:
            continue
        of = {rd: scenario(A[rd]) for rd in runs}
        n_scen = len(set(of.values()))
        m = measure._paired_metrics(A, B, runs, sel.base, rnd, ctx.n_boot, cluster=of)
        both = [rd for rd in runs if counts[a][rd] and counts[b][rd]]
        b_scen = len({of[rd] for rd in both})
        claims = {k: boot_pooled_diff([counts[a][rd][k] for rd in both],
                                      [counts[b][rd][k] for rd in both], ctx.n_boot, ctx.seed,
                                      [of[rd] for rd in both])
                  if both else None for k in ("bcp", "acr")}
        rows.append([a, b, runs_scen_cell(len(runs), n_scen), measure._ci(m["ASR"], 100),
                     measure._ci(m["BR"], 100), measure._ci(m["BMP"], 100),
                     measure._ci(m["collateral"], 1, 2), measure._ci(m["poison left"], 100),
                     runs_scen_cell(len(both), b_scen), measure._ci(claims["bcp"], 100),
                     measure._ci(claims["acr"], 100)])
        nums.append({"a": a, "b": b, "population": "incident", "runs": len(runs),
                     "scenarios": n_scen, "small_sample": small(n_scen),
                     **{k: None if m[k] is None else [round(x, 4) for x in m[k]] for k in m},
                     "claims_runs": len(both), "claims_scenarios": b_scen,
                     **{k.upper(): None if v is None else [round(x, 4) for x in v]
                        for k, v in claims.items()}})
    rebuild = any(sel.rep[lab]["mode"] == "heal_rebuild" for lab in labels)
    sec.text("### Paired: heal minus incident (same detection), "
             + ("heal_rebuild and heal_oracle minus heal" if rebuild else "heal_oracle minus heal"),
             "A minus B on the incident runs both have, each run's pair kept inside its "
             f"scenario: {SCENARIO_BOOT}. ASR is the mean of per-run ASR "
             "(measure.paired_differences); BCP and ACR are pooled claim rates on the runs "
             "where both have a bcp.py row. Each configuration is one replay per run: read a "
             "difference against Section 2's replay-noise floor.")
    if rows:
        sec.table(["A", "B", "incident runs (scenarios)", "ASR (pp)", "BR (x100)", "BMP (pp)",
                   "collateral", "poison left (pp)", "runs with BCP (scenarios)", "BCP (pp)",
                   "ACR (pp)"], rows)
    else:
        sec.text("No pair: no incident configuration on the incident runs of the heal sweep.")
    sec.numbers["paired"] = nums


def heal_points(sel: HealSelection, labels: list[str], counts: dict, ctx: Ctx) -> dict:
    """What analysis_heal.png draws, per configuration on the incident runs (all runs when
    there are none): x = ACR or 1 - ASR, y = BCP, each (value, lo, hi) in percent with
    its n, every interval resampling scenarios. A point also carries what its direct label
    is built from (verdicts, modifiers, knobs)."""
    pop, keep = next(((p, k) for p, k in HEAL_POPULATIONS
                      if any(sel.records(lab, k) for lab in labels)), HEAL_POPULATIONS[-1])

    def pct(ci):
        return None if ci is None else [round(100 * x, 2) for x in ci]

    out = []
    for lab in labels:
        rs = sel.records(lab, keep)
        if not rs:
            continue
        rep = sel.rep[lab]
        blocked = boot_runs(rs, lambda r: (lambda n, d: (d - n, d) if d else None)(
            *measure.asr_counts([r])), ctx)
        got = [(counts[lab][r["run_dir"]], scenario(r)) for r in rs if counts[lab][r["run_dir"]]]
        boot = lambda key: boot_pooled([c[key] for c, _ in got], ctx.n_boot, ctx.seed,
                                       [s for _, s in got])
        out.append({"label": lab, "mode": rep["mode"], "verdicts": rep.get("verdicts"),
                    "modifiers": sorted(rep.get("modifiers") or ()),
                    "knobs": [list(k) for k in measure.drop_knobs(knobs(rep), ("repeat",))],
                    "runs": len(rs), "scenarios": len({scenario(r) for r in rs}),
                    "measured": len(got), "measured_scenarios": len({s for _, s in got}),
                    "blocked": pct(blocked[2]), "acr": pct(boot("acr")), "bcp": pct(boot("bcp"))})
    return {"population": pop, "points": out}


# recover.config_tag's suffix for a knob (the name of a configuration's copy): the short
# form a direct label gives a knob in.
KNOB_TAGS = {"family": str, "k": "k{}".format, "depth": "d{}".format, "cf_method": str,
             "cf_groups": lambda v: "groups", "cf_workers": "w{}".format, "observe": str,
             "stop_after_action": lambda v: "stop", "verify_n": "v{}".format,
             "claim_n": "c{}".format, "ddmin_budget": "b{}".format}
# Fill styles of a mode's further configurations; the first (fewest distinguishing
# parts) is filled. A sixth configuration of one mode repeats one, its label still apart.
VARIANT_FILLS = ("left", "bottom", "right", "top")
LABEL_WIDTH = 24            # characters per line of a direct label before it wraps


def _tokens(p: dict) -> list[tuple[str, str]]:
    """(kind, text): what makes a point's configuration one of its mode's, in label order:
    the verdict source, the modifiers, each knob away from its default (KNOB_TAGS)."""
    out = [("verdicts", f"[{p['verdicts']}]")] if p.get("verdicts") else []
    out += [("modifier", f"+{m}") for m in p.get("modifiers") or ()]
    return out + [(f"knob:{k}", KNOB_TAGS.get(k, lambda v, k=k: f"{k}={v}")(v))
                  for k, v in p.get("knobs") or ()]


def direct_labels(points: list[dict]) -> tuple[list[str], list[int], list, dict[str, str]]:
    """Per point: its direct label (the mode plus only what sets it apart from the other
    drawn configurations of its mode, '+verify v3', say) and its variant index (0 = the
    mode's plainest configuration, drawn filled); then the knob pairs every point shares
    (the figure note states them once) and the knob tags the labels use, spelled out."""
    shared = None
    for p in points:
        kn = {tuple(k) for k in p.get("knobs") or ()}
        shared = kn if shared is None else shared & kn
    by_mode: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(points):
        by_mode[p["mode"]].append(i)
    names, variant, used = [""] * len(points), [0] * len(points), {}
    for mode, idx in by_mode.items():
        toks = {i: _tokens(points[i]) for i in idx}
        common = set.intersection(*(set(t) for t in toks.values())) if len(idx) > 1 else None
        own = {i: [t for t in toks[i] if common is not None and t not in common] for i in idx}
        for rank, i in enumerate(sorted(idx, key=lambda i: (len(own[i]), points[i]["label"]))):
            variant[i] = rank
            head = mode + "".join(text for kind, text in own[i] if not kind.startswith("knob"))
            tags = [text for kind, text in own[i] if kind.startswith("knob")]
            for kind, text in own[i]:
                if kind.startswith("knob"):
                    k = kind.split(":", 1)[1]
                    v = next(v for kk, v in points[i]["knobs"] if kk == k)
                    used[text] = f"{k}={v}"
            names[i] = " ".join([head, *tags])
    order = [k for k in KNOBS if any(k == s[0] for s in shared or ())]
    shared_pairs = sorted(shared or (), key=lambda kv: order.index(kv[0]))
    return names, variant, shared_pairs, {t: used[t] for t in sorted(used) if t != used[t]}


def _tint(colour: str, amount: float = 0.45) -> str:
    """`colour` mixed with the white surface: the pale fill of a small-sample point."""
    rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - amount * (255 - c)):02x}" for c in rgb)


def heal_note(points: dict, shared: list, codes: dict[str, str]) -> list[str]:
    """The figure note, wrapped to the page: population, the knob setting every point
    shares, the label tags spelled out, how to read it."""
    parts = [points["population"]]
    if shared:
        parts.append("every point: " + ", ".join(f"{k}={v}" for k, v in shared))
    if codes:
        parts.append("label tags: " + ", ".join(f"{t} = {s}" for t, s in codes.items()))
    parts += ["up and right is better", f"intervals: {SCENARIO_BOOT}"]
    return textwrap.wrap("; ".join(parts), 125)


def label_clusters(xy: list[tuple[float, float, str]], tol: float = 2.0
                   ) -> list[tuple[float, float, str]]:
    """Direct labels of points that sit within `tol` percentage points of each other on
    both axes, joined into one label at the first of them (several configurations often
    reach the same rate), so no two labels print over each other."""
    groups: list[list[tuple[float, float, str]]] = []
    for p in xy:
        for g in groups:
            if abs(g[0][0] - p[0]) <= tol and abs(g[0][1] - p[1]) <= tol:
                if p[2] not in [q[2] for q in g]:
                    g.append(p)
                break
        else:
            groups.append([p])
    return [(g[0][0], g[0][1], "\n".join(q[2] for q in g)) for g in groups]


# Where a direct label may sit relative to its point (offset in points, alignment), tried
# in order until it overlaps no other label and no marker and stays inside the axes
# (bcp.py's figure places its labels the same way).
LABEL_SPOTS = [(dx, dy, ha, va) for dy, va in ((0, "center"), (7, "bottom"), (-7, "top"),
                                                (14, "bottom"), (-14, "top"))
               for dx, ha in ((6, "left"), (-6, "right"))]


def place_labels(ax, names: list[tuple[float, float, str]], renderer) -> None:
    """Each (x, y, text) of label_clusters at the first LABEL_SPOTS spot that collides
    with nothing already drawn. When none is free, the spot inside the axes with the
    fewest collisions, and a label that still crosses the axes' edge is moved in: a label
    never leaves the axes (so the figure never cuts it off)."""
    frame = ax.get_window_extent(renderer)
    marks = [ax.transData.transform((x, y)) for x, y, _ in names]
    placed = []
    px_per_pt = ax.figure.dpi / 72

    def draw(x, y, text, spot):
        dx, dy, ha, va = spot
        t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va=va, fontsize=6, color=INK2, linespacing=1.1, zorder=4)
        return t, t.get_window_extent(renderer).padded(1.5)

    def outside(box) -> float:
        return (max(frame.x0 - box.x0, 0) + max(box.x1 - frame.x1, 0)
                + max(frame.y0 - box.y0, 0) + max(box.y1 - frame.y1, 0))

    def clashes(box) -> int:
        return (sum(box.overlaps(b) for b in placed)
                + sum(box.contains(*m) for m in marks))

    for x, y, text in names:
        best = None
        for spot in LABEL_SPOTS:
            t, box = draw(x, y, text, spot)
            score = (outside(box) > 0, clashes(box), outside(box))
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, box)
            else:
                t.remove()
            if score[:2] == (False, 0):
                break
        _, t, box = best
        if outside(box):                         # move it in, by whole points
            sx = max(frame.x0 - box.x0, 0) - max(box.x1 - frame.x1, 0)
            sy = max(frame.y0 - box.y0, 0) - max(box.y1 - frame.y1, 0)
            dx, dy = t.xyann
            t.xyann = (dx + math.ceil(abs(sx) / px_per_pt) * (1 if sx > 0 else -1),
                       dy + math.ceil(abs(sy) / px_per_pt) * (1 if sy > 0 else -1))
            box = t.get_window_extent(renderer).padded(1.5)
        placed.append(box)


def plot_heal(points: dict, path: Path) -> Path:
    plt, fig, (ax, ax2) = _figure(2, 3.6)
    pts = [p for p in points["points"] if p["bcp"]]
    names, variant, shared, codes = direct_labels(pts)
    per_mode = Counter(p["mode"] for p in pts)
    any_small = any_variant = False
    labels = []
    for a, key in ((ax, "acr"), (ax2, "blocked")):
        xy = []
        for p, name, var in zip(pts, names, variant):
            if not p[key]:
                continue
            slot = HEAL_COMPARE.index(p["mode"])
            colour, marker = HEAL_COLOURS[slot], HEAL_MARKERS[slot]
            (x, xlo, xhi), (y, ylo, yhi) = p[key], p["bcp"]
            pale = small(p.get("measured_scenarios", p["measured"]))
            any_small |= pale
            any_variant |= var > 0
            a.plot([xlo, xhi], [y, y], color=colour, lw=1.2, alpha=0.7, zorder=2)
            a.plot([x, x], [ylo, yhi], color=colour, lw=1.2, alpha=0.7, zorder=2)
            # A mode's plainest configuration filled, its others half-filled; pale for a
            # small sample, so both can be told apart without the label.
            a.plot(x, y, marker=marker, ms=8, ls="none", color=colour, zorder=3,
                   mfc=_tint(colour) if pale else colour, mec=colour, mew=1.6,
                   fillstyle="full" if var == 0 else VARIANT_FILLS[(var - 1) % len(VARIANT_FILLS)],
                   markerfacecoloralt=SURFACE)
            xy.append((x, y, textwrap.fill(name, LABEL_WIDTH)))
        labels.append((a, label_clusters(xy)))
        a.set_xlim(-3, 108)
        a.set_ylim(-3, 108)
        a.set_xticks(range(0, 101, 20))
        a.set_yticks(range(0, 101, 20))
    shown = [(slot, mode) for slot, mode in enumerate(HEAL_COMPARE) if per_mode[mode]]
    for slot, mode in shown:
        ax.plot([], [], marker=HEAL_MARKERS[slot], ms=7, ls="none", color=HEAL_COLOURS[slot],
                label=mode if slot < HEAL_PRIMARY else f"other: {mode}")
    if any_variant:
        ax.plot([], [], marker="o", ms=7, ls="none", color=MUTED, fillstyle="left",
                markerfacecoloralt=SURFACE, label="half fill: another setting of the mode")
    if any_small:
        ax.plot([], [], marker="o", ms=7, ls="none", color=MUTED, mfc=_tint(MUTED),
                label=f"pale: fewer than {SMALL} scenarios (small sample)")
    _titles(ax, "Benign kept vs attack claims removed", "ACR: attack claims removed (%)",
            "BCP: benign claims kept (%)")
    _titles(ax2, "Benign kept vs attack blocked", "1 - ASR: replayed tasks not attacked (%)",
            "BCP: benign claims kept (%)")
    note = heal_note(points, shared, codes)
    # Four legend columns fit the page while every entry is short; the 'other:' entries
    # and the fill keys need three.
    entries = len(shown) + any_variant + any_small
    ncol = 4 if entries <= 5 and not (any_variant or any_small) else 3
    _legend(fig, [ax], False, ncol=ncol, note="\n".join(note))

    def direct(renderer) -> None:
        for a, xy in labels:
            place_labels(a, xy, renderer)
    return _save(fig, plt, path, legend_rows=math.ceil(entries / ncol) + len(note),
                 after_layout=direct)


# --- report -----------------------------------------------------------------------------

# Section 9 is not here: it runs once per --heal file (heal_specs, sections()).
SECTIONS = (s1_families, s2_repeats, s3_k, s4_depth, s5_reretrieve, s6_groups, s7_claims, s8_flat)


def run_section(fn, inp: Inputs, ctx: Ctx, title: str | None = None) -> Section:
    """A section that hits something unexpected in a partial file is reported as failed
    and the others still run: the batch may be mid-write."""
    try:
        return fn(inp, ctx)
    except Exception as exc:                      # noqa: BLE001 - keep the other sections
        traceback.print_exc()
        return Section(title or fn.__name__).skip(f"failed: {type(exc).__name__}: {exc}")


def sections(inp: Inputs, ctx: Ctx) -> list[Section]:
    """Sections 1-8, then section 9 once per heal file."""
    out = [run_section(fn, inp, ctx) for fn in SECTIONS]
    for name, title, figure in heal_specs(inp):
        out.append(run_section(lambda i, c, n=name, t=title, f=figure: s9_heal(i, c, n, t, f),
                               inp, ctx, title))
    return out


def inputs_table(inp: Inputs) -> list[str]:
    rows = [[name, f["path"], f["state"], f["records"], f["runs"], f["part_files"]]
            for name, f in inp.found.items()]
    runs_state = "ok" if inp.runs_dir.is_dir() else "missing"
    rows.append(["runs", str(inp.runs_dir), runs_state, "-", "-", "-"])
    return [measure.md_table(["input", "path", "state", "records", "runs",
                              "unmerged part files"], rows), ""]


def report(sections: list[Section], inp: Inputs, ctx: Ctx) -> str:
    md = ["# MemHeal experiment analysis\n",
          "Generated by `analyze.py` from recover.py records and run summaries; every number "
          "below is computed from those files. ASR = attacker outcomes over every replayed "
          "task (Wilson 95%); BMP = probes still correct among those the same run answered "
          "correctly with no recovery; BR = mean share of tasks whose action matches the "
          f"clean run (measure.py). Cells over fewer than {SMALL} runs are marked 'small "
          f"sample' (section 9: fewer than {SMALL} scenarios; its intervals resample "
          "scenarios). Section 2's replay-noise floor applies to every single-replay "
          "comparison.\n",
          "## Inputs\n", *inputs_table(inp)]
    for s in sections:
        md.append(f"## {s.title}\n")
        if s.skipped:
            md += [f"Skipped: {s.skipped}.", ""]
            continue
        md += s.md
        md += [f"![{p.stem}]({p.name})\n" for p in s.figures]
    return "\n".join(md)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, default in INPUTS.items():
        if name == "heal":
            ap.add_argument("--heal", type=Path, nargs="+", default=[ROOT / p for p in HEAL_INPUTS],
                            metavar="PATH", help="heal sweep records files, a section each "
                            f"(default {' '.join(HEAL_INPUTS)})")
            continue
        ap.add_argument(f"--{name.replace('_', '-')}", type=Path, default=ROOT / default,
                        dest=name, metavar="PATH", help=f"records file (default {default})")
    ap.add_argument("--runs", type=Path, default=ROOT / RUNS, metavar="DIR",
                    help=f"history run folders (default {RUNS})")
    ap.add_argument("--preservation", type=Path, default=ROOT / PRESERVATION, metavar="PATH",
                    help=f"bcp.py's per-record rows (default {PRESERVATION})")
    ap.add_argument("--flat-range", default=FLAT_RANGE, help="flat history run numbers")
    ap.add_argument("--evolving-range", default=EVOLVING_RANGE,
                    help="evolving history run numbers")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--aa", action="store_true",
                    help="also bootstrap the canonical sweep's A/A rows (slow: ~10 s)")
    ap.add_argument("--no-parts", action="store_true",
                    help="ignore a running batch's unmerged .parts files")
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    heal = heal_input_names(args.heal)
    paths = {}
    for name in INPUTS:
        paths.update(heal if name == "heal" else [(name, getattr(args, name))])
    inp = Inputs(paths, args.runs, not args.no_parts, args.preservation,
                 tuple(name for name, _ in heal))
    ctx = Ctx(out=args.out, n_boot=args.boot, figures=not args.no_figures, aa=args.aa,
              flat_range=args.flat_range, evolving_range=args.evolving_range)
    for name in paths:
        inp.get(name)
    inp.preservation()
    for name, f in inp.found.items():
        if f["state"] != "ok":
            print(f"input {f['state']}: {name} ({f['path']}; {f['records']} records, "
                  f"{f['runs']} runs{', +%d part files' % f['part_files'] if f['part_files'] else ''})")
    secs = sections(inp, ctx)
    (args.out / "analysis_tables.md").write_text(report(secs, inp, ctx))
    numbers = {"inputs": inp.found, "small_sample_runs": SMALL,
               "sections": {s.title: ({"skipped": s.skipped} if s.skipped else s.numbers)
                            for s in secs}}
    (args.out / "analysis.json").write_text(json.dumps(numbers, indent=1, default=str))
    for s in secs:
        print(f"{'skipped' if s.skipped else 'ok':8} {s.title}"
              + (f": {s.skipped}" if s.skipped else "")
              + "".join(f"\n         figure {p}" for p in s.figures))
    print(f"wrote {args.out / 'analysis_tables.md'} and {args.out / 'analysis.json'}")
    return numbers


if __name__ == "__main__":
    main()
