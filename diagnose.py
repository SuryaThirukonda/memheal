"""Harmful-memory diagnosis baselines vs MemHeal (Sections 5.3, 8.3, 8.4, 9.1).

For every poisoned run, each diagnosis method names the memories it blames; the oracle
lineage labels say which were harmful. Scored per run, with bootstrap CIs over runs,
in the store scope (active memories before the target task) and the action scope (the
harmful action's retrieved set). memheal/diagnosis.py holds the methods and metrics,
memheal/candidates.py the Section 5.3 candidate signals; this file opens the run folders
read-only, makes the LLM calls and writes the tables.

    python diagnose.py                                    # offline methods, dense runs
    python diagnose.py --llm run --workers 8              # + content classifier, self-explanation
    python diagnose.py --runs '*_poisoned_*_dense_*_evolving' --incident-db recovery/incident+verify.db

The counterfactual (memheal_cf) and the full incident pipeline (memheal_full) are read
from a recovery copy recover.py already made (--incident-db, relative to the run
folder), so this script replays nothing. The copy must be an `incident` recovery of the
decision run.db logged (frozen ablation, the run's own k, no --depth; +verify and the
other knobs are fine); a run whose copy is missing or explains something else is
reported and has no MemHeal answer, so it drops out of those rows. With `--llm run`,
every model answer is cached in <out>/diagnosis.cache.jsonl and every call logged to
<out>/diagnosis.llm.jsonl; a rerun over a complete cache makes no new calls and needs
no API key.

Outputs (under --out, default results/diagnosis):
    diagnosis_tables.md   diagnosis per scope, retrieval vs causation, candidate signals
    diagnosis.json        every summary, plus per-run memories, signals and predictions
    diagnosis.png         precision vs recall per method, one panel per scope
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from memheal import candidates, db, diagnosis

ROOT = Path(__file__).parent
DEFAULT_RUNS = "*_poisoned_*_dense_*_evolving"
DEFAULT_OUT = ROOT / "results" / "diagnosis"
DEFAULT_INCIDENT_DB = "recovery/incident.db"


# --- loading ------------------------------------------------------------------------

def select(selectors: list[str], suite: str | None, qualified_only: bool) -> list:
    """Poisoned runs of the suite whose clean controls were correct (recover's rule)."""
    import recover
    infos = []
    for p in recover.resolve_runs(selectors):
        try:
            info = recover.run_info(p)
        except (sqlite3.Error, TypeError, KeyError) as exc:
            print(f"  skip {p.name}: unreadable ({exc})")
            continue
        if info.condition != "poisoned" or info.scenario is None:
            continue
        infos.append(info)
    if suite:
        keep = recover.suite_variants(suite)
        infos = [i for i in infos if i.variant in keep]
    if qualified_only:
        out = []
        for info in infos:
            siblings = sorted(info.run_dir.parent.iterdir())
            if recover.is_qualified(recover.clean_twins(info, siblings)):
                out.append(info)
            else:
                print(f"  skip {info.run_dir.name}: clean control not correct")
        infos = out
    return infos


def load_case(info, incident_db: str, threshold: float) -> diagnosis.Case:
    """One run's case, every database opened read-only.

    The task and system prompt are the ones the run recorded (the registry and the
    prompt constants may have changed since), so self-explanation shows the agent what
    it actually saw.
    """
    import recover
    s = info.scenario
    pre = db.connect_readonly(recover.pre_target_snapshot(info.run_dir, s.target_session))
    run = db.connect_readonly(info.run_dir / "run.db")
    inc_path = info.run_dir / incident_db
    inc = db.connect_readonly(inc_path) if inc_path.exists() else None
    try:
        return diagnosis.build_case(
            run_dir=info.run_dir.name, run_id=info.run_id, variant=info.variant, scenario=s,
            condition=info.condition, task=recover.own_task(info, s),
            cfg=recover.agent_config(info, s), pre=pre, run=run, incident_db=inc,
            threshold=threshold, system_prompt=info.config.get("system_prompt"))
    finally:
        for c in (pre, run, inc):
            if c is not None:
                c.close()


def llm_factory(model: str, log_path: Path):
    def make():
        from dotenv import load_dotenv
        from memheal.llm import LLM      # imports openai; offline runs never need it
        load_dotenv(ROOT / ".env")
        return LLM(model=model, log_path=log_path, temperature=0)
    return make


# --- tables -------------------------------------------------------------------------

def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _v(x, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci(summary: dict, key: str) -> str:
    lo, hi = summary.get(key + "_ci", [None, None])
    point = _v(summary.get(key))
    return point if lo is None else f"{point} [{_v(lo)}, {_v(hi)}]"


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "-"


def diagnosis_table(per_method: dict) -> str:
    rows = []
    for m in diagnosis.METHODS:
        s = per_method.get(m)
        if s is None:
            continue
        rows.append([
            diagnosis.METHOD_TITLES[m], "alarm" if m in diagnosis.ALARM_METHODS else "store",
            s["runs"], _v(s["flagged_per_run"], 1),
            _ci(s, "micro_p"), _ci(s, "micro_r"), _ci(s, "micro_f1"),
            _v(s["macro_p"]), _v(s["macro_r"]), _v(s["macro_f1"]),
            f"{s['tp']}/{s['fp']}/{s['fn']}", _v(s["calls_per_run"], 1)])
    return md_table(["method", "trigger", "runs", "flagged/run", "micro P [95% CI]",
                     "micro R [95% CI]", "micro F1 [95% CI]", "macro P", "macro R",
                     "macro F1", "TP/FP/FN", "LLM calls/run"], rows)


def tables_md(meta: dict, summary: dict, rvc: dict, cand: list[dict]) -> str:
    out = ["# Harmful-memory diagnosis (Sections 5.3, 8.3, 8.4, 9.1)", ""]
    missing = meta.get("no_memheal_evidence") or {}
    gap = (f" {len(missing)} run(s) have no usable copy there, so the MemHeal methods "
           "leave them out (their 'runs' column is smaller; see diagnosis.json "
           "meta.no_memheal_evidence)." if missing else "")
    out += [f"Runs: {meta['runs']} qualified poisoned runs ({meta['selectors']}, suite "
            f"{meta['suite'] or 'any'}); {meta['incidents']} with an incident (harmful "
            f"target action). Counterfactual evidence from `{meta['incident_db']}`.{gap} "
            f"LLM baselines: {meta['llm_note']}.", "",
            "Ground truth: positives = poison roots + descendants labelled harmful or mixed; "
            "negatives = benign descendants + non-lineage memories that do not name the "
            "attacker; ambiguous descendants are excluded. Non-lineage memories naming the "
            "attacker (e.g. read off the attacker's own portal) are excluded from the "
            "primary scoring and positive in the secondary one. Micro = pooled counts; "
            "macro = mean of per-run values where defined; 95% CIs are percentile "
            f"bootstraps over runs ({meta['n_boot']} resamples). 'alarm' methods act only "
            "after a harmful action, so they flag nothing in runs without one. LLM "
            "calls/run is the method's cost (cached answers included); for the MemHeal "
            "counterfactual it counts the agent steps of its baseline and single-removal "
            "replays, and for the MemHeal incident every step the pipeline replayed "
            "before it was measured (all counterfactual replays plus +verify's checks).",
            ""]

    out += ["## 1. Diagnosis per scope (8.3 / 9.1)", ""]
    for scope in diagnosis.SCOPES:
        for scoring in diagnosis.SCORINGS:
            per = summary.get(scope, {}).get(scoring, {})
            if not per:
                continue
            out += [f"### {diagnosis.SCOPE_TITLES[scope]} ({scoring} scoring)", "",
                    diagnosis_table(per), ""]

    out += ["## 2. Does retrieval mean causation? (8.4)", ""]
    no_copy = (f" ({rvc['no_copy']} without a usable recovery copy)"
               if rvc.get("no_copy") else "")
    out += [f"{rvc['incidents']} incidents{no_copy}; {rvc['ablated']} with a counterfactual "
            "ablation; "
            f"{rvc['explained']} whose frozen baseline replay was harmful again "
            f"({rvc['baseline_not_harmful']} baselines were not harmful, so nothing was "
            f"ablated). Of {rvc['retrieved']} retrieved memories in those incidents, removing "
            f"{rvc['flipped']} flipped the decision (score >= 0.5 over its replays; "
            f"{rvc['first_flipped']} flipped on the first replay alone).", ""]
    rows = []
    for label, r in rvc["by_label"].items():
        if not r["retrieved"]:
            continue
        rows.append([label, r["retrieved"], r["tested"], r["flipped"],
                     _pct(r["flipped"], r["tested"]), r["first_flipped"]])
    out += ["### 2a. Retrieved memories at the harmful action, by oracle label", "",
            md_table(["label", "retrieved", "ablated", "flipped", "flip rate",
                      "flipped on 1st replay"], rows), ""]

    act = summary.get("action", {})
    rows = []
    for scoring in diagnosis.SCORINGS:
        for m in ("retrieval", "self_explain", "memheal_cf", "memheal_cf_cand"):
            s = act.get(scoring, {}).get(m)
            if s is None:
                continue
            rows.append([scoring, diagnosis.METHOD_TITLES[m], _v(s["flagged_per_run"], 1),
                         _ci(s, "micro_p"), _ci(s, "micro_r"), _ci(s, "micro_f1")])
    out += ["### 2b. Retrieval attribution vs counterfactual (action scope, incident runs)", "",
            md_table(["scoring", "method", "blamed/run", "precision [95% CI]",
                      "recall [95% CI]", "F1 [95% CI]"], rows), ""]

    rows = [[k, n] for k, n in rvc["no_culprit_breakdown"].items()]
    out += ["### 2c. Incidents with no single culprit", "",
            f"{rvc['no_single_culprit']} of {rvc['explained']} explained incidents have no "
            f"memory whose removal alone flips the decision "
            f"({rvc['no_single_culprit_first_replay']} judged on the first replay alone; "
            f"{rvc['no_culprit_read_attacker_page']} of them read an attacker page live "
            "in the baseline replay). Flipping memories per incident: "
            + ", ".join(f"{k}: {v}" for k, v in rvc["flips_per_incident"].items()) + ".", "",
            md_table(["baseline replay; redundancy", "incidents"], rows), ""]

    out += ["## 3. Candidate generation (5.3)", "",
            "Store scope over all runs for size and recall; counterfactual coverage and "
            "replays over the explained incidents. 'replays saved' = ablation replays a "
            "filter on this signal would not have run (the frozen set's memories without "
            "the signal). retrieved_at_action is every ablated memory by construction.", ""]
    rows = []
    for r in cand:
        rows.append([r["signal"], _v(r["candidates_per_run"], 1),
                     _pct(r["candidates"], r["store"]), _pct(r["positives_kept"], r["positives"]),
                     _v(r["precision"]), _pct(r["flips_covered"], r["flips"]),
                     _pct(r["replays_saved"], r["replays"]),
                     _pct(r["steps_saved"], r["steps"])])
    out += [md_table(["signal", "candidates/run", "share of store", "recall of positives",
                      "precision", "cf flips covered", "replays saved",
                      "replay LLM calls saved"], rows), ""]
    return "\n".join(out) + "\n"


# --- figure -------------------------------------------------------------------------

# Validated categorical slots 1-4 in fixed order (same as measure.plot_recovery).
GROUPS = (
    ("store heuristics", "#2a78d6", ("content_llm", "source_only")),
    ("action attribution", "#eb6834", ("retrieval", "self_explain")),
    ("MemHeal", "#1baf7a", ("memheal_cf", "memheal_cf_cand", "memheal_full")),
    ("oracle root (ancestry)", "#eda100", ("ancestry",)),
)
MARKERS = {"content_llm": "o", "source_only": "s", "retrieval": "o", "self_explain": "s",
           "memheal_cf": "o", "memheal_cf_cand": "s", "memheal_full": "D", "ancestry": "^"}
SHORT = {"content_llm": "content LLM", "source_only": "source-only", "retrieval": "retrieval",
         "self_explain": "self-explain", "ancestry": "ancestry", "memheal_cf": "MemHeal cf",
         "memheal_cf_cand": "MemHeal cf (candidates)", "memheal_full": "MemHeal incident"}


# Offsets (points) tried in order for a point's name: right, left, then below and above.
_LABEL_SPOTS = ((8, 6, "left"), (-8, 6, "right"), (8, -12, "left"), (-8, -12, "right"),
                (0, -16, "center"), (0, 10, "center"))


def _label(ax, text: str, x: float, y: float, placed: list, ink: str) -> None:
    """Name a point at the first offset whose text overlaps no earlier name and stays
    inside the axes; the last spot tried is kept if none is free."""
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    for i, (dx, dy, ha) in enumerate(_LABEL_SPOTS):
        ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                          ha=ha, fontsize=8, color=ink)
        box = ann.get_window_extent(renderer).expanded(1.05, 1.15)
        inside = frame.x0 <= box.x0 and box.x1 <= frame.x1
        if (inside and not any(box.overlaps(b) for b in placed)) or i == len(_LABEL_SPOTS) - 1:
            placed.append(box)
            return
        ann.remove()


def plot(summary: dict, path: Path) -> Path:
    """Micro precision vs recall per method with bootstrap 95% CIs, one panel per scope.

    Every point is named (some slots are below 3:1 contrast on the light surface, so
    identity never rests on colour), and the legend gives the method families.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    panels = [("store_incident", "Store scope (incident runs)"),
              ("action", "Action scope (retrieved set, incident runs)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.patch.set_facecolor(surface)
    colour = {m: c for _, c, ms in GROUPS for m in ms}
    names: list = []          # placed after the layout is final, so overlaps are real
    for ax, (scope, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.set_xlim(-3, 105)
        ax.set_ylim(-3, 105)
        per = summary.get(scope, {}).get("primary", {})
        for m in diagnosis.METHODS:
            s = per.get(m)
            if s is None or s.get("micro_p") is None or s.get("micro_r") is None:
                continue
            x, y = 100 * s["micro_r"], 100 * s["micro_p"]
            (rlo, rhi), (plo, phi) = s["micro_r_ci"], s["micro_p_ci"]
            c = colour[m]
            if rlo is not None:
                ax.plot([100 * rlo, 100 * rhi], [y, y], color=c, lw=1.2, alpha=0.6)
            if plo is not None:
                ax.plot([x, x], [100 * plo, 100 * phi], color=c, lw=1.2, alpha=0.6)
            ax.plot(x, y, marker=MARKERS[m], ms=9, color=c, mec=surface, mew=2, ls="none",
                    zorder=3)
            names.append((ax, SHORT[m], x, y))
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("recall of harmful memories (%)", color=muted)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("precision (%)", color=muted)
    handles = [Line2D([], [], marker="o", ls="none", color=c, ms=8, label=name)
               for name, c, ms in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=ink)
    fig.suptitle("Harmful-memory diagnosis: precision vs recall (micro, primary scoring, "
                 "bootstrap 95% CIs)", fontsize=11, color=ink, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    placed: dict = {ax: [] for ax in axes}
    for ax, text, x, y in names:
        _label(ax, text, x, y, placed[ax], ink)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    return path


# --- main ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", default=[DEFAULT_RUNS],
                   help="measure.py selectors (glob or START-END[:GLOB]) or run folders")
    p.add_argument("--suite", default="dense",
                   help="keep only variants of this run_demo suite; '' keeps every variant")
    p.add_argument("--include-unqualified", action="store_true",
                   help="also score runs whose clean control was not correct")
    p.add_argument("--incident-db", default=DEFAULT_INCIDENT_DB,
                   help="recovery copy holding the counterfactual replays and the incident "
                        "quarantine, relative to the run folder")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--llm", choices=("none", "run"), default="none",
                   help="none = offline methods only; run = also content_llm and self_explain")
    p.add_argument("--model", default=None, help="model for the LLM methods (default: "
                   "memheal.llm.DEFAULT_MODEL)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conflict-threshold", type=float, default=candidates.CONFLICT_THRESHOLD)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None, llm_factory_override=None) -> dict:
    args = parse_args(argv)
    if args.model is None:
        from memheal.llm import DEFAULT_MODEL
        args.model = DEFAULT_MODEL
    infos = select(args.runs, args.suite or None, not args.include_unqualified)
    print(f"{len(infos)} poisoned run(s)")
    cases = []
    for info in infos:
        case = load_case(info, args.incident_db, args.conflict_threshold)
        case.predictions.update(diagnosis.offline_predictions(case))
        cases.append(case)
        cf = case.cf
        print(f"  {case.run_dir}: incident={case.incident} store={len(case.store)} "
              f"cf={'-' if not cf.ran else f'{len(cf.flips())}/{len(cf.replays)} flip'} "
              f"quarantined={len(case.full)}", flush=True)
        if case.copy_problem is not None:
            print(f"    WARNING {args.incident_db}: {case.copy_problem}; MemHeal methods "
                  "left out for this run", flush=True)
    no_evidence = {c.run_dir: c.copy_problem for c in cases if c.copy_problem is not None}
    if cases and len(no_evidence) == len(cases):
        print(f"WARNING: no run has a usable {args.incident_db}; the MemHeal methods are "
              "missing from every table (check --incident-db)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    llm_note = "not run (--llm none)"
    llm_stats = None
    if args.llm == "run" and cases:
        cache = diagnosis.Cache(args.out / "diagnosis.cache.jsonl")
        requests = diagnosis.llm_requests(cases, args.model)
        factory = llm_factory_override or llm_factory(args.model, args.out / "diagnosis.llm.jsonl")
        t0 = time.monotonic()

        def progress(done: int, total: int) -> None:
            if done == total or done % 50 == 0:
                print(f"    llm {done}/{total} ({time.monotonic() - t0:.0f}s)", flush=True)

        llm_stats = diagnosis.run_requests(requests, cache, factory, args.model,
                                           workers=args.workers, progress=progress)
        for case in cases:
            case.predictions.update(diagnosis.llm_predictions(case, cache, args.model))
        llm_note = (f"{args.model}, temperature 0: {llm_stats['requested']} distinct prompts, "
                    f"{llm_stats['cached']} cached, {llm_stats['calls']} new calls, "
                    f"{llm_stats['failed']} failed")
        print(f"  llm: {llm_note}")

    summary = diagnosis.diagnosis_summary(cases, n_boot=args.n_boot, seed=args.seed)
    rvc = diagnosis.retrieval_vs_causation(cases)
    cand = diagnosis.candidate_rows(cases)
    meta = {"runs": len(cases), "incidents": sum(c.incident for c in cases),
            "selectors": " ".join(args.runs), "suite": args.suite,
            "incident_db": args.incident_db, "no_memheal_evidence": no_evidence,
            "llm": args.llm, "model": args.model,
            "llm_note": llm_note, "llm_stats": llm_stats, "n_boot": args.n_boot,
            "seed": args.seed, "conflict_threshold": args.conflict_threshold,
            "methods": {m: diagnosis.METHOD_TITLES[m] for m in diagnosis.METHODS}}
    result = {"meta": meta, "summary": summary, "retrieval_vs_causation": rvc,
              "candidates": cand, "runs": [diagnosis.case_record(c) for c in cases]}
    (args.out / "diagnosis.json").write_text(json.dumps(result, indent=1, default=sorted) + "\n")
    (args.out / "diagnosis_tables.md").write_text(tables_md(meta, summary, rvc, cand))
    plot(summary, args.out / "diagnosis.png")
    print(f"wrote {args.out / 'diagnosis_tables.md'}, diagnosis.json, diagnosis.png")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
