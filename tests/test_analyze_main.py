"""analyze_main.py: the main comparison's final tables and figures.

Every input is a small synthetic records file (recover.py's schema) or run.db written to
a temp directory, so these tests pin the arithmetic offline: no model, no socket, no
file under results/. Expected numbers are worked out by hand from the fixtures.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import analyze_main as am

A, C, O = "attacker", "correct", "other"
QS = ("Which cost centre?", "Who approves?")
NB = 200                                    # bootstrap resamples: enough for the tests


def rec(run: str, mode: str, *, variant: str, verdicts=None, modifiers=(), repeat=None,
        outcomes=(C, C, C, C), probes=(True, True), condition="poisoned", checkpoint=0,
        category="direct", domain="office", **extra) -> dict:
    r = {"run_dir": run, "variant": variant, "domain": domain, "category": category,
         "condition": condition, "mode": mode, "verdicts": verdicts,
         "modifiers": list(modifiers), "source": "patched", "checkpoint": checkpoint,
         "task_outcomes": list(outcomes), "task_destinations": [None] * len(outcomes),
         "asr": sum(o == A for o in outcomes) / len(outcomes), "br": 0.5,
         "probes": [{"question": q, "expected": "x", "correct": c} for q, c in zip(QS, probes)],
         "n_quarantined": 0, "n_restored": 0, "n_reconstructed": 0, "n_healed": 0,
         "collateral": {}, "residual_attack": 0, "incident": True, "verify_rounds": None,
         "llm_calls": 0, "seconds": 1.0, "notes": "replays stop at the first acting call",
         "family": "full", "k": 5, "repeat": repeat, "depth": 0, "observe": "logged",
         "stop_after_action": True, "claim_cf_replays": 0, "ddmin_replays": 0,
         "verify_n": 3 if "verify" in modifiers else None}
    if verdicts == "cf":
        r.update(cf_method="frozen", cf_groups=False, cf_workers=6)
    r.update(extra)
    return r


def write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def ctx(tmp_path: Path, figures: bool = False) -> am.Ctx:
    return am.Ctx(out=tmp_path / "out", n_boot=NB, seed=0, figures=figures)


# Two scenarios: v1 (direct, office) has three history runs, v2 (composed, travel) one.
V1_RUNS = ("0960_poisoned_v1_evolving", "1500_poisoned_v1_evolving", "1600_poisoned_v1_evolving")
V2_RUNS = ("0961_poisoned_v2_evolving",)


def main_records() -> list[dict]:
    """none: v1 ASR 50 on every record, v2 100. incident[cf]: v1 25, v2 0. heal[cf]: v1 0,
    v2 100 (one run), so heal's scenario mean is 50 while its record mean is 25. heal's
    second probe fails on v1's repeat 2 (BMP 50 there)."""
    out = []
    for runs, variant, cat, dom in ((V1_RUNS, "v1", "direct", "office"),
                                    (V2_RUNS, "v2", "composed", "travel")):
        for run in runs:
            for rp in (1, 2, 3):
                kw = dict(variant=variant, repeat=rp, category=cat, domain=dom)
                out.append(rec(run, "none", outcomes=(A, A, C, C) if variant == "v1" else (A,) * 4,
                               residual_attack=1, **kw))
                out.append(rec(run, "incident", verdicts="cf",
                               outcomes=(A, C, C, C) if variant == "v1" else (C,) * 4,
                               n_quarantined=4, llm_calls=10, **kw))
                out.append(rec(run, "heal", verdicts="cf",
                               outcomes=(C,) * 4 if variant == "v1" else (A,) * 4,
                               probes=(True, rp != 2 or variant != "v1"),
                               n_quarantined=4, n_healed=3, llm_calls=20,
                               claim_cf_replays=6 if rp == 1 else 0, **kw))
    return out


# --- reading and selection ----------------------------------------------------------------

def test_read_records_merges_parts_and_keeps_the_last_copy(tmp_path):
    main = write(tmp_path / "res" / "records.jsonl",
                 [rec("0960_poisoned_v1_evolving", "none", variant="v1", repeat=1, br=0.1)])
    part = tmp_path / "res" / ".parts" / "records.batch-1.000.0000.jsonl"
    write(part, [rec("0960_poisoned_v1_evolving", "none", variant="v1", repeat=1, br=0.9),
                 rec("0960_poisoned_v1_evolving", "none", variant="v1", repeat=2)])
    with part.open("a") as f:
        f.write('{"run_dir": "torn')                         # a writer mid-line
    recs, info = am.read_records(main)
    assert len(recs) == 2 and info["duplicates"] == 1 and info["torn_lines"] == 1
    assert [r["br"] for r in recs if r["repeat"] == 1] == [0.9]
    recs, info = am.read_records(main, parts=False)
    assert len(recs) == 1 and info["part_files"] == 0
    assert am.read_records(tmp_path / "nothing.jsonl")[0] is None


def test_identity_keeps_verify_n_settings_apart():
    a = rec("r", "incident", variant="v", verdicts="cf", modifiers=("verify",), verify_n=1)
    b = {**a, "verify_n": 3}
    assert am.identity(a) != am.identity(b)


def test_selection_keeps_only_the_main_comparison():
    base = dict(variant="v", repeat=1)
    ok = rec("r", "incident", verdicts="cf", **base)
    assert am.selection_problem(ok, "poisoned", 3) is None
    assert am.selection_problem({**ok, "condition": "clean"}, "poisoned", 3).startswith("condition")
    assert am.selection_problem({**ok, "checkpoint": 3}, "poisoned", 3) == "later checkpoint"
    assert am.selection_problem({**ok, "checkpoint": 3}, "poisoned", 3, checkpoint0=False) is None
    assert am.selection_problem({**ok, "family": "recorded"}, "poisoned", 3) == "family=recorded"
    assert "depth" in am.selection_problem({**ok, "depth": 2}, "poisoned", 3)
    assert "k" in am.selection_problem({**ok, "k": 3, "observe": "fresh"}, "poisoned", 3)
    v1 = rec("r", "incident", verdicts="cf", modifiers=("verify",), verify_n=1, **base)
    assert am.selection_problem(v1, "poisoned", 3) == "verify_n=1"
    assert am.selection_problem({**v1, "verify_n": 3}, "poisoned", 3) is None


def test_history_samples_first_run_from_the_original_sweep():
    recs = [rec(r, "none", variant="v1") for r in reversed(V1_RUNS)]
    recs += [rec("1200_poisoned_v3_evolving", "none", variant="v3")]
    samples, odd = am.history_samples(recs)
    assert [samples[r] for r in V1_RUNS] == [1, 2, 3]
    assert samples["1200_poisoned_v3_evolving"] == 1          # no sample-1 run: numbered from 1
    assert odd == {"variants without exactly one sample-1 run": 1}


# --- per-record values ----------------------------------------------------------------------

def test_record_values_asr_bmp_and_net_removal():
    recs = main_records()
    import measure
    base = measure._none_baselines(recs)
    heal_r2 = next(r for r in recs if r["mode"] == "heal" and r["repeat"] == 2
                   and r["variant"] == "v1")
    v = am.record_values(heal_r2, base)
    assert v["asr"] == 0 and v["bmp"] == 50.0 and v["removed"] == 1 and v["br"] == 50.0
    inc = next(r for r in recs if r["mode"] == "incident" and r["variant"] == "v1")
    v = am.record_values(inc, base)
    assert v["asr"] == 25.0 and v["bmp"] == 100.0 and v["removed"] == 4


def test_config_labels_and_groups():
    assert am.config_of(rec("r", "incident", variant="v", verdicts="cf",
                            modifiers=("verify",))) == am.INC_V
    assert am.group_of(am.HEAL_V) == 1 and am.group_of(am.INC) == 0
    assert am.group_of("selective[cf]+verify") == 0            # detection-driven quarantine
    assert am.group_of(am.HEAL_OR) == 2 and am.group_of(am.NONE) == 3


# --- statistics -----------------------------------------------------------------------------

def units_of(recs):
    samples, _ = am.history_samples(recs)
    return am.build_units(recs, samples=samples)[0]


def test_summary_averages_per_scenario_first(tmp_path):
    units = units_of(main_records())
    c = ctx(tmp_path)
    s = am.summary(units, "asr", c, am.NONE)
    assert s["mean"] == pytest.approx(75.0) and s["n"] == 2 and s["small_sample"]
    assert s["runs"] == 4 and s["records"] == 12
    heal = am.summary(units, "asr", c, am.HEAL)
    # Scenario means 0 (v1, 9 records) and 100 (v2, 3 records): 50, not the record mean 25.
    assert heal["mean"] == pytest.approx(50.0)
    assert 0.0 <= heal["lo"] <= heal["mean"] <= heal["hi"] <= 100.0
    bmp = am.summary(units, "bmp", c, am.HEAL)
    # v1: repeat 2 of each of its 3 runs keeps 1 of 2 probes -> (3 x 50 + 6 x 100) / 9.
    assert bmp["mean"] == pytest.approx(((3 * 50 + 6 * 100) / 9 + 100) / 2)


def test_boot_is_reproducible_and_degenerate_for_equal_values(tmp_path):
    c = ctx(tmp_path)
    assert am.boot([3.0, 3.0, 3.0], c) == (3.0, 3.0, 3.0)
    assert am.boot([0.0, 10.0, 20.0], c) == am.boot([0.0, 10.0, 20.0], c)
    assert am.boot([], c) is None
    # Order-free: paired_scenarios' order follows a set, i.e. Python's hash seed.
    vals = [float(v) for v in range(0, 100, 7)]
    assert am.boot(vals, c) == am.boot(list(reversed(vals)), c) == am.boot(vals[3:] + vals[:3], c)


def test_paired_difference_uses_matching_run_and_repeat(tmp_path):
    units = units_of(main_records())
    d = am.paired(units, "asr", am.HEAL, am.INC, ctx(tmp_path))
    # v1: 0 - 25 = -25 on 9 pairs; v2: 100 - 0 = +100 on 3 pairs -> scenario mean +37.5.
    assert d["diff"] == pytest.approx(37.5) and d["n"] == 2 and d["pairs"] == 12
    assert d["mean_a"] == pytest.approx(50.0) and d["mean_b"] == pytest.approx(12.5)
    # Drop one incident record: that (run, repeat) no longer pairs.
    fewer = [u for u in units if not (u["config"] == am.INC and u["repeat"] == 3
                                      and u["run"] == V2_RUNS[0])]
    assert am.paired(fewer, "asr", am.HEAL, am.INC, ctx(tmp_path))["pairs"] == 11
    assert am.paired(units, "asr", am.HEAL, "nothing", ctx(tmp_path)) is None


def test_direction_and_verdict():
    lower = {"diff": -2, "lo": -3, "hi": -1, "n": 20, "small_sample": False}
    assert am.direction(lower) == -1 and am.verdict(lower, "asr") == "lower (better)"
    assert am.verdict(lower, "bmp") == "lower (worse)"
    span = {**lower, "hi": 1}
    assert am.direction(span) == 0 and am.verdict(span, "asr") == "interval includes 0"
    assert am.verdict(None, "asr") == "not measured"


def _balanced_reference(groups):
    """Textbook balanced nested ANOVA (S scenarios x R runs x n replays)."""
    y = np.array([[v for v in runs.values()] for runs in groups.values()], dtype=float)
    S, R, n = y.shape
    grand, sm, rm = y.mean(), y.mean(axis=(1, 2)), y.mean(axis=2)
    msa = R * n * ((sm - grand) ** 2).sum() / (S - 1)
    msb = n * ((rm - sm[:, None]) ** 2).sum() / (S * (R - 1))
    msw = ((y - rm[..., None]) ** 2).sum() / (S * R * (n - 1))
    return max(0, (msa - msb) / (R * n)), max(0, (msb - msw) / n), msw


def test_nested_components_match_the_balanced_formulas(tmp_path):
    groups = {"s1": {"a": [1, 3], "b": [5, 7]}, "s2": {"a": [10, 12], "b": [20, 22]},
              "s3": {"a": [4, 4], "b": [9, 13]}}
    v = am.nested_components(groups, ctx(tmp_path))
    s, h, e = _balanced_reference(groups)
    assert v["scenario"]["variance"] == pytest.approx(s)
    assert v["history"]["variance"] == pytest.approx(h)
    assert v["replay"]["variance"] == pytest.approx(e)
    total = s + h + e
    assert v["history"]["share"] == pytest.approx(100 * h / total)
    assert v["scenarios"] == 3 and v["runs"] == 6 and v["values"] == 12


def test_nested_components_put_the_variance_where_it_is(tmp_path):
    c = ctx(tmp_path)
    replays_only = {f"s{i}": {"a": [0, 100], "b": [100, 0]} for i in range(5)}
    assert am.nested_components(replays_only, c)["replay"]["share"] == pytest.approx(100)
    scenarios_only = {f"s{i}": {"a": [10.0 * i] * 3, "b": [10.0 * i] * 3} for i in range(5)}
    v = am.nested_components(scenarios_only, c)
    assert v["scenario"]["share"] == pytest.approx(100) and v["replay"]["share"] == 0
    # No replicate at some level: not estimable.
    assert am.nested_components({"s1": {"a": [1, 2]}, "s2": {"a": [3, 4]}}, c) is None
    assert am.nested_components({"s1": {"a": [1], "b": [2]}, "s2": {"a": [3], "b": [4]}}, c) is None


def test_single_replay_disagreement(tmp_path):
    rows = []
    for rp, out in ((1, (A, C, C, C)), (2, (C,) * 4), (3, (C,) * 4)):     # majority: clean
        rows.append(rec(V1_RUNS[0], "incident", verdicts="cf", variant="v1", repeat=rp, outcomes=out))
    for rp in (1, 2):                                                     # only two replays
        rows.append(rec(V1_RUNS[1], "incident", verdicts="cf", variant="v1", repeat=rp))
    d = am.disagreement(units_of(rows), am.INC, ctx(tmp_path))
    assert d["runs"] == 1 and d["disagreeing"] == 1 and d["replays"] == 3 and d["runs_split"] == 1
    assert d["share"]["mean"] == pytest.approx(100 / 3)


# --- joins ---------------------------------------------------------------------------------

def test_preservation_join_by_copy_then_identity():
    r1 = rec(V1_RUNS[0], "heal", verdicts="cf", variant="v1", repeat=1,
             work_db="data/runs/x/main/heal@full@w6@stop.db")
    r2 = rec(V1_RUNS[0], "heal", verdicts="cf", variant="v1", repeat=2)       # no work_db
    r3 = rec(V1_RUNS[0], "heal", verdicts="cf", variant="v1", repeat=3)
    rows = [{"work_db": "data/runs/x/main/heal@full@w6@stop.db", "status": "ok",
             "benign": 4, "benign_kept": 3, "bcp": 0.75},
            {**{k: r2[k] for k in ("run_dir", "mode", "verdicts", "modifiers", "source",
                                   "checkpoint", "condition", "family", "repeat", "cf_workers")},
             "status": "ok", "benign": 0, "benign_kept": 0, "bcp": None},
            {**{k: r3[k] for k in ("run_dir", "mode", "verdicts", "modifiers", "source",
                                   "checkpoint", "condition", "family", "repeat", "cf_workers")},
             "status": "missing_copy"}]
    p = am.Preservation(rows)
    assert p.bcp(r1) == (75.0, "ok")
    assert p.bcp(r2) == (None, "no benign claims")
    assert p.bcp(r3) == (None, "missing_copy")
    assert p.bcp({**r3, "repeat": 4}) == (None, "no row")


def make_run_db(path: Path, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE experiment_runs (condition TEXT, model TEXT, config_json TEXT)")
    conn.execute("INSERT INTO experiment_runs VALUES ('poisoned', ?, '{}')", (model,))
    conn.commit()
    conn.close()


def test_model_index_reads_run_db_read_only(tmp_path):
    make_run_db(tmp_path / "runs" / "1450_poisoned_v1_evolving" / "run.db", "gpt-5.5")
    idx = am.ModelIndex(tmp_path / "runs")
    assert idx("1450_poisoned_v1_evolving") == "gpt-5.5"
    assert idx("1451_missing") is None


def test_suite_labels(monkeypatch):
    monkeypatch.setattr(am, "LONGMEMEVAL", {"lme_case": None})
    monkeypatch.setattr(am, "MEMBENCH", {"mb_case": None})
    models = lambda run: "gpt-4.1-mini"                           # noqa: E731
    assert am.suite_label("external", {"variant": "lme_case"}, models) == "LongMemEval"
    assert am.suite_label("external", {"variant": "mb_case"}, models) == "MemBench"
    assert am.suite_label("models", {"run_dir": "x"}, models) == "subset gpt-4.1-mini"
    assert am.suite_label("mem0_additive", {}, models) == "Mem0 additive"


# --- end to end -----------------------------------------------------------------------------

def fixture_inputs(tmp_path: Path) -> list[str]:
    res = tmp_path / "results"
    write(res / "main" / "records.jsonl", main_records())
    clean = []
    for run, variant in (("0962_clean_v1_evolving", "v1"), ("0963_clean_v2_evolving", "v2")):
        clean.append(rec(run, "none", variant=variant, condition="clean"))
        clean.append(rec(run, "incident", verdicts="cf", modifiers=("verify",), variant=variant,
                         condition="clean", n_quarantined=3, probes=(True, False)))
        clean.append(rec(run, "heal", verdicts="cf", modifiers=("verify",), variant=variant,
                         condition="clean", n_quarantined=3, n_healed=2))
        clean.append(rec(run, "incident", verdicts="cf", modifiers=("verify",), variant=variant,
                         condition="clean", verify_n=1, n_quarantined=9))      # left out
    write(res / "main" / "clean_control.jsonl", clean)
    rein = []
    for run, variant in ((V1_RUNS[0], "v1"), (V2_RUNS[0], "v2")):
        for mode, verdicts, carry in (("heal", "cf", 1), ("selective", "oracle", 0)):
            for cp in (0, 1, 3):
                rein.append(rec(run, mode, verdicts=verdicts, modifiers=("immunize",),
                                variant=variant, checkpoint=cp,
                                outcomes=(A,) * 4 if cp and mode == "heal" else (C,) * 4,
                                residual_attack=carry,
                                notes=f"reinfect: poison_writes=1 blocked=0 "
                                      f"active_naming_attacker={carry}"))
    write(res / "main" / "reinfect.jsonl", rein)
    flat = [rec(r.replace("_evolving", ""), r_mode, variant=v, verdicts=vd, outcomes=out)
            for r, v in (("1198_poisoned_v1_evolving", "v1"), ("1199_poisoned_v2_evolving", "v2"))
            for r_mode, vd, out in (("none", None, (A,) * 4), ("incident", "cf", (A, C, C, C)),
                                    ("heal", "cf", (C,) * 4))]
    write(res / "flat" / "main.jsonl", flat)
    models = []
    for run, model in (("1450_poisoned_v1_evolving", "gpt-5.5"), ("1451_poisoned_v1_evolving", "gpt-5.4")):
        make_run_db(tmp_path / "runs" / run / "run.db", model)
        models += [rec(run, "none", variant="v1", outcomes=(A,) * 4),
                   rec(run, "incident", verdicts="cf", variant="v1"),
                   rec(run, "heal", verdicts="cf", variant="v1")]
    write(res / "models" / "main.jsonl", models)
    return ["--main", str(res / "main" / "records.jsonl"),
            "--clean", str(res / "main" / "clean_control.jsonl"),
            "--reinfect", str(res / "main" / "reinfect.jsonl"),
            "--flat", str(res / "flat" / "main.jsonl"),
            "--models", str(res / "models" / "main.jsonl"),
            "--adaptive", str(res / "adaptive" / "main.jsonl"),          # missing
            "--external", str(res / "external" / "main.jsonl"),          # missing
            "--mem0", str(res / "mem0" / "main.jsonl"),
            "--mem0-additive", str(res / "mem0_additive" / "main.jsonl"),
            "--preservation", str(res / "preservation_final" / "preservation.jsonl"),
            "--runs", str(tmp_path / "runs"), "--n-boot", str(NB)]


def test_main_end_to_end_with_figures(tmp_path):
    out = tmp_path / "final"
    result = am.main([*fixture_inputs(tmp_path), "--out", str(out)])
    md = (out / "main_tables.md").read_text()
    for title in ("1. Main trade-off", "2. Healing vs quarantine", "3. Per attack category",
                  "4. Repetition", "5. Clean false alarms", "6. Reinfection", "7. Cost",
                  "8. Cross-suite"):
        assert title in md
    assert "small sample" in md and "**Takeaway:**" in md
    secs = result["sections"]
    assert all(s["skipped"] is None for s in secs.values()), {k: s["skipped"] for k, s in secs.items()}
    assert secs["tradeoff"]["numbers"]["configs"]["none"]["asr"]["mean"] == 75.0
    assert secs["heal_vs_quarantine"]["numbers"]["asr"]["heal[cf] - incident[cf]"]["diff"] == 37.5
    # Clean: the verify_n=1 record is left out; heal removes 1 net memory, incident 3.
    assert result["inputs"]["clean"]["dropped"] == {"verify_n=1": 2}
    clean = secs["clean"]["numbers"]["paired"]["heal[cf]+verify - incident[cf]+verify"]
    assert clean["removed"]["diff"] == -2.0 and clean["bmp"]["diff"] == 50.0
    # Reinfection: heal's store names the attacker at every checkpoint, selective's never.
    rein = secs["reinfection"]["numbers"]["configs"]
    assert rein["heal[cf]+immunize"]["carrying"]["3"]["mean"] == 100.0
    assert rein["selective[oracle]+immunize"]["carrying"]["3"]["mean"] == 0.0
    assert rein["heal[cf]+immunize"]["asr"]["1"]["mean"] == 100.0
    # Cross-suite: the model subset is split by run.db's model; missing suites are absent.
    suites = secs["suites"]["numbers"]["configs"]
    assert {"dense evolving", "flat", "subset gpt-5.5", "subset gpt-5.4"} == set(suites)
    assert suites["flat"]["incident[cf]"]["asr"]["mean"] == 25.0
    # Repetition: heal's claim-level replays only on repeat 1; replay-level ASR by repeat.
    assert secs["cost"]["numbers"]["heal_replays"]["heal[cf]"]["claim_cf_replays"]["mean"] == 2.0
    rep = secs["repetition"]["numbers"]["by_replay"]["none"]
    assert [rep[str(r)]["mean"] for r in (1, 2, 3)] == [75.0, 75.0, 75.0]
    assert result["inputs"]["adaptive"]["state"] == "missing"
    assert result["preservation"]["files"][0]["state"] == "missing"
    pngs = sorted(p.name for p in out.glob("main_*.png"))
    for name in ("main_tradeoff_bmp.png", "main_tradeoff_bcp.png", "main_heal_vs_quarantine.png",
                 "main_category.png", "main_domain.png", "main_repeats.png", "main_variance.png",
                 "main_clean.png", "main_reinfection.png", "main_cost.png",
                 "main_heal_replays.png", "main_suites.png"):
        assert name in pngs
    json.loads((out / "main.json").read_text())


IDENTITY_FIELDS = ("run_dir", "mode", "verdicts", "modifiers", "source", "checkpoint",
                   "condition", "family", "repeat", "cf_workers", "cf_method", "cf_groups",
                   "stop_after_action", "verify_n")


def bcp_row(r: dict, records: str, kept: int, benign: int = 4) -> dict:
    """A bcp.py output row for record `r`, written for records file `records`."""
    return {**{k: r[k] for k in IDENTITY_FIELDS if k in r}, "records": records,
            "experiment": Path(records).parent.name, "status": "ok", "benign": benign,
            "benign_kept": kept}


def test_main_with_bcp_rows(tmp_path):
    args = fixture_inputs(tmp_path)
    recs = main_records()
    main_file = str(tmp_path / "results" / "main" / "records.jsonl")
    rows = [bcp_row(r, main_file, {"none": 4, "incident": 1, "heal": 3}[r["mode"]]) for r in recs]
    # Rows another experiment's records file wrote for the model-subset runs, which have
    # no row of their own: the same identity, but not these records' trial.
    models = am.read_jsonl(tmp_path / "results" / "models" / "main.jsonl")[0]
    rows += [bcp_row(r, "results/families/records.jsonl", 0) for r in models]
    write(tmp_path / "results" / "preservation_final" / "preservation.jsonl", rows)
    out = tmp_path / "final"
    result = am.main([*args, "--out", str(out), "--no-figures"])
    cfg = result["sections"]["tradeoff"]["numbers"]["configs"]
    assert cfg["heal[cf]"]["bcp"]["mean"] == 75.0 and cfg["incident[cf]"]["bcp"]["mean"] == 25.0
    d = result["sections"]["heal_vs_quarantine"]["numbers"]["bcp"]["heal[cf] - incident[cf]"]
    assert d["diff"] == 50.0 and d["lo"] == 50.0 and d["hi"] == 50.0
    status = result["preservation"]["status"]
    assert status["main"] == {"ok": len(recs)}
    assert status["subset gpt-5.5"] == {"no row for this file": 3}
    assert status["subset gpt-5.4"] == {"no row for this file": 3}
    assert "no row for this file: 3" in (out / "main_tables.md").read_text()


def test_bcp_fallback_never_takes_another_records_files_row():
    """results/families/records.jsonl repeats the main suite's identity (sample-1 run,
    configuration, family=full, repeat) without a work_db: its row must not fill a main
    record's BCP."""
    main_rec = rec(V1_RUNS[0], "ancestry", variant="v1", repeat=1,
                   work_db="data/runs/x/main/ancestry@full@stop.db")
    fam_rec = {k: v for k, v in main_rec.items() if k != "work_db"}
    assert am.Preservation.key(fam_rec) == am.Preservation.key(main_rec)
    families = bcp_row(fam_rec, "results/families/records.jsonl", kept=1)
    p = am.Preservation([families])
    assert p.bcp(main_rec, "results/main/records.jsonl") == (None, "no row for this file")
    assert p.bcp(main_rec) == (None, "no row for this file")          # the record names no file
    assert p.bcp(fam_rec, "results/families/records.jsonl") == (25.0, "ok")
    # The record's own file's row is used, wherever the two name the file from.
    own = bcp_row(main_rec, "results/main/records.jsonl", kept=3)
    p = am.Preservation([own, families])
    assert p.bcp(main_rec, "results/main/records.jsonl") == (75.0, "ok")
    assert p.bcp(main_rec, str(am.ROOT / "results" / "main" / "records.jsonl")) == (75.0, "ok")
    # The recovered copy is still the first join: it names the measured file itself.
    by_copy = {**bcp_row(main_rec, "results/other.jsonl", kept=2), "work_db": main_rec["work_db"]}
    assert am.Preservation([by_copy, families]).bcp(main_rec, "results/main/records.jsonl") == (50.0, "ok")


# --- matched measurements -------------------------------------------------------------------

# Three history runs per scenario; heal so far covers sample 1 only, where incident's ASR
# (50%) is higher than on samples 2-3 (0%). Unmatched, heal (25%) looks worse than
# incident (16.7% = (3 x 50 + 6 x 0) / 9); on the measurements both have it is better.
U_RUNS = {"v1": ("0960_poisoned_v1_evolving", "1500_poisoned_v1_evolving", "1600_poisoned_v1_evolving"),
          "v2": ("0961_poisoned_v2_evolving", "1501_poisoned_v2_evolving", "1601_poisoned_v2_evolving")}
U_KIND = {"v1": ("direct", "office"), "v2": ("composed", "travel")}


def uneven_records() -> list[dict]:
    out = []
    for variant, runs in U_RUNS.items():
        cat, dom = U_KIND[variant]
        for i, run in enumerate(runs):
            first = i == 0
            for rp in (1, 2, 3):
                kw = dict(variant=variant, repeat=rp, category=cat, domain=dom)
                out.append(rec(run, "none", outcomes=(A,) * 4, **kw))
                out.append(rec(run, "incident", verdicts="cf",
                               outcomes=(A, A, C, C) if first else (C,) * 4,
                               llm_calls=30 if first else 10, seconds=60.0 if first else 6.0, **kw))
                if first:
                    out.append(rec(run, "heal", verdicts="cf", outcomes=(A, C, C, C),
                                   llm_calls=40, seconds=80.0, claim_cf_replays=4, **kw))
    return out


def uneven_inputs(tmp_path: Path) -> list[str]:
    res = tmp_path / "results"
    write(res / "main" / "records.jsonl", uneven_records())
    clean = []
    for run, variant, heal_too, nq in (("0962_clean_v1_evolving", "v1", True, 3),
                                       ("0963_clean_v2_evolving", "v2", True, 3),
                                       ("1502_clean_v1_evolving", "v1", False, 9)):
        clean.append(rec(run, "none", variant=variant, condition="clean"))
        clean.append(rec(run, "incident", verdicts="cf", modifiers=("verify",), variant=variant,
                         condition="clean", n_quarantined=nq))
        if heal_too:
            clean.append(rec(run, "heal", verdicts="cf", modifiers=("verify",), variant=variant,
                             condition="clean", n_quarantined=3, n_healed=2))
    write(res / "main" / "clean_control.jsonl", clean)
    rein = []
    for run, variant in ((U_RUNS["v1"][0], "v1"), (U_RUNS["v2"][0], "v2")):
        for mode, verdicts in (("heal", "cf"), ("selective", "oracle")):
            if mode == "heal" and variant == "v2":
                continue                                 # heal followed only v1's run
            carry = 1 if variant == "v2" else 0
            for cp in (0, 1, 3):
                rein.append(rec(run, mode, verdicts=verdicts, modifiers=("immunize",),
                                variant=variant, checkpoint=cp, residual_attack=carry,
                                outcomes=(A,) * 4 if carry else (C,) * 4,
                                notes=f"reinfect: poison_writes=1 blocked=0 "
                                      f"active_naming_attacker={carry}"))
    write(res / "main" / "reinfect.jsonl", rein)
    missing = tmp_path / "missing"
    return ["--main", str(res / "main" / "records.jsonl"),
            "--clean", str(res / "main" / "clean_control.jsonl"),
            "--reinfect", str(res / "main" / "reinfect.jsonl"),
            *[x for n in ("flat", "adaptive", "models", "external", "mem0", "mem0-additive")
              for x in (f"--{n}", str(missing / f"{n}.jsonl"))],
            "--preservation", str(missing / "p.jsonl"), "--runs", str(missing),
            "--n-boot", str(NB)]


def test_matched_basis_requires_every_configuration_that_has_the_metric(tmp_path):
    units = units_of(uneven_records())
    b = am.matched_basis(units, [am.NONE, am.INC, am.HEAL], "asr")
    assert len(b.keys) == 6 and b.scenarios == 2                  # 2 sample-1 runs x 3 replays
    assert b.uneven and b.limiting == [am.HEAL]
    assert b.per_config == {am.NONE: 18, am.INC: 18, am.HEAL: 6}
    assert "limited by heal[cf]" in b.describe()
    # A configuration with no value of the metric is left out of the requirement.
    for u in units:
        if u["config"] == am.HEAL:
            u["bcp"] = None
        else:
            u["bcp"] = 50.0
    b = am.matched_basis(units, [am.NONE, am.INC, am.HEAL], "bcp")
    assert len(b.keys) == 18 and not b.uneven and am.HEAL not in b.per_config
    # Even coverage: the matched numbers are every measurement's.
    even = units_of(main_records())
    st, bases = am.matched(even, [am.NONE, am.INC, am.HEAL], ["asr"], ctx(tmp_path))
    assert not bases["asr"].uneven
    assert st[am.HEAL]["asr"] == am.summary(even, "asr", ctx(tmp_path), am.HEAL)


def capture(monkeypatch) -> dict:
    """Replace the plot functions with ones that record what they were given."""
    seen: dict = {}

    def keep(name):
        def fake(path, *args, **kwargs):
            seen[Path(path).name] = (args, kwargs)
            return Path(path)
        return fake

    for fn in ("plot_tradeoff", "plot_forest", "plot_grid", "plot_series_rows", "plot_variance",
               "plot_reinfect", "plot_heal_replays"):
        monkeypatch.setattr(am, fn, keep(fn))
    return seen


def series_stats(seen: dict, name: str, panel: int, series: int = 0) -> dict:
    """The {row: stat} of one series of a plot_series_rows figure."""
    args, _ = seen[name]
    return args[1][panel][1][series]["stats"]


def test_uneven_coverage_tables_and_figures_compare_matched_measurements(tmp_path, monkeypatch):
    seen = capture(monkeypatch)
    result = am.main([*uneven_inputs(tmp_path), "--out", str(tmp_path / "final")])
    secs = result["sections"]
    assert all(s["skipped"] is None for s in secs.values()), {k: s["skipped"] for k, s in secs.items()}

    # Section 1: the headline numbers and the trade-off figure stand on heal's sample-1
    # measurements, where incident is at 50%; the coverage-dependent 16.7% is secondary.
    s1 = secs["tradeoff"]["numbers"]
    assert s1["uneven"] and s1["basis"]["asr"]["measurements"] == 6
    assert s1["configs"]["incident[cf]"]["asr"]["mean"] == 50.0
    assert s1["configs"]["heal[cf]"]["asr"]["mean"] == 25.0
    assert s1["all_measurements"]["incident[cf]"]["asr"]["mean"] == pytest.approx(50 / 3, abs=1e-3)
    for fig in ("main_tradeoff_bmp.png", "main_tradeoff_bcp.png"):
        args, _ = seen[fig]
        stats = args[0]
        assert stats["incident[cf]"][1]["mean"] == 50.0 and stats["heal[cf]"][1]["mean"] == 25.0
    assert seen["main_tradeoff_bmp.png"][0][2].startswith("matched")
    md = (tmp_path / "final" / "main_tables.md").read_text()
    assert "Coverage-dependent view" in md and "limited by heal[cf]" in md
    # The paired difference agrees in sign with the matched figure.
    assert secs["heal_vs_quarantine"]["numbers"]["asr"]["heal[cf] - incident[cf]"]["diff"] == -25.0

    # Section 3: matched within each category (each holds one scenario here).
    args, _ = seen["main_category.png"]
    grid = args[2]
    assert grid["asr"]["direct"]["incident[cf]"]["mean"] == 50.0
    assert grid["asr"]["composed"]["heal[cf]"]["mean"] == 25.0
    assert "matched" in args[4]
    s3 = secs["breakdown"]["numbers"]
    assert s3["all_measurements"]["by_category"]["asr"]["direct"]["incident[cf]"]["mean"] == pytest.approx(50 / 3, abs=1e-3)

    # Section 7: cost on the same measurements (incident's sample-1 recoveries cost 30 calls).
    cost = series_stats(seen, "main_cost.png", 0)
    assert cost["incident[cf]"]["mean"] == 30.0 and cost["heal[cf]"]["mean"] == 40.0
    assert secs["cost"]["numbers"]["all_measurements"]["incident[cf]"]["llm_calls"]["mean"] == pytest.approx(50 / 3, abs=1e-3)
    assert seen["main_cost.png"][0][2].startswith("matched")

    # Section 8: the dense evolving row on the measurements none, incident and heal share.
    asr = {s["name"]: s["stats"] for s in seen["main_suites.png"][0][1][0][1]}
    assert asr["incident[cf]"]["dense evolving"]["mean"] == 50.0
    assert asr["heal[cf]"]["dense evolving"]["mean"] == 25.0
    assert secs["suites"]["numbers"]["basis"]["dense evolving"]["asr"]["measurements"] == 6

    # Section 5: the clean run heal has not reached (incident removed 9 there) is left out.
    removed = series_stats(seen, "main_clean.png", 0)
    assert removed["incident[cf]+verify"]["mean"] == 3.0
    clean = secs["clean"]["numbers"]
    assert clean["all_measurements"]["incident[cf]+verify"]["removed"]["mean"] == 4.5   # v1 (3+9)/2, v2 3

    # Section 6: every point on the one run both configurations followed.
    args, kwargs = seen["main_reinfection.png"]
    st = args[2]
    assert st["selective[oracle]+immunize"]["carrying"][3]["mean"] == 0.0
    assert secs["reinfection"]["numbers"]["all_measurements"]["selective[oracle]+immunize"]["carrying"]["3"]["mean"] == 50.0
    assert kwargs["note"].startswith("matched")


def test_footer_grows_the_figure_for_every_note_line():
    plt = am._plt()
    heights = []
    for note in ("one line", "one line\nand a second", "x " * 200):
        fig, ax = plt.subplots(figsize=(am.WIDTH, 3.0))
        bottom = am._footer(fig, [am._hollow_handle()], ["hollow"], 1, note)
        heights.append(fig.get_figheight())
        assert 0 < bottom < 0.5
        plt.close(fig)
    assert heights[0] < heights[1] and heights[0] < heights[2]       # wrapped lines count too


def test_main_with_every_input_missing(tmp_path):
    missing = tmp_path / "none"
    args = [f"--{n.replace('_', '-')}" if i % 2 == 0 else str(missing / f"{n}.jsonl")
            for n in am.INPUTS for i in (0, 1)]
    result = am.main([*args, "--preservation", str(missing / "p.jsonl"), "--runs",
                      str(missing), "--out", str(tmp_path / "final"), "--n-boot", "50"])
    assert all(s["skipped"] for s in result["sections"].values())
    assert "Skipped" in (tmp_path / "final" / "main_tables.md").read_text()
