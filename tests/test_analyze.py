"""analyze.py (plan item T10): tables and figures over the experiment records files.

Every input is a tiny synthetic records file or run folder written to a temp directory,
in recover.py's record schema, so these tests pin the arithmetic offline: no run.db, no
model, no socket. Expected numbers are worked out by hand from the fixture below.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

import analyze
import measure
from memheal.heal import CLAIM_N, DDMIN_BUDGET, RULES

STOP = "replays stop at the first acting call"
R1 = "0010_poisoned_office_a_dense_direct_evolving"
R2 = "0011_poisoned_office_b_dense_direct_evolving"
R3 = "0012_poisoned_office_c_dense_direct_evolving"
C1 = "0013_clean_office_a_dense_direct_evolving"
FL1 = "0020_poisoned_office_a_dense_direct"
FL2 = "0021_poisoned_office_b_dense_direct"
QS = ("Which cost centre?", "Who approves?")


def rec(run, mode, *, verdicts=None, modifiers=(), outcomes=("correct",), probes=(True, True),
        condition="poisoned", checkpoint=0, incident=True, **extra) -> dict:
    variant = run.split("_", 2)[2].removesuffix("_evolving")
    r = {"run_dir": run, "variant": variant, "domain": "office", "category": "direct",
         "condition": condition, "mode": mode, "verdicts": verdicts,
         "modifiers": list(modifiers), "source": "patched", "checkpoint": checkpoint,
         "task_outcomes": list(outcomes), "task_destinations": [None] * len(outcomes),
         "asr": sum(o == "attacker" for o in outcomes) / len(outcomes), "br": 1.0,
         "probes": [{"question": q, "expected": "x", "correct": c} for q, c in zip(QS, probes)],
         "n_quarantined": 0, "n_restored": 0, "n_reconstructed": 0,
         "collateral": {"harmful": 0, "mixed": 0, "benign": 0, "ambiguous": 0, "non_lineage": 0},
         "residual_attack": 0, "incident": incident, "root_was_consolidated": False,
         "verify_rounds": None, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
         "seconds": 1.0, "notes": "", "n_claim_repaired": 0}
    r.update(extra)
    return r


def fam(run, mode, repeat, outcomes, **kw) -> dict:
    return rec(run, mode, outcomes=outcomes, family="full", repeat=repeat, k=5, depth=0,
               observe="logged", **kw)


A, C = "attacker", "correct"


def families() -> list[dict]:
    out = []
    for rp, f1 in ((1, (A, A, C, C)), (2, (A, A, C, C)), (3, (A, C, C, C))):
        out.append(fam(R1, "none", rp, f1))
        out.append(fam(R2, "none", rp, (C, C, C, C), incident=False))
        out.append(fam(R1, "selective", rp, (C, C, C, C), verdicts="oracle", probes=(True, False)))
        out.append(fam(R2, "selective", rp, (C, C, C, C), verdicts="oracle", incident=False))
        out.append(fam(R1, "incident", rp, (C, C, C, A) if rp == 1 else (C, C, C, C),
                       verdicts="cf", cf_method="frozen", cf_groups=False, cf_workers=1,
                       culprit_sets=[["m1"]] if rp < 3 else []))
        out.append(fam(R2, "incident", rp, (C, C, C, C), verdicts="cf", incident=False,
                       cf_method="frozen", cf_groups=False, cf_workers=1, culprit_sets=[]))
    return out


def canonical() -> list[dict]:
    inc = "incident: inferred_roots=1 replays=6"
    out = [
        rec(R1, "none", outcomes=(A,)), rec(R2, "none", outcomes=(A,)),
        rec(R3, "none", outcomes=(C,), incident=False),
        *[rec(r, "ancestry", n_quarantined=5) for r in (R1, R2, R3)],
        rec(R1, "selective", verdicts="cf", outcomes=(A,)), rec(R2, "selective", verdicts="cf"),
        rec(R3, "selective", verdicts="cf", incident=False),
        rec(R1, "incident", verdicts="cf", notes=inc), rec(R2, "incident", verdicts="cf", notes=inc),
        rec(R3, "incident", verdicts="cf", incident=False,
            notes="incident: no harmful action observed"),
        rec(R1, "incident", verdicts="cf", modifiers=("verify",), seconds=20.0, llm_calls=10,
            notes="incident: inferred_roots=1 replays=8"),
        rec(R2, "incident", verdicts="cf", modifiers=("verify",), seconds=30.0, llm_calls=12,
            notes="incident: inferred_roots=1 replays=8"),
        rec(R1, "root", outcomes=(A,)), rec(R2, "root", outcomes=(A,)),
        # written twice on one deterministic store: replay noise (the last copy is kept)
        rec(R1, "selective", verdicts="oracle", outcomes=(A,)),
        rec(R1, "selective", verdicts="oracle"), rec(R2, "selective", verdicts="oracle"),
        *[rec(R1, "selective", verdicts="oracle", modifiers=("immunize",), checkpoint=cp,
              outcomes=(A,) if cp == 1 else (C,)) for cp in measure.CHECKPOINTS],
        rec(C1, "none", condition="clean", incident=False),
        rec(C1, "incident", verdicts="cf", modifiers=("verify",), condition="clean",
            incident=False, n_quarantined=2, probes=(True, False),
            notes="control: simulated false alarm on driveline.example"),
    ]
    return out


def k_ablation() -> list[dict]:
    fresh = {"observe": "fresh", "family": "recorded", "depth": 0}
    out = []
    for k, outs in ((3, (C, A, C)), (10, (A, A, A))):
        for run, o in zip((R1, R2, R3), outs):
            out.append(rec(run, "none", outcomes=(o,), k=k, **fresh))
        for run, hit in ((R1, True), (R2, True), (R3, False)):
            out.append(rec(run, "incident", verdicts="cf", k=k, incident=hit, cf_method="frozen",
                           cf_replays=(k + 1 if hit else 0), cf_seconds=1.0, **fresh))
    return out


def depth_ablation() -> list[dict]:
    out = []
    for d in (1, 2, 3):
        for run in (R1, R2, R3):
            out.append(rec(run, "none", outcomes=(A,), depth=d, observe="fresh", residual_attack=1))
            out.append(rec(run, "selective", verdicts="oracle", depth=d, observe="fresh",
                           n_quarantined=4, collateral={"benign": d, "harmful": 3}))
    return out


def reretrieve() -> list[dict]:
    kw = {"cf_method": "reretrieve", "cf_groups": False, "cf_workers": 1}
    return [rec(R1, "incident", verdicts="cf", culprit_sets=[["m1"]], cf_replays=5,
                cf_seconds=9.0, seconds=10.0, **kw),
            rec(R2, "incident", verdicts="cf", culprit_sets=[["m9"]], cf_replays=7,
                cf_seconds=9.0, seconds=10.0, **kw),
            rec(R3, "incident", verdicts="cf", incident=False, culprit_sets=[], cf_replays=0, **kw)]


def groups() -> list[dict]:
    kw = {"verdicts": "cf", "modifiers": ("verify",), "cf_method": "frozen", "cf_groups": True,
          "cf_workers": 6}
    return [rec(R1, "incident", culprit_sets=[["m1"]], cf_replays=12, cf_seconds=8.0,
                seconds=10.0, llm_calls=30, notes=STOP, **kw),
            rec(R2, "incident", culprit_sets=[["m2", "m3"]], cf_replays=20, cf_seconds=9.0,
                seconds=12.0, llm_calls=40, notes=STOP, **kw),
            rec(R3, "incident", incident=False, culprit_sets=[], cf_replays=0, notes=STOP, **kw)]


def groups_clean() -> list[dict]:
    return [rec(C1, "none", condition="clean", incident=False, notes=STOP),
            rec(C1, "incident", verdicts="cf", modifiers=("verify",), condition="clean",
                incident=False, cf_groups=True, cf_workers=6, culprit_sets=[["m5"]],
                n_quarantined=2, probes=(True, False),
                notes=f"{STOP}; control: simulated false alarm on driveline.example")]


def claims() -> list[dict]:
    out = []
    for run in (R1, R2):
        out += [rec(run, "selective", verdicts="oracle"),
                rec(run, "selective_recon", verdicts="oracle", llm_calls=1),
                rec(run, "selective_claims", verdicts="oracle", n_claim_repaired=2, llm_calls=2,
                    probes=(True, False)),
                rec(run, "selective", verdicts="cf"),
                rec(run, "selective_claims", verdicts="cf", n_claim_repaired=1)]
    return out


def reinfect() -> list[dict]:
    return [rec(R1, "selective_claims", verdicts="oracle", modifiers=("immunize",), checkpoint=cp)
            for cp in measure.CHECKPOINTS]


def flat() -> list[dict]:
    return [rec(FL1, "none", outcomes=(A,)), rec(FL2, "none", outcomes=(A,)),
            rec(FL1, "root"), rec(FL2, "root"),
            rec(FL1, "selective", verdicts="oracle"), rec(FL2, "selective", verdicts="oracle")]


HEAL_FIELDS = {"n_healed": 0, "n_heal_quarantined": 0, "claims_kept": 0, "claims_dropped": 0,
               "claim_cf_replays": 0}


def heal_rec(run, mode, outcomes, *, incident=True, **extra) -> dict:
    """A heal-sweep record (default knobs, as recover.py writes them for a plain sweep)."""
    verdicts = {"none": None, "heal_oracle": "oracle"}.get(mode, "cf")
    kw = {**HEAL_FIELDS, "family": "recorded", "k": 5, "repeat": None, "depth": 0,
          "observe": "logged", "stop_after_action": False}
    if verdicts:
        kw.update(cf_method="frozen", cf_groups=False, cf_workers=1)
    return rec(run, mode, verdicts=verdicts, outcomes=outcomes, incident=incident,
               **{**kw, **extra})


def heal() -> list[dict]:
    """R1 and R2 are incidents, R3 is not. heal clears R1 but not R2; incident and
    heal_oracle clear both. The heal sweep re-ran incident itself, so its incident[cf]
    replaces the canonical one on every run; a repeat-2 heal on R1 is not the replay used."""
    out = []
    for run, hit in ((R1, True), (R2, True), (R3, False)):
        base = {"incident": hit}
        out.append(heal_rec(run, "none", (A,) if hit else (C,), **base))
        out.append(heal_rec(run, "heal", (A,) if run == R2 else (C,), **base,
                            n_healed={R1: 2, R2: 1, R3: 0}[run],
                            n_heal_quarantined=1 if run == R1 else 0,
                            claims_kept={R1: 3, R2: 3, R3: 0}[run],
                            claims_dropped={R1: 2, R2: 1, R3: 0}[run],
                            claim_cf_replays={R1: 4, R2: 2, R3: 0}[run],
                            n_quarantined=2 if hit else 0, residual_attack=1 if run == R2 else 0,
                            llm_calls=8, seconds=30.0, cf_replays=6 if hit else 0))
        out.append(heal_rec(run, "heal_oracle", (C,), **base, n_healed=2 if hit else 0,
                            cf_replays=6 if hit else 0))
        out.append(heal_rec(run, "incident", (C,), **base, n_quarantined=3 if hit else 0,
                            probes=(True, False) if hit else (True, True),
                            cf_replays=6 if hit else 0))
    out.append(heal_rec(R1, "heal", (A,), repeat=2))
    return out


def pres_row(path: Path, r: dict, benign: int, benign_kept: int, attack: int,
             attack_kept: int, status: str = "ok") -> dict:
    """bcp.py's row for record `r` of records file `path` (only the fields analyze reads)."""
    knob = {k: measure.knob_value(r, k) for k in measure.KNOBS}
    return {"experiment": path.stem, "records": str(path), "run_dir": r["run_dir"],
            "condition": "poisoned", "mode": r["mode"], "verdicts": r["verdicts"],
            "modifiers": r["modifiers"], "source": "patched", "checkpoint": 0, **knob,
            "status": status, "benign": benign, "benign_kept": benign_kept,
            "attack": attack, "attack_kept": attack_kept}


def preservation(heal_path: Path, recovery_path: Path, claims_path: Path) -> list[dict]:
    """BCP: heal keeps 8/10 and 5/6 benign claims on the incident runs, incident 4/10 and
    2/6, heal_oracle 9/10 and 6/6; ACR: heal removes 2/2 and 1/2, the others all. R3
    (no incident) loses nothing and removes nothing. The canonical incident[cf] row of R1
    (another copy, 10/10) must not be read: the heal sweep's incident is the one used."""
    hs = {(r["run_dir"], r["mode"]): r for r in heal() if not r.get("repeat")}
    numbers = {"heal": {R1: (10, 8, 2, 0), R2: (6, 5, 2, 1)},
               "incident": {R1: (10, 4, 2, 0), R2: (6, 2, 2, 0)},
               "heal_oracle": {R1: (10, 9, 2, 0), R2: (6, 6, 2, 0)}}
    out = [pres_row(heal_path, hs[(rd, mode)], *v) for mode, by in numbers.items()
           for rd, v in by.items()]
    out += [pres_row(heal_path, hs[(R3, mode)], 3, 3, 1, 1) for mode in numbers]
    out.append(pres_row(recovery_path, rec(R1, "incident", verdicts="cf"), 10, 10, 2, 0))
    sc = [r for r in claims() if r["mode"] == "selective_claims" and r["verdicts"] == "oracle"]
    out.append(pres_row(claims_path, sc[0], 4, 3, 1, 0))
    out.append(pres_row(claims_path, sc[1], 4, 0, 1, 0, status="copy_mismatch"))
    return out


def write(path: Path, recs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def history(runs: Path) -> None:
    """Evolving 0010-0019 (suffix), flat 0020-0039: poisoned a/b attacker on evolving,
    only a on flat; clean all correct."""
    outcomes = {R1: "attacker", R2: "attacker", C1: "correct", FL1: "attacker",
                FL2: "correct", "0014_clean_office_b_dense_direct_evolving": "correct",
                "0022_clean_office_a_dense_direct": "correct",
                "0023_clean_office_b_dense_direct": "correct", R3: "correct"}
    for name, outcome in outcomes.items():
        (runs / name).mkdir(parents=True)
        (runs / name / "summary.json").write_text(json.dumps({"outcome": outcome, "k": 5}))


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise OSError("network access blocked in test")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def world(tmp_path) -> list[str]:
    files = {"families": families(), "recovery": canonical(), "k": k_ablation(),
             "depth": depth_ablation(), "reretrieve": reretrieve(), "groups": groups(),
             "groups_clean": groups_clean(), "claims": claims(), "reinfect": reinfect(),
             "flat": flat(), "heal": heal()}
    assert set(files) == set(analyze.INPUTS)        # never the repo's own results files
    argv, paths = [], {}
    for name, recs in files.items():
        paths[name] = write(tmp_path / "in" / f"{name}.jsonl", recs)
        argv += [f"--{name.replace('_', '-')}", str(paths[name])]
    pres = write(tmp_path / "in" / "preservation.jsonl",
                 preservation(paths["heal"], paths["recovery"], paths["claims"]))
    history(tmp_path / "runs")
    return argv + ["--preservation", str(pres), "--runs", str(tmp_path / "runs"),
                   "--evolving-range", "0001-0019", "--flat-range", "0020-0039",
                   "--out", str(tmp_path / "out"), "--boot", "200"]


def section(numbers: dict, prefix: str) -> dict:
    return next(v for k, v in numbers["sections"].items() if k.startswith(prefix))


@pytest.fixture
def numbers(world, no_network) -> dict:
    return analyze.main([*world, "--no-figures"])


# --- end to end -------------------------------------------------------------------------

def test_every_section_runs_and_writes_outputs(world, no_network, tmp_path):
    numbers = analyze.main(world)
    out = tmp_path / "out"
    assert all("skipped" not in v for v in numbers["sections"].values()), numbers["sections"]
    for name in ("analysis_k", "analysis_depth", "analysis_groups", "analysis_flat",
                 "analysis_heal"):
        assert (out / f"{name}.png").stat().st_size > 0
    md = (out / "analysis_tables.md").read_text()
    assert "Skipped" not in md and "failed" not in md
    assert md.count("(small sample)") > 10          # every cell here is a handful of runs
    assert json.loads((out / "analysis.json").read_text())["small_sample_runs"] == analyze.SMALL


def test_missing_inputs_skip_sections_without_crashing(tmp_path, no_network, capsys):
    argv = []
    for name in analyze.INPUTS:
        argv += [f"--{name.replace('_', '-')}", str(tmp_path / "nowhere" / f"{name}.jsonl")]
    numbers = analyze.main([*argv, "--runs", str(tmp_path / "noruns"),
                            "--preservation", str(tmp_path / "nowhere" / "pres.jsonl"),
                            "--out", str(tmp_path / "out"), "--no-figures"])
    assert all("skipped" in v for v in numbers["sections"].values())
    assert all(f["state"] == "missing" for f in numbers["inputs"].values())
    assert set(numbers["inputs"]) == {*analyze.INPUTS, "preservation"}
    out = capsys.readouterr().out
    assert "input missing: families" in out and "input missing: preservation" in out
    assert "Skipped:" in (tmp_path / "out" / "analysis_tables.md").read_text()


def test_thin_input_is_flagged(numbers):
    assert numbers["inputs"]["families"] == {
        "path": numbers["inputs"]["families"]["path"], "records": 18, "runs": 2,
        "part_files": 0, "state": "thin"}


# --- 1. task families --------------------------------------------------------------------

def test_family_asr_per_task_disagreement_and_any_task(numbers):
    s1 = section(numbers, "1.")
    none = s1["none"]
    assert (none["asr"]["n"], none["asr"]["d"]) == (2, 8)          # repeat 1: R1 A A C C
    assert [(t["n"], t["d"]) for t in none["per_task"]] == [(1, 2), (1, 2), (0, 2), (0, 2)]
    assert (none["tasks_disagree"]["n"], none["tasks_disagree"]["d"]) == (1, 2)
    assert none["harmful_any_task"]["n"] == 1 and none["small_sample"]
    assert (s1["incident[cf]"]["asr"]["n"], s1["incident[cf]"]["asr"]["d"]) == (1, 8)
    # selective loses one of the two probes R1's own repeat-1 'none' got right; R2 keeps both
    assert (s1["selective[oracle]"]["bmp"]["n"], s1["selective[oracle]"]["bmp"]["d"]) == (3, 4)


# --- 2. repeats --------------------------------------------------------------------------

def test_repeat_flips_spread_and_majority(numbers):
    s2 = section(numbers, "2.")
    none = s2["none"]
    assert none["runs_all_repeats"] == 2
    assert [r["n"] for r in none["asr_per_repeat"]] == [2, 2, 1]
    assert none["spread_pp"] == 12.5
    assert (none["attack_differs"]["n"], none["attack_differs"]["d"]) == (1, 8)  # R1 task 1
    assert none["majority_of_3_asr"]["n"] == 2 and none["single_replay_asr"]["n"] == 2
    assert none["majority_changes_task"]["n"] == 0
    inc = s2["incident[cf]"]        # task 3 attacked on repeat 1 only: the vote drops it
    assert inc["majority_of_3_asr"]["n"] == 0 and inc["majority_changes_task"]["n"] == 1
    # the noise floor pools deterministic stores only (none, selective[oracle]), not incident
    assert s2["replay_noise"]["units"] == 16 and s2["replay_noise"]["attack_differs"]["n"] == 1


def test_majority_is_strict():
    reps = [{"task_outcomes": [A, A, C]}, {"task_outcomes": [A, C, C]},
            {"task_outcomes": [C, A, C]}]
    assert analyze.majority(reps) == [True, True, False]


# --- 3/4. knob sweeps --------------------------------------------------------------------

def test_k_sweep_uses_canonical_own_k_and_counts_replays_per_incident(numbers):
    s3 = section(numbers, "3.")
    assert [s3[f"none k={k}"]["asr"]["n"] for k in (3, 5, 10)] == [1, 2, 3]
    assert s3["none k=5"]["runs"] == 3                     # k=5 read from summary.json
    assert s3["incident[cf] k=3"]["cf_replays_per_incident"]["mean"] == 4
    assert s3["incident[cf] k=5"]["cf_replays_per_incident"]["mean"] == 6   # from notes
    assert s3["incident[cf] k=10"]["cf_replays_per_incident"]["mean"] == 11
    assert s3["incident[cf] k=5"]["incident_runs"] == 2


def test_depth_sweep_collateral_and_poison_left(numbers):
    s4 = section(numbers, "4.")
    assert [s4[f"selective[oracle] depth={d}"]["collateral"] for d in (1, 2, 3)] == [1, 2, 3]
    assert s4["none depth=2"]["poison_left"]["n"] == 3
    assert s4["none depth=0"]["asr"]["n"] == 2          # canonical sweep on the same runs
    assert s4["selective[oracle] depth=0"]["common_runs"] == 2


def only(tmp_path: Path, **files) -> list[str]:
    """analyze.py arguments with just `files` present; every other input is missing."""
    argv = []
    for name in analyze.INPUTS:
        path = tmp_path / "in" / f"{name}.jsonl"
        if name in files:
            write(path, files[name])
        argv += [f"--{name.replace('_', '-')}", str(path)]
    pres = tmp_path / "in" / "preservation.jsonl"
    if "preservation" in files:
        write(pres, files["preservation"])
    return argv + ["--preservation", str(pres), "--runs", str(tmp_path / "runs"),
                   "--out", str(tmp_path / "out"), "--boot", "200"]


def partial_depth() -> dict[str, list[dict]]:
    """A batch cut part-way: depth 2 reached R1 and R2 but not R3, the one run whose
    poison is gone and whose quarantine hit benign memories. Over all runs poison left
    rises 2/3 -> 2/2 and collateral falls 1.0 -> 0.0 with depth only because R3 drops
    out; on the runs measured at every depth both stay flat."""
    def at(d, run, mode, **kw):
        extra = {"depth": d, "observe": "fresh"} if d else {}
        return rec(run, mode, **extra, **kw)
    out = {"recovery": [], "depth": []}
    for d in (0, 1, 2):
        for run in (R1, R2, R3) if d < 2 else (R1, R2):
            out["depth" if d else "recovery"] += [
                at(d, run, "none", outcomes=(A if run == R1 else C,),
                   residual_attack=0 if run == R3 else 1),
                at(d, run, "selective", verdicts="oracle",
                   collateral={"benign": 3 if run == R3 else 0})]
    return out


def series_spy(monkeypatch) -> list[tuple]:
    """Every _series call as (name, points, joined, annotate), in drawing order."""
    drawn, real = [], analyze._series

    def spy(ax, xs, points, slot, n_series, name, step, connect=True, annotate=False):
        drawn.append((name, list(points), connect, annotate))
        return real(ax, xs, points, slot, n_series, name, step, connect, annotate)
    monkeypatch.setattr(analyze, "_series", spy)
    return drawn


def test_depth_paired_columns_hold_the_same_runs_at_every_depth(tmp_path, no_network):
    s4 = section(analyze.main([*only(tmp_path, **partial_depth()), "--no-figures"]), "4.")
    left = {d: s4[f"none depth={d}"]["poison_left"] for d in (0, 1, 2)}
    assert [(v["n"], v["d"]) for v in left.values()] == [(2, 3), (2, 3), (2, 2)]
    paired = [s4[f"none depth={d}"]["poison_left_common_runs"] for d in (0, 1, 2)]
    assert [(v["n"], v["d"]) for v in paired] == [(2, 2)] * 3
    sel = [s4[f"selective[oracle] depth={d}"] for d in (0, 1, 2)]
    assert [r["collateral"] for r in sel] == [1.0, 1.0, 0.0]
    assert [r["collateral_common_runs"] for r in sel] == [0.0, 0.0, 0.0]
    assert all(r["common_runs"] == 2 for r in sel)
    md = (tmp_path / "out" / "analysis_tables.md").read_text()
    assert "| poison left on those runs |" in md and "| collateral on those runs |" in md


def test_depth_figure_draws_the_runs_measured_at_every_depth(tmp_path, no_network,
                                                            monkeypatch):
    drawn = series_spy(monkeypatch)
    analyze.main(only(tmp_path, **partial_depth()))
    asr, left = [s for s in drawn if s[0] == "none"]
    assert asr[2] and asr[3] and left[2] and left[3]        # joined, n written per point
    assert [p[3] for p in asr[1]] == [2, 2, 2]              # R3 is left out at every depth
    assert [p[0] for p in asr[1]] == [50, 50, 50]             # all runs: 33, 33, 50
    assert [p[0] for p in left[1]] == [100, 100, 100]       # all runs would read 67, 67, 100
    assert (tmp_path / "out" / "analysis_depth.png").stat().st_size > 0


def test_k_figure_and_paired_replays_use_the_runs_measured_at_every_k(tmp_path, no_network,
                                                                      monkeypatch):
    fresh = {"observe": "fresh", "family": "recorded", "depth": 0, "verdicts": "cf"}
    k_recs = [rec(R1, "incident", k=3, cf_replays=4, **fresh),
              rec(R2, "incident", k=3, cf_replays=6, **fresh),
              rec(R1, "incident", k=10, cf_replays=11, **fresh)]
    canon = [rec(run, "incident", verdicts="cf", notes="incident: inferred_roots=1 replays=6")
             for run in (R1, R2)]
    for run in (R1, R2):
        (tmp_path / "runs" / run).mkdir(parents=True)
        (tmp_path / "runs" / run / "summary.json").write_text(json.dumps({"k": 5}))
    drawn = series_spy(monkeypatch)
    s3 = section(analyze.main(only(tmp_path, k=k_recs, recovery=canon)), "3.")
    k3 = s3["incident[cf] k=3"]
    assert k3["cf_replays_per_incident"]["mean"] == 5                  # R1 and R2
    assert k3["cf_replays_per_incident_common_runs"]["mean"] == 4      # R1 only
    assert k3["cf_replays_per_incident_common_runs"]["n"] == 1
    assert "cf_replays_per_incident_common_runs" not in s3.get("none k=3", {})
    replays = [s for s in drawn if s[0] == "incident[cf]"][-1]
    assert [p[0] for p in replays[1]] == [4, 6, 11] and replays[2]
    assert [p[3] for p in replays[1]] == [1, 1, 1]


def test_sweep_levels_joins_only_runs_measured_at_every_level():
    a0, b0, a1, b1 = ({"r": i} for i in range(4))
    assert analyze.sweep_levels({0: {"a": a0, "b": b0}, 1: {"a": a1}}, [0, 1]) == (
        [[a0], [a1]], True)
    # no run at both levels: every run, unjoined, so no trend is drawn
    assert analyze.sweep_levels({0: {"a": a0}, 1: {"b": b1}}, [0, 1]) == ([[a0], [b1]], False)
    assert analyze.sweep_levels({}, [0, 1]) == ([[], []], True)


# --- 5. re-retrieval and the ablation table ----------------------------------------------

def test_culprit_agreement_and_frozen_noise_floor(numbers):
    agree = section(numbers, "5.")["culprit_agreement"]
    # R1: families' repeat-1 culprit m1 = re-retrieval's m1. R2: no frozen record with
    # culprits except the groups sweep, whose only set is a pair -> no single culprit.
    assert agree["counts"] == {"same culprits": 1, "only re-retrieval found one": 1}
    assert agree["frozen_sources"] == {"families sweep": 1, "groups sweep (single removals)": 1}
    # families repeats of R1: {m1}, {m1}, {} -> same once, frozen-only twice
    assert agree["noise_floor_counts"] == {"same culprits": 1, "only frozen found one": 2}


def test_reretrieve_costs_and_paired_rows(numbers):
    fr = section(numbers, "5.")["frozen_vs_reretrieve"]
    assert fr["frozen (canonical)"]["cf_replays"]["mean"] == 6
    assert fr["re-retrieval"]["cf_replays"]["mean"] == 6
    assert fr["frozen (canonical)"]["cf_seconds"] is None        # never measured
    assert [p["a"] for p in fr["paired"]] == ["incident[cf]+reretrieve"]


def test_ablation_table_rows_and_replay_noise(numbers):
    ab = section(numbers, "5.")["ablation"]
    assert ab["rows"]["incident runs: none"]["runs"] == 2
    assert ab["rows"]["incident runs: selective[cf]"]["asr"]["n"] == 1
    assert {(p["a"], p["b"]) for p in ab["paired"]} == {
        ("incident[cf]", "selective[cf]"), ("ancestry", "selective[cf]"),
        ("incident[cf]", "ancestry")}
    assert ab["replay_noise"]["pairs"] == 1 and ab["replay_noise"]["attack_flips"]["n"] == 1


# --- 6. groups ---------------------------------------------------------------------------

def test_culprit_kinds_latency_and_clean_control(numbers):
    s6 = section(numbers, "6.")
    kinds = s6["culprit_kinds"]
    assert (kinds["alarms"], kinds["no_incident"]) == (2, 1)
    assert kinds["single"]["n"] == 1 and kinds["pair"]["n"] == 1
    lat = s6["latency"]
    assert lat["recovery seconds"]["sequential"]["mean"] == 25
    assert lat["recovery seconds"]["parallel"]["mean"] == 11
    assert lat["cf replays"]["sequential"]["mean"] == 8 and lat["cf replays"]["paired_runs"] == 2
    # the groups sweep pairs with the canonical one despite its stop-after-action knob
    assert [p["population"] for p in s6["asr"]["paired"]] == ["incident", "all"]
    clean = s6["clean_control"]["incident[cf]+verify (groups)"]
    assert clean["culprit_kinds"] == {"single": 1}
    assert (clean["bmp"]["n"], clean["bmp"]["d"]) == (1, 2)


def test_culprit_kind_by_smallest_set():
    assert analyze.culprit_kind({"incident": False, "notes": ""}) == "no incident"
    assert analyze.culprit_kind({"incident": True, "culprit_sets": []}) == "none"
    assert analyze.culprit_kind({"incident": True, "culprit_sets": [["a", "b"], ["c"]]}) == "single"
    assert analyze.culprit_kind({"incident": True, "culprit_sets": [["a", "b", "c"]]}) == "group"
    assert analyze.culprit_kind({"incident": True}) == "unknown"


def test_agreement_classes():
    f = analyze.culprits
    a = f({"culprit_sets": [["m1"], ["m2"]]})
    assert analyze.agreement(a, f({"culprit_sets": [["m2"], ["m1"]]})) == "same culprits"
    assert analyze.agreement(a, f({"culprit_sets": [["m2"]]})) == "overlapping"
    assert analyze.agreement(a, f({"culprit_sets": [["m3"]]})) == "disjoint"
    assert analyze.agreement(frozenset(), frozenset()) == "neither found one"
    assert f({"culprit_sets": [["m1", "m2"]]}, singles_only=True) == frozenset()


# --- 7. claims ---------------------------------------------------------------------------

def test_claims_rows_paired_and_reinfection(numbers):
    s7 = section(numbers, "7.")
    assert s7["selective_claims[oracle]"]["claim_repaired"] == 2
    bmp = s7["selective_claims[oracle]"]["bmp"]
    assert (bmp["n"], bmp["d"]) == (2, 4)
    pairs = {(p["a"], p["b"], p["population"]): p for p in s7["paired"]}
    diff = pairs[("selective_claims[oracle]", "selective[oracle]", "all")]["BMP"]
    assert diff[0] == pytest.approx(-0.5)
    aa = pairs[("selective[oracle]", "selective[oracle]+recovery_sweep", "all")]
    assert aa["ASR"] == [0, 0, 0]
    rein = s7["reinfection"]
    assert rein["selective_claims[oracle]+immunize"]["runs"] == 1
    assert rein["selective[oracle]+immunize"]["reinfected"]["n"] == 1   # cp 1 attacked again


# --- 8. flat vs evolving -----------------------------------------------------------------

def test_history_split_and_exact_mcnemar(numbers):
    hist = section(numbers, "8.")["history"]
    assert hist["flat poisoned all"]["attacker"]["n"] == 1
    assert hist["evolving poisoned all"]["attacker"]["n"] == 2
    m = hist["mcnemar poisoned all"]
    assert (m["pairs"], m["both"], m["flat_only"], m["evolving_only"]) == (2, 1, 0, 1)
    assert m["p"] == 1.0


def test_root_only_effect_flat_vs_evolving(numbers):
    root = section(numbers, "8.")["recovery"]["root_only"]
    assert root["flat"]["cleared_by_root"]["n"] == 2
    assert root["evolving"]["cleared_by_root"]["n"] == 0
    assert root["flat_minus_evolving_reduction_pp"]["mean"] == 100


def test_mcnemar_exact_values():
    assert analyze.mcnemar_exact(0, 0) == 1.0
    assert analyze.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert analyze.mcnemar_exact(1, 1) == 1.0
    assert analyze.mcnemar_exact(2, 8) == pytest.approx(2 * (1 + 10 + 45) / 1024)


def test_history_rows_classify_by_range_and_suffix(tmp_path):
    history(tmp_path)
    (tmp_path / "0030_poisoned_office_z_dense_direct_evolving").mkdir()   # evolving name, flat range
    rows, unreadable = analyze.history_rows(tmp_path, "0020-0039", "0001-0019")
    memory = {r["run_dir"]: r["memory"] for r in rows}
    assert memory[R1] == "evolving" and memory[FL1] == "flat"
    assert "0030_poisoned_office_z_dense_direct_evolving" not in memory and unreadable == 0
    assert {r["category"] for r in rows} == {"direct"}


# --- 9. healing vs quarantine --------------------------------------------------------------

def test_heal_rows_read_each_configuration_from_one_sweep_at_its_lowest_repeat(numbers):
    s9 = section(numbers, "9.")
    assert (s9["runs"], s9["incident_runs"]) == (3, 2)
    assert s9["absent_modes"] == ["unmerge"]
    src = s9["sources"]
    assert src["heal[cf]"] == {"heal": 3} and src["incident[cf]"] == {"heal": 3}
    assert src["incident[cf]+verify"] == {"recovery": 2}         # the heal sweep never ran it
    assert src["selective[oracle]"] == {"claims": 2}             # claims before canonical
    assert src["selective[cf]"] == {"claims": 2, "recovery": 1}  # R3 only in the canonical one
    assert list(src)[:4] == ["heal[cf]", "heal_oracle[oracle]", "incident[cf]",
                             "incident[cf]+verify"]
    rows = s9["rows"]
    heal_inc = rows["incident runs: heal[cf]"]          # R1 cleared (not the repeat-2 replay)
    assert (heal_inc["asr"]["n"], heal_inc["asr"]["d"], heal_inc["runs"]) == (1, 2, 2)
    assert heal_inc["residual_attack"] == 0.5
    assert (rows["all runs: heal[cf]"]["asr"]["n"], rows["all runs: heal[cf]"]["asr"]["d"]) == (1, 3)
    # BMP against the heal sweep's own 'none' (both probes right): incident drops one per run
    bmp = rows["incident runs: incident[cf]"]["bmp"]
    assert (bmp["n"], bmp["d"]) == (2, 4)


def test_heal_actions_and_costs(numbers):
    act = section(numbers, "9.")["actions"]
    h = act["heal[cf]"]
    assert (h["n_healed"], h["claims_kept"], h["claims_dropped"]) == (1.0, 2.0, 1.0)
    assert h["n_heal_quarantined"] == pytest.approx(1 / 3, abs=1e-3)
    assert h["claim_cf_replays"]["mean"] == 2 and h["cf_replays"]["mean"] == 4
    assert act["incident[cf]+verify"]["n_healed"] is None         # older record: no field
    assert act["incident[cf]+verify"]["cf_replays"]["mean"] == 8  # read from the notes


def test_heal_bcp_matches_rows_to_the_records_file_they_measured(numbers):
    pres = section(numbers, "9.")["preservation"]
    h = pres["incident runs: heal[cf]"]
    assert (h["bcp"]["n"], h["bcp"]["d"], h["acr"]["n"], h["acr"]["d"]) == (13, 16, 3, 4)
    # the canonical sweep's incident[cf] row for R1 (10/10) is another copy: not used
    inc = pres["incident runs: incident[cf]"]
    assert (inc["bcp"]["n"], inc["bcp"]["d"], inc["acr"]["n"]) == (6, 16, 4)
    allr = pres["all runs: heal[cf]"]
    assert (allr["bcp"]["n"], allr["bcp"]["d"], allr["acr"]["n"], allr["acr"]["d"]) == (16, 19, 3, 5)
    sc = pres["incident runs: selective_claims[oracle]"]          # R2's copy disagreed
    assert (sc["measured"], sc["runs"], sc["bcp"]["n"], sc["bcp"]["d"]) == (1, 2, 3, 4)
    assert pres["incident runs: incident[cf]+verify"]["measured"] == 0
    lo, hi = h["bcp"]["boot"]["ci"]
    assert lo <= 13 / 16 <= hi


def test_heal_paired_differences_heal_minus_incident(numbers):
    pairs = {(p["a"], p["b"]): p for p in section(numbers, "9.")["paired"]}
    assert set(pairs) == {("heal[cf]", "incident[cf]"), ("heal_oracle[oracle]", "incident[cf]"),
                          ("heal_oracle[oracle]", "heal[cf]")}
    p = pairs[("heal[cf]", "incident[cf]")]
    assert (p["runs"], p["claims_runs"], p["population"]) == (2, 2, "incident")
    assert p["ASR"][0] == pytest.approx(0.5) and p["BMP"][0] == pytest.approx(0.5)
    assert p["BCP"][0] == pytest.approx(7 / 16) and p["ACR"][0] == pytest.approx(-0.25)
    loc = pairs[("heal_oracle[oracle]", "heal[cf]")]
    assert loc["BCP"][0] == pytest.approx(2 / 16) and loc["ASR"][0] == pytest.approx(-0.5)


def test_heal_figure_points(numbers):
    fig = section(numbers, "9.")["figure"]
    assert fig["population"] == "incident runs"
    pts = {p["label"]: p for p in fig["points"]}
    h = pts["heal[cf]"]
    assert h["blocked"][0] == 50 and h["acr"][0] == 75 and h["bcp"][0] == 81.25
    assert h["blocked"][1] <= 50 <= h["blocked"][2]               # Wilson around 1 - ASR
    assert pts["incident[cf]+verify"]["bcp"] is None              # not drawn: no BCP


def test_heal_without_preservation_still_reports_outcomes(tmp_path, no_network):
    numbers = analyze.main(only(tmp_path, heal=heal(), recovery=canonical()))
    s9 = section(numbers, "9.")
    assert "skipped" not in s9 and s9["preservation"] is None
    assert s9["rows"]["incident runs: heal[cf]"]["asr"]["n"] == 1
    assert all(p["BCP"] is None for p in s9["paired"])
    assert not (tmp_path / "out" / "analysis_heal.png").exists()
    md = (tmp_path / "out" / "analysis_tables.md").read_text()
    assert "Benign claim preservation skipped" in md and "not drawn" in md


def test_heal_section_skips_without_heal_records(tmp_path, no_network):
    numbers = analyze.main([*only(tmp_path, recovery=canonical(), claims=claims()),
                            "--no-figures"])
    assert "no heal or heal_oracle" in section(numbers, "9.")["skipped"]


def test_heal_pairs_fall_back_to_incident_without_cf_knobs(tmp_path, no_network):
    grp = {"cf_groups": True, "cf_workers": 6}
    recs = [heal_rec(run, "heal", (C,), **grp) for run in (R1, R2)]
    numbers = analyze.main([*only(tmp_path, heal=recs, recovery=canonical()), "--no-figures"])
    s9 = section(numbers, "9.")
    assert [(p["a"], p["b"]) for p in s9["paired"]] == [
        ("heal[cf] cf_groups=True cf_workers=6", "incident[cf]")]
    assert s9["sources"]["incident[cf]"] == {"recovery": 2}


# --- 9. heal v2: heal_rebuild, dropped_by, post_heal_harmful, the v2 knobs -----------------

def v2_rec(run, mode, outcomes, *, dropped=None, post=None, ddmin=0, **extra) -> dict:
    """A heal v2 record, the fields as recover.heal_v2_fields writes them."""
    claim = mode in ("heal", "heal_rebuild")
    verify = "verify" in extra.get("modifiers", ())
    return heal_rec(run, mode, outcomes, **{
        "verify_n": 1 if verify else None, "claim_n": CLAIM_N if claim else None,
        "ddmin_budget": DDMIN_BUDGET if claim else None, "ddmin_replays": ddmin,
        "post_heal_harmful": post, "dropped_by": {r: 0 for r in RULES} | (dropped or {}),
        **extra})


def heal_v2() -> list[dict]:
    """R1 and R2 are incidents. heal drops 2 'name' + 1 'cf' claims on R1 (post-heal check
    harmful on a third of its replays, 4 ddmin replays) and 1 'source-taint' on R2 (check
    clean); heal_rebuild drops 1 'support' and 1 by a rule this script does not know on R1
    and leaves R2 attacked; heal_oracle drops by its labels and runs no check. incident
    and heal also ran +verify at verify_n=3, and heal once more at claim_n=5."""
    out = []
    for run in (R1, R2):
        r1 = run == R1
        out += [v2_rec(run, "none", (A,)),
                v2_rec(run, "incident", (C,)),
                v2_rec(run, "incident", (C,), modifiers=("verify",), verify_n=3),
                v2_rec(run, "heal", (C,), dropped={"name": 2, "cf": 1} if r1 else
                       {"source-taint": 1}, post=1 / 3 if r1 else 0.0, ddmin=4 if r1 else 0),
                v2_rec(run, "heal", (C,), modifiers=("verify",), verify_n=3,
                       dropped={"name": 1}, post=0.0),
                v2_rec(run, "heal", (C,), claim_n=5, dropped={"name": 1}),
                v2_rec(run, "heal_rebuild", (C,) if r1 else (A,),
                       dropped={"support": 1, "new-rule": 1} if r1 else {}),
                v2_rec(run, "heal_oracle", (C,), dropped={"oracle": 2})]
    return out


@pytest.fixture
def v2(tmp_path, no_network) -> dict:
    return section(analyze.main([*only(tmp_path, heal=heal_v2(), recovery=canonical()),
                                 "--no-figures"]), "9.")


def test_heal_rebuild_is_a_heal_mode_and_the_v2_knobs_are_rows_of_their_own(v2):
    labels = list(v2["sources"])
    assert labels[:3] == ["heal[cf]", "heal[cf] claim_n=5", "heal[cf]+verify verify_n=3"]
    assert "heal_rebuild[cf]" in labels and "incident[cf]+verify verify_n=3" in labels
    assert labels.index("heal_rebuild[cf]") < labels.index("heal_oracle[oracle]") \
        < labels.index("incident[cf]")
    # The heal sweep's own +verify at verify_n=3 is not pooled with the canonical one.
    assert v2["sources"]["incident[cf]+verify"] == {"recovery": 2}
    assert v2["sources"]["incident[cf]+verify verify_n=3"] == {"heal": 2}
    rb = v2["rows"]["incident runs: heal_rebuild[cf]"]
    assert (rb["asr"]["n"], rb["asr"]["d"]) == (1, 2)
    assert "heal_rebuild" not in v2["absent_modes"]


def test_heal_decisions_report_dropped_by_and_the_post_heal_check(v2):
    d = v2["decisions"]
    assert set(d) == {"heal[cf]", "heal[cf] claim_n=5", "heal[cf]+verify verify_n=3",
                      "heal_rebuild[cf]", "heal_oracle[oracle]"}          # heal modes only
    h = d["heal[cf]"]
    assert h["claims_dropped"] == 4 and h["runs"] == 2
    assert h["dropped_by"] == {"name": 2, "source-taint": 1, "cf": 1, "ddmin": 0,
                               "support": 0, "oracle": 0, "new-rule": 0}
    assert h["ddmin_replays"]["mean"] == 2
    assert h["post_heal_runs"] == 2 and h["post_heal_runs_harmful"]["n"] == 1
    assert h["post_heal_harmful_share"]["mean"] == pytest.approx(1 / 6, abs=1e-4)
    rb = d["heal_rebuild[cf]"]
    assert rb["dropped_by"]["support"] == 1 and rb["dropped_by"]["new-rule"] == 1
    assert rb["post_heal_runs"] == 0 and rb["post_heal_runs_harmful"] is None
    assert d["heal_oracle[oracle]"]["dropped_by"]["oracle"] == 4


def test_heal_v2_pairs_meet_incident_and_heal_on_their_own_setting(v2):
    pairs = {(p["a"], p["b"]) for p in v2["paired"]}
    assert pairs == {
        ("heal[cf]", "incident[cf]"),
        ("heal[cf] claim_n=5", "incident[cf]"),                  # claim_n is one-sided
        ("heal[cf]+verify verify_n=3", "incident[cf]+verify verify_n=3"),
        ("heal_rebuild[cf]", "incident[cf]"), ("heal_rebuild[cf]", "heal[cf]"),
        ("heal_oracle[oracle]", "incident[cf]"), ("heal_oracle[oracle]", "heal[cf]")}
    rb = next(p for p in v2["paired"] if (p["a"], p["b"]) == ("heal_rebuild[cf]", "heal[cf]"))
    assert rb["ASR"][0] == pytest.approx(0.5)


def test_heal_v2_tables_say_why_and_older_records_read_as_before(tmp_path, no_network):
    analyze.main([*only(tmp_path, heal=heal_v2(), recovery=canonical()), "--no-figures"])
    md = (tmp_path / "out" / "analysis_tables.md").read_text()
    assert "### Why claims were dropped, and the post-heal check" in md
    assert "| heal[cf] | 2 (2 scenarios, small sample) | 4 | 2 (50%) | 1 (25%) | 1 (25%) |" in md
    assert "heal_rebuild and heal_oracle minus heal" in md
    # A heal v1 sweep (no v2 field): no such table, heal_rebuild never reported missing.
    numbers = analyze.main([*only(tmp_path / "v1", heal=heal(), recovery=canonical()),
                            "--no-figures"])
    s9 = section(numbers, "9.")
    assert "decisions" not in s9 and "heal_rebuild" not in s9["absent_modes"]
    md = (tmp_path / "v1" / "out" / "analysis_tables.md").read_text()
    assert "Why claims were dropped" not in md
    assert "### Paired: heal minus incident (same detection), heal_oracle minus heal" in md


def test_a_tuned_heal_meets_the_incident_of_its_own_counterfactual_options(tmp_path, no_network):
    """claim_n is set aside before the counterfactual's options are, so heal@c5 with six
    workers meets the heal sweep's six-worker incident, not the canonical one."""
    recs = [v2_rec(rd, m, (C,), cf_workers=6, **({"claim_n": 5} if m == "heal" else {}))
            for rd in (R1, R2) for m in ("none", "incident", "heal")]
    s9 = section(analyze.main([*only(tmp_path, heal=recs, recovery=canonical()),
                               "--no-figures"]), "9.")
    assert [(p["a"], p["b"]) for p in s9["paired"]] == [
        ("heal[cf] cf_workers=6 claim_n=5", "incident[cf] cf_workers=6")]


def _phase2_heal_outs(jobs: list[Path]) -> dict[str, set[str]]:
    """--out -> the flags of the recover lines in `jobs` that run a heal mode."""
    import shlex
    outs: dict[str, set[str]] = {}
    for f in jobs:
        for line in f.read_text().splitlines():
            argv = shlex.split(line)
            if not argv or argv[0] != "recover" or "--mode" not in argv:
                continue
            i = argv.index("--mode") + 1
            modes = []
            while i < len(argv) and not argv[i].startswith("--"):
                modes.append(argv[i])
                i += 1
            if set(modes) & set(analyze.HEAL_MODES):
                outs.setdefault(argv[argv.index("--out") + 1], set()).update(
                    a for a in argv if a.startswith("--"))
    return outs


def test_default_heal_inputs_are_the_files_the_phase2_heal_jobs_write():
    """Section 9 reads, by default, what data/main_*_jobs.txt's heal sweeps write (the
    dense file first, its flat twins second), and every poisoned heal sweep there is read
    by this script or by analyze_history.py's section 5. data/ is not in git: without the
    job files there is nothing to check against."""
    import analyze_history
    defaults = analyze.build_parser().parse_args([]).heal
    assert defaults == [analyze.ROOT / p for p in analyze.HEAL_INPUTS]
    assert analyze.INPUTS["heal"] == analyze.HEAL_INPUTS[0] == "results/main/records.jsonl"
    jobs = sorted((analyze.ROOT / "data").glob("main_*_jobs.txt"))
    if not jobs:
        pytest.skip("no data/main_*_jobs.txt in this checkout")
    outs = _phase2_heal_outs(jobs)
    assert set(analyze.HEAL_INPUTS) <= set(outs)
    history = {str(p.relative_to(analyze.ROOT)) for ps in analyze_history.MAIN.values()
               for p in ps}
    poisoned = {o for o, flags in outs.items() if not flags & {"--reinfect", "--include-clean"}}
    assert poisoned <= set(analyze.HEAL_INPUTS) | history


def test_phase2_heal_outs_reads_modes_and_out(tmp_path):
    jobs = tmp_path / "main_x_jobs.txt"
    jobs.write_text("recover --runs a --mode none incident --out results/x.jsonl\n"
                    "recover --runs b --mode heal heal_oracle --family full --out results/h.jsonl\n"
                    "recover --runs c --include-clean --mode incident heal --out results/c.jsonl\n"
                    "history --suite dense\n")
    assert _phase2_heal_outs([jobs]) == {
        "results/h.jsonl": {"--runs", "--mode", "--family", "--out"},
        "results/c.jsonl": {"--runs", "--include-clean", "--mode", "--out"}}


def test_each_heal_file_is_a_section_of_its_own(tmp_path, no_network):
    """--heal with the dense file and a flat one: two sections (9a, 9b), each on its own
    runs, its own input name (heal, heal_flat), its own figure; a bcp.py row of the flat
    file is matched to the flat section only."""
    argv = only(tmp_path, recovery=canonical())
    dense = write(tmp_path / "in" / "main" / "records.jsonl", heal())
    flat_recs = [heal_rec(rd, m, (A,) if m == "none" else (C,)) for rd in (FL1, FL2)
                 for m in ("none", "incident", "heal")]
    flat_path = write(tmp_path / "in" / "flat" / "main.jsonl", flat_recs)
    pres = [pres_row(flat_path, r, 4, 3 if r["mode"] == "heal" else 1, 2, 0)
            for r in flat_recs if r["mode"] != "none"]
    pres += [pres_row(dense, r, 4, 2, 2, 0) for r in heal() if r["mode"] == "heal_oracle"]
    i = argv.index("--heal")
    argv[i + 1:i + 2] = [str(dense), str(flat_path)]
    argv[argv.index("--preservation") + 1] = str(write(tmp_path / "in" / "pres.jsonl", pres))
    numbers = analyze.main(argv)
    assert {"heal", "heal_flat"} <= set(numbers["inputs"])
    titles = [t for t in numbers["sections"] if t.startswith("9")]
    assert titles == [f"9a. Healing vs quarantine (Phase 2, H): {dense}",
                      f"9b. Healing vs quarantine (Phase 2, H): {flat_path}"]
    a, b = (numbers["sections"][t] for t in titles)
    assert (a["runs"], b["runs"]) == (3, 2)                     # never pooled
    assert b["sources"]["heal[cf]"] == {"heal_flat": 2}
    assert b["preservation"]["incident runs: heal[cf]"]["bcp"]["n"] == 6
    assert "incident runs: heal[cf]" not in (a["preservation"] or {}) or \
        a["preservation"]["incident runs: heal[cf]"]["measured"] == 0
    out = tmp_path / "out"
    assert (out / "analysis_heal.png").is_file() and (out / "analysis_heal_flat.png").is_file()
    assert [n for n, _ in analyze.heal_input_names(["x/main/r.jsonl", "y/main/r.jsonl",
                                                    "z/main/r.jsonl"])] == [
        "heal", "heal_main", "heal_main_2"]


def test_heal_section_counts_and_resamples_scenarios(tmp_path, no_network):
    """A second history run of R1's scenario (office_a): four runs, three scenarios; the
    cells say so, flag by scenarios, and the paired interval resamples scenarios."""
    twin = "0015_poisoned_office_a_dense_direct_evolving"
    recs = heal() + [heal_rec(twin, "none", (A,)),
                     heal_rec(twin, "heal", (A,), residual_attack=1),
                     heal_rec(twin, "incident", (C,))]
    s9 = section(analyze.main([*only(tmp_path, heal=recs, recovery=canonical()),
                               "--no-figures"]), "9.")
    assert (s9["runs"], s9["scenarios"]) == (4, 3)
    assert (s9["incident_runs"], s9["incident_scenarios"]) == (3, 2)
    row = s9["rows"]["incident runs: heal[cf]"]
    assert (row["runs"], row["scenarios"], row["small_sample"]) == (3, 2, True)
    assert (row["asr"]["n"], row["asr"]["d"]) == (2, 3)
    assert row["asr"]["interval"] == "scenario bootstrap"
    p = next(p for p in s9["paired"] if (p["a"], p["b"]) == ("heal[cf]", "incident[cf]"))
    assert (p["runs"], p["scenarios"]) == (3, 2)
    md = (tmp_path / "out" / "analysis_tables.md").read_text()
    assert "4 runs (3 scenarios) a heal mode was applied to" in md
    assert "| 3 (2 scenarios, small sample) |" in md


def _point(label, mode, knobs, xy, verdicts="cf", modifiers=(), measured=12):
    x, y = xy
    return {"label": label, "mode": mode, "verdicts": verdicts, "modifiers": list(modifiers),
            "knobs": [list(k) for k in knobs], "runs": measured, "scenarios": measured,
            "measured": measured, "measured_scenarios": measured,
            "blocked": [x, max(x - 10, 0), min(x + 5, 100)],
            "acr": [x, max(x - 10, 0), min(x + 5, 100)], "bcp": [y, max(y - 20, 0), min(y + 5, 100)]}


FULL = (("family", "full"), ("cf_workers", 6), ("stop_after_action", True))
V3 = (*FULL, ("verify_n", 3))


def test_direct_labels_name_only_what_sets_a_configuration_apart():
    pts = [_point("heal", "heal", FULL, (91, 64)),
           _point("heal+v", "heal", V3, (91, 91), modifiers=("verify",)),
           _point("inc", "incident", FULL, (91, 33)),
           _point("inc+v", "incident", V3, (91, 33), modifiers=("verify",)),
           _point("oracle", "heal_oracle", FULL, (91, 100), verdicts="oracle"),
           _point("claims", "selective_claims", FULL[:1] + FULL[2:], (50, 50), verdicts="oracle"),
           _point("claims cf", "selective_claims", (), (40, 40))]
    names, variant, shared, codes = analyze.direct_labels(pts)
    assert names == ["heal", "heal+verify v3", "incident", "incident+verify v3", "heal_oracle",
                     "selective_claims[oracle] full stop", "selective_claims[cf]"]
    assert variant == [0, 1, 0, 1, 0, 1, 0]
    assert shared == []                                  # one point has default knobs
    assert codes == {"full": "family=full", "stop": "stop_after_action=True",
                     "v3": "verify_n=3"}
    names, _, shared, _ = analyze.direct_labels(pts[:5])
    assert shared == [("family", "full"), ("cf_workers", 6), ("stop_after_action", True)]
    assert not any("family" in n or "cf_workers" in n for n in names)
    note = analyze.heal_note({"population": "incident runs"}, shared, {"v3": "verify_n=3"})
    assert "every point: family=full, cf_workers=6, stop_after_action=True" in " ".join(note)
    assert all(len(line) <= 125 for line in note)


def test_place_labels_never_leaves_the_axes_when_no_spot_is_free():
    """Twelve labels at one point on the right edge: once the free spots are used up,
    the old fallback (the first spot, right of the point) ran past the axes."""
    plt, fig, (ax,) = analyze._figure(1, 3.0)
    ax.set_xlim(-3, 108)
    ax.set_ylim(-3, 108)
    fig.tight_layout()
    renderer = fig.canvas.get_renderer()
    analyze.place_labels(ax, [(100, 50, f"heal+verify v{i}") for i in range(12)], renderer)
    frame = ax.get_window_extent(renderer)
    for t in ax.texts:
        box = t.get_window_extent(renderer)
        assert frame.x0 <= box.x0 and box.x1 <= frame.x1, t.get_text()
        assert frame.y0 <= box.y0 and box.y1 <= frame.y1, t.get_text()
    plt.close(fig)


def test_heal_figure_labels_are_short_and_inside_the_axes(tmp_path, monkeypatch):
    """The Phase 2 shape (heal and incident each with a +verify v3 twin, every point at
    family=full ...), crowded at the right edge and in the corners: each label names
    only what sets it apart, stays inside its axes, and the shared knobs go in the note."""
    pts = [_point("heal", "heal", FULL, (91, 64)),
           _point("heal+v", "heal", V3, (91, 91), modifiers=("verify",)),
           _point("rebuild", "heal_rebuild", FULL, (91, 50)),
           _point("inc", "incident", FULL, (91, 33), measured=4),
           _point("inc+v", "incident", V3, (91, 33), modifiers=("verify",), measured=4),
           _point("oracle", "heal_oracle", FULL, (100, 100), verdicts="oracle"),
           _point("sel", "selective", FULL, (100, 0), verdicts="oracle"),
           _point("unm", "unmerge", FULL, (0, 100), verdicts="oracle"),
           _point("recon", "selective_recon", FULL, (0, 0), verdicts="oracle")]
    seen, notes = [], []
    real = analyze._save

    def spy(fig, plt, path, legend_rows=1, after_layout=None):
        def check(renderer):
            after_layout(renderer)
            notes.extend(t.get_text() for t in fig.texts)
            for ax in fig.axes:
                frame = ax.get_window_extent(renderer)
                for t in ax.texts:
                    box = t.get_window_extent(renderer)
                    seen.append((t.get_text(), frame.x0 <= box.x0 and box.x1 <= frame.x1
                                 and frame.y0 <= box.y0 and box.y1 <= frame.y1))
        return real(fig, plt, path, legend_rows, check)

    monkeypatch.setattr(analyze, "_save", spy)
    path = analyze.plot_heal({"population": "incident runs", "points": pts}, tmp_path / "h.png")
    assert path.stat().st_size > 0
    assert seen and all(inside for _, inside in seen), [t for t, ok in seen if not ok]
    texts = " ".join(t for t, _ in seen)
    assert "heal+verify v3" in texts and "family=" not in texts and "cf_workers" not in texts
    assert "every point: family=full" in " ".join(notes)


def test_heal_rules_keep_a_rule_this_script_does_not_know():
    assert analyze.heal_rules([]) == list(RULES)
    assert analyze.heal_rules([{"dropped_by": {"zz": 1, "name": 2}}, {}]) == [*RULES, "zz"]


def test_heal_figure_draws_more_than_four_modes_without_a_fifth_hue(tmp_path, no_network):
    context = [{**heal_rec(rd, m, (C,)), "verdicts": "oracle"} for rd in (R1, R2)
               for m in ("selective_claims", "selective")]
    recs = heal_v2() + context
    heal_path = tmp_path / "in" / "heal.jsonl"
    pres = [pres_row(heal_path, r, 4, 2 + (r["mode"].startswith("heal")), 2, 0)
            for r in recs if r["mode"] != "none"]
    analyze.main([*only(tmp_path, heal=recs, recovery=canonical(), preservation=pres)])
    assert (tmp_path / "out" / "analysis_heal.png").stat().st_size > 0
    assert set(analyze.HEAL_COLOURS[:analyze.HEAL_PRIMARY]) == set(analyze.PALETTE)
    assert set(analyze.HEAL_COLOURS[analyze.HEAL_PRIMARY:]) == {analyze.INK2}
    assert len(analyze.HEAL_MARKERS) == len(analyze.HEAL_COMPARE)


def test_boot_pooled_and_paired_difference():
    assert analyze.boot_pooled([], 50) is None and analyze.boot_pooled([(0, 0)], 50) is None
    assert analyze.boot_pooled([(1, 2), (1, 2)], 50) == (0.5, 0.5, 0.5)
    obs, lo, hi = analyze.boot_pooled([(3, 4), (0, 4)], 200)
    assert obs == 3 / 8 and lo == 0 and hi == 0.75
    d = analyze.boot_pooled_diff([(3, 4), (2, 4)], [(1, 4), (0, 4)], 200)
    assert d[0] == pytest.approx(0.5) and d[1] <= 0.5 <= d[2]
    assert analyze.boot_pooled_diff([(1, 1)], [(0, 0)], 50) is None


def test_bootstraps_resample_scenarios_when_given_clusters():
    """Nine runs of one scenario and one of another: by run the interval of 9/10 stays
    high; by scenario it spans 0-100% (both draws the lone run, or neither). One cluster
    per run draws exactly what resampling runs does."""
    pairs = [(1, 1)] * 9 + [(0, 1)]
    scen = ["s1"] * 9 + ["s2"]
    by_run = analyze.boot_pooled(pairs, 400)
    assert by_run == analyze.boot_pooled(pairs, 400, clusters=list(range(10)))
    assert by_run[1] > 0.5
    assert analyze.boot_pooled(pairs, 400, clusters=scen) == (0.9, 0.0, 1.0)
    vals = [1.0] * 9 + [0.0]
    assert analyze.boot_mean(vals, 400) == analyze.boot_mean(vals, 400, clusters=list(range(10)))
    assert analyze.boot_mean(vals, 400, clusters=scen)[1:] == (0.0, 1.0)
    assert analyze.boot_mean([None, 2.0, 4.0], 50, clusters=["a", "b", "b"]) == (3.0, 3.0, 3.0)
    # All-or-nothing: no draw differs, so the bound is Wilson's over the scenarios.
    none_hit = analyze.boot_pooled([(0, 4)] * 6, 200, clusters=["a", "a", "b", "c", "d", "e"])
    assert none_hit == (0.0, 0.0, pytest.approx(measure.wilson(0, 5)[1]))
    assert analyze.boot_pooled([(0, 4)] * 6, 200) == (0.0, 0.0, 0.0)   # unclustered: as before
    d = analyze.boot_pooled_diff(pairs, [(0, 1)] * 10, 400, clusters=scen)
    assert d[0] == pytest.approx(0.9) and (d[1], d[2]) == (0.0, 1.0)
    assert analyze.runs_scen_cell(12, 9) == "12 (9 scenarios, small sample)"
    assert analyze.runs_scen_cell(3, 3, 5) == "3 of 5 (3 scenarios, small sample)"
    assert analyze.runs_scen_cell(40, 12) == "40 (12 scenarios)"


def test_claim_counts_only_for_measured_rows():
    row = {"status": "ok", "benign": 5, "benign_kept": 4, "attack": 2, "attack_kept": 1}
    assert analyze.claim_counts(row) == {"bcp": (4, 5), "acr": (1, 2)}
    assert analyze.claim_counts({**row, "status": "copy_mismatch"}) is None
    assert analyze.claim_counts(None) is None


# --- helpers -----------------------------------------------------------------------------

def test_read_records_merges_unmerged_parts_and_skips_torn_lines(tmp_path, capsys):
    main = write(tmp_path / "groups.jsonl", [rec(R1, "none", outcomes=(A,))])
    parts = tmp_path / ".parts"
    write(parts / "groups.054.0000.jsonl", [rec(R1, "none", outcomes=(C,)), rec(R2, "none")])
    (parts / "groups.054.0001.jsonl").write_text('{"run_dir": "torn')
    write(parts / "groups_clean.054.0000.jsonl", [rec(R3, "none")])   # another file's part
    recs, n_parts = analyze.read_records(main)
    assert n_parts == 2
    assert {r["run_dir"]: r["task_outcomes"] for r in recs} == {R1: [C], R2: [C]}
    assert "not valid JSON" in capsys.readouterr().out
    assert analyze.read_records(main, parts=False)[1] == 0
    assert analyze.read_records(tmp_path / "absent.jsonl") == (None, 0)


def test_tagged_copies_drop_knobs_and_keep_the_original():
    r = groups()[0]
    t = analyze.tagged([r], "groups")[0]
    assert analyze.knobs(t) == () and analyze.knobs(r) != ()
    assert analyze.label(t) == "incident[cf]+groups+verify"
    assert r["cf_groups"] is True and r["modifiers"] == ["verify"]


def test_cf_replays_reads_field_or_notes():
    assert analyze.cf_replays({"cf_replays": 3, "notes": "replays=9"}) == 3
    assert analyze.cf_replays({"notes": "incident: replays=4; verify: replays=2"}) == 6
    assert analyze.cf_replays({"notes": "no harmful action"}) is None


def test_small_sample_label():
    assert analyze.runs_cell(9) == "9 (small sample)"
    assert analyze.runs_cell(10) == "10"
    assert analyze.rate(1, 4)["small_sample"] and not analyze.rate(5, 10)["small_sample"]
