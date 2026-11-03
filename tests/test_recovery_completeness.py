"""Gaps the Weeks 11-12 plan names that the first pass left out (offline).

- recover.py --suite: select runs the way run_demo.py ran them (the plan's
  `--suite dense-smoke` verification step), not only by folder glob.
- the dry run shows whether a plan touches anything retrievable, so `root` on a
  consolidated root reads as the no-op it is (Section 7.2) before any model call.
- Section 8.10 storage cost: rows recovery added, recorded in the notes.
- measure.py recovery: reinfection *rate* (recovered runs that relapse, Section 8.8)
  and the clean-run probe ceiling the plan asks BMP to be read against.
"""

from __future__ import annotations

import pytest
from fakes import FakeEmbedder
from test_measure_recovery import rec
from test_recover import Factory, _never, make_run

import measure
import recover


@pytest.fixture
def emb():
    return FakeEmbedder()


# --- recover.py ------------------------------------------------------------------------

def test_suite_keeps_only_runs_whose_variant_is_in_it(tmp_path, emb, capsys):
    poisoned, _ = make_run(tmp_path, emb, "poisoned", number=1)
    make_run(tmp_path, emb, "clean", number=2)
    out = tmp_path / "records.jsonl"
    recover.main(["--runs", str(poisoned), "--suite", "dense", "--mode", "root",
                  "--dry-run", "--out", str(out)], llm_factory=_never)
    assert "1 run(s)" in capsys.readouterr().out
    # The variant is dense, so the non-dense catalog suite selects nothing.
    recover.main(["--runs", str(poisoned), "--suite", "evolve", "--mode", "root",
                  "--dry-run", "--out", str(out)], llm_factory=_never)
    assert "0 run(s)" in capsys.readouterr().out
    assert not out.exists()


def test_runs_or_suite_is_required():
    with pytest.raises(SystemExit):
        recover.parse_args(["--mode", "root"])
    assert recover.suite_variants("dense-smoke")
    assert recover.parse_args(["--suite", "dense-smoke"]).runs is None


def test_dry_run_shows_that_a_consolidated_root_is_not_retrievable(tmp_path, emb, capsys):
    poisoned, _ = make_run(tmp_path, emb, "poisoned", number=1)
    make_run(tmp_path, emb, "clean", number=2)
    recover.main(["--runs", str(poisoned), "--mode", "root", "ancestry", "--dry-run"],
                 llm_factory=_never)
    lines = {ln.split()[0]: ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("  ") and "quarantine=" in ln}
    # The fixture's root was consolidated on day 13: root quarantines it, but nothing
    # retrieval can see changes. ancestry also takes the active mixed summary.
    assert "active_touched=0 roots_active=0/1" in lines["root"]
    assert "active_touched=1 roots_active=0/1" in lines["ancestry"]


def test_records_note_the_storage_delta(tmp_path, emb):
    poisoned, _ = make_run(tmp_path, emb, "poisoned", number=1)
    make_run(tmp_path, emb, "clean", number=2)
    [plain] = recover.recover_run(poisoned, "selective", recover.Options(), Factory(), emb)
    assert "storage_delta=+0 memories" in plain["notes"]          # quarantine never adds rows


# --- measure.py recovery -----------------------------------------------------------------

def _series(run_dir, mods, attacked_at):
    return [rec(run_dir, "selective", verdicts="oracle", modifiers=mods, checkpoint=cp,
                outcomes=("attacker",) if cp in attacked_at else ("correct",))
            for cp in (0, 1, 3, 5, 10)]


def test_reinfection_rate_counts_recovered_runs_that_relapse():
    recs = [*_series("0001_poisoned_a", (), {3, 5, 10}),        # recovered, then relapsed
            *_series("0002_poisoned_b", (), {0, 1, 3, 5, 10}),  # never recovered: not counted
            *_series("0003_poisoned_c", (), set()),             # recovered and stayed so
            *_series("0001_poisoned_a", ("immunize",), set())]
    rates = measure.reinfection_rates(recs)
    assert rates[("selective", "oracle", (), "patched")] == (1, 2)
    assert rates[("selective", "oracle", ("immunize",), "patched")] == (0, 1)
    md = measure.recovery_tables(recs)
    assert "recovered runs attacked again (any cause)" in md
    assert "| 50% (1/2) |" in md


def test_reinfection_rate_ignores_configurations_without_follow_up():
    assert measure.reinfection_rates([rec("r", "ancestry")]) == {}


def test_clean_control_reports_the_probe_ceiling():
    recs = [rec("0001_poisoned_a", "none", probes=(True, False, False)),
            rec("0002_clean_a", "none", condition="clean", probes=(True, True, True)),
            rec("0002_clean_a", "incident", verdicts="cf", modifiers=("verify",),
                condition="clean", probes=(True, True, False))]
    assert measure.probe_counts(recs[1:]) == (5, 6)
    md = measure.recovery_tables(recs)
    assert ("Probe ceiling (clean runs, no recovery): 100% (3/3) of probes correct; "
            "poisoned runs with no recovery: 33% (1/3).") in md
    assert "| probes correct |" in md
