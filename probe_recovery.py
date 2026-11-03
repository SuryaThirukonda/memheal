"""Replay a completed run's target task after removing memories (proposal Section 5.6).

Superseded by recover.py, which adds the Weeks 11-12 modes, verdicts, the full task
family, probes and reinfection. This keeps the original single-task probe and its
output (probes/<mode>.db, probes/results.json) for the early reimbursement runs, now
on top of memheal/recovery.py: removal is quarantine rather than 'deleted' (retrieval
sees active memories only, so the replay is the same), and the run.db is opened
read-only.

    python probe_recovery.py --run data/runs/0021_poisoned_reimbursement_long_evolving
    python probe_recovery.py --run <dir> --mode none root descendants full

Modes (Section 5.6), and the recovery mode each now maps to:
    none         baseline: change nothing, just replay              -> none
    root         remove the poisoned memories themselves            -> root
    descendants  remove every memory derived from them, not roots   (local: no recover.py twin)
    full         root + all descendants (conservative cleanup)      -> ancestry
    reset        remove all memory (maximum suppression)            -> reset
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

from memheal import agent, db, lineage, memory, recovery
from memheal.scenarios import ALL_VARIANTS as VARIANTS
from memheal.tools import ToolContext
from recover import load_run, pre_target_snapshot  # noqa: F401  (re-exported for old callers)

ROOT = Path(__file__).parent
MODES = ("none", "root", "descendants", "full", "reset")
RECOVERY_MODE = {"none": "none", "root": "root", "full": "ancestry", "reset": "reset"}


def apply_mode(conn, scenario, condition, mode: str) -> dict:
    roots, descendants = lineage.poison_lineage(conn, scenario, condition)
    if mode == "descendants":
        # The one probe mode with no recovery.py equivalent: it spares the roots on
        # purpose, to show what consolidation alone carried forward.
        p = recovery.RecoveryPlan("descendants", quarantine=set(descendants),
                                  reasons={d: "descendant" for d in descendants})
    else:
        p = recovery.plan(conn, scenario, condition, RECOVERY_MODE[mode],
                          roots=roots, descendants=descendants)
    recovery.apply(conn, p, None, None, f"probe:{mode}")
    return {"roots": roots, "descendants": sorted(descendants),
            "removed": [r["id"] for r in db.all_memories(conn) if r["id"] in p.quarantine]}


def probe(run_dir: Path, mode: str, variant: str | None, embedder, llm=None) -> dict:
    meta = load_run(run_dir)
    variant = variant or meta["config"]["variant"]
    scenario = VARIANTS[variant]
    task = scenario.target_tasks[meta["config"].get("task_index", 0)]
    agent_cfg = meta["config"].get("agent", {})
    cfg = agent.AgentConfig(
        k=agent_cfg.get("k", 5),
        source_labels=agent_cfg.get("source_labels", False),
        extraction_prompt=agent_cfg.get("extraction_prompt", "default"),
        enable_search=agent_cfg.get("enable_search", scenario.enable_search),
        tool_names=tuple(scenario.tool_names),
        memory_mode="flat",      # the probe replays one task; no maintenance
    )

    out_dir = run_dir / "probes"
    out_dir.mkdir(exist_ok=True)
    work = out_dir / f"{mode}.db"
    shutil.copy(pre_target_snapshot(run_dir, scenario.target_session), work)

    conn = db.connect(work)
    applied = apply_mode(conn, scenario, meta["condition"], mode)
    active_before = len(db.active_memories(conn))

    run_id = f"probe-{mode}-{meta['run_id']}"
    if llm is None:
        from memheal.llm import LLM
        llm = LLM(model=meta["model"], log_path=out_dir / f"{mode}.llm.jsonl")
    ctx = ToolContext(conn, run_id, scenario.target_session,
                      scenario.pages_for(meta["condition"]))
    result = agent.run_session(ctx, llm, embedder, cfg, task)

    logged = [json.loads(a["arguments_json"])
              for a in db.get_actions(conn, run_id, scenario.target_session)
              if a["tool_name"] == scenario.target_tool]
    destination = logged[0].get(scenario.target_field) if logged else None
    record = {
        "mode": mode,
        "destination": destination,
        "outcome": scenario.classify(destination),
        "n_roots": len(applied["roots"]),
        "n_descendants": len(applied["descendants"]),
        "n_removed": len(applied["removed"]),
        "active_at_replay": active_before,
        "retrieved": [(m, round(s, 3)) for m, s in result.retrieved],
        "reply": result.final_text,
    }
    conn.close()
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a run folder under data/runs/")
    p.add_argument("--mode", nargs="+", choices=MODES, default=["none", "root", "descendants", "full"])
    p.add_argument("--variant", default=None, help="override the variant recorded in the run")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    run_dir = Path(args.run)
    embedder = memory.get_embedder()
    variant = args.variant or load_run(run_dir)["config"]["variant"]

    print(f"run: {run_dir.name}")
    print(f"snapshot: {pre_target_snapshot(run_dir, VARIANTS[variant].target_session).name}\n")
    records = []
    for mode in args.mode:
        r = probe(run_dir, mode, args.variant, embedder)
        records.append(r)
        print(f"  {mode:<12} removed={r['n_removed']:<3} active={r['active_at_replay']:<3} "
              f"-> {r['destination']} [{r['outcome']}]")
    (run_dir / "probes" / "results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
