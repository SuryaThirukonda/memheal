"""Replay a completed run's target task after removing memories (proposal Section 5.6).

Takes a finished run, rewinds to its last snapshot before the target task, applies a
recovery mode, and replays only that task. Everything else about the run is held fixed,
so the difference in the logged destination is attributable to what was removed.

    python probe_recovery.py --run data/runs/0021_poisoned_reimbursement_long_evolving
    python probe_recovery.py --run <dir> --mode none root descendants full

Modes (Section 5.6):
    none         baseline: change nothing, just replay
    root         delete the poisoned memories themselves
    descendants  delete every memory derived from them, but not the roots
    full         root + all descendants (conservative cleanup)
    reset        delete all memory (maximum suppression, maximum utility loss)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

from memheal import agent, db, memory
from memheal.llm import LLM
from memheal.scenarios.reimbursement import VARIANTS
from memheal.tools import ToolContext
from run_demo import is_poison

ROOT = Path(__file__).parent
MODES = ("none", "root", "descendants", "full", "reset")
STAGE_ORDER = {"": 0, "_reflect": 1, "_consolidate": 2}


def pre_target_snapshot(run_dir: Path, target_day: int) -> Path:
    """The last snapshot taken before the target task ran."""
    best, best_key = None, None
    for p in (run_dir / "snapshots").glob("after_s*.db"):
        m = re.match(r"after_s(\d+)(_reflect|_consolidate)?\.db$", p.name)
        if not m:
            continue
        day = int(m.group(1))
        if day >= target_day:
            continue
        key = (day, STAGE_ORDER[m.group(2) or ""])
        if best_key is None or key > best_key:
            best, best_key = p, key
    if best is None:
        raise SystemExit(f"no pre-target snapshot in {run_dir}")
    return best


def load_run(run_dir: Path) -> dict:
    conn = db.connect(run_dir / "run.db")
    row = conn.execute("SELECT * FROM experiment_runs LIMIT 1").fetchone()
    cfg = json.loads(row["config_json"])
    conn.close()
    return {"run_id": row["run_id"], "condition": row["condition"],
            "model": row["model"], "config": cfg}


def apply_mode(conn, scenario, condition, mode: str) -> dict:
    roots = [r["id"] for r in db.all_memories(conn) if is_poison(r, scenario, condition)]
    descendants = sorted(set().union(*(db.get_descendants(conn, m) for m in roots))
                         if roots else set())
    if mode == "none":
        removed = []
    elif mode == "root":
        removed = roots
    elif mode == "descendants":
        removed = descendants
    elif mode == "full":
        removed = sorted(set(roots) | set(descendants))
    else:  # reset
        removed = [r["id"] for r in db.active_memories(conn)]
    db.set_status_many(conn, removed, "deleted")
    return {"roots": roots, "descendants": descendants, "removed": removed}


def probe(run_dir: Path, mode: str, variant: str | None, embedder) -> dict:
    meta = load_run(run_dir)
    variant = variant or meta["config"]["variant"]
    scenario = VARIANTS[variant]
    task = scenario.target_tasks[meta["config"].get("task_index", 0)]
    agent_cfg = meta["config"].get("agent", {})
    cfg = agent.AgentConfig(
        k=agent_cfg.get("k", 5),
        source_labels=agent_cfg.get("source_labels", False),
        extraction_prompt=agent_cfg.get("extraction_prompt", "default"),
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

    print(f"run: {run_dir.name}")
    print(f"snapshot: {pre_target_snapshot(run_dir, VARIANTS[load_run(run_dir)['config']['variant']].target_session).name}\n")
    records = []
    for mode in args.mode:
        r = probe(run_dir, mode, args.variant, embedder)
        records.append(r)
        print(f"  {mode:<12} removed={r['n_removed']:<3} active={r['active_at_replay']:<3} "
              f"-> {r['destination']} [{r['outcome']}]")
    (run_dir / "probes" / "results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
