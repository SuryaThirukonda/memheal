# MemHeal: next to-dos and open research questions

Written November 16, 2026, after the N=3 sweep (commit `acf8aff` on `remaining-proposal-work`).
It covers what is left from the proposal and the research questions the results opened, in the
order to do them.

## Where things stand

| Area | Status |
|---|---|
| Proposal Weeks 1-12, Section 9.3 ablations, stronger and cross-model runs, LongMemEval and MemBench, MS extensions (except LongMemEval-V2) | Done, at 3 history runs per scenario |
| Healing (`heal`, `heal_rebuild`, `heal_oracle`) | Done. With the same detection, heal keeps 77% vs 38% of benign facts at equal attack success (6.9% vs 7.5%) |
| Heal and quarantine with the verify loop (`+verify`, 3 replays per check) | **Partial**: the run stopped when API credits ran out (main suite about 90 of 693 run x replay pairs) |
| Reinfection for healing | **Not comparable yet**: heal (no oracle) was compared with oracle selective cleanup |
| MemSecBench (Section 7.6) | **Blocked**: no public data release |
| Final report PDF | **Not written**: figures and tables exist (`results/final/`, `results/analysis/`) |

## Ground rules for everything below

1. **Budget first.** Before any step that calls the model API: run the cost estimate (P0.4),
   post the dollar figure, and wait for an explicit yes. gpt-5.4-mini only unless asked.
2. **Smoke first.** Every paid step runs on the 4-8 run smoke set before the full set, and each
   run is logged in `docs/remaining-work-plan.md` (command, scope, run folders, outcome).
3. **Local only.** Simulated tools, `.example` attacker hosts, and no network in tests.
   `scripts/checks.sh` enforces this before every push and in CI.
4. **Scenario is the unit.** Report scenario-level means with scenario-bootstrap intervals
   (`analyze_main.py`); call a difference real only when its interval excludes 0.

### How to estimate cost

Measured from the logs: a recovery or history model call averages about **850 input and 55
output tokens** on gpt-5.4-mini. A history run makes about 60 calls; a recovery configuration
makes 12-55 calls (quarantine about 12, heal about 20, either with verify about 50).

```
cost  ~=  calls x (850 x input price + 55 x output price) / 1,000,000
```

**Rough rate until checked against OpenAI's price page:** the last sweep billed about $700 for
about 558M input and 35M output tokens across all four models. That gives a blended upper bound
of about **$1.25 per 1M input tokens**, or about **$1.05 per 1,000 gpt-5.4-mini calls**. The
estimates below use this rate and are meant to be refined by P0.4.

## P0: free, offline (no model calls)

### P0.1 Final report PDF
- **What:** write `docs/MemHeal_Final_Report.pdf` from `results/final/main_tables.md`,
  `results/analysis/*`, `results/diagnosis_n3/*`, `results/preservation_final/*`,
  `results/cost/*` and `results/final/rq1_n3.json`.
- **Must state:** the credit cutoff and the partial `+verify` pairs; that the reinfection
  comparison is oracle vs no-oracle source blocking; that BCP uses oracle claim labels.
- **Done when:** every number traces to a file, checked by an independent pass.

### P0.2 Survival and propagation tables at N=3 (Sections 8.1, 8.2)
- **What:** rerun `measure.py` survival, propagation and per-operation tables on all three
  history samples, and on flat, Mem0, adaptive and the cross-model runs.
- **Why:** these tables still come from one history run per scenario.
- **Done when:** poison survival rate (PSR), propagation size (PSize), harmful propagation size
  (HPSize) and benign descendant fraction (BDF) have scenario-bootstrap intervals, plus a per-operation table (store, consolidate,
  reflect, Mem0 UPDATE/DELETE).

### P0.3 Merge and documentation
- **What:** open a PR from `remaining-proposal-work` to `main`, let CI pass, merge. Add the heal
  modes, `bcp.py`, `analyze_main.py` and `batch.py` to the README.

### P0.4 Cost estimator (enforces rule 1)
- **What:** `scripts/estimate_cost.py JOBFILE [--price-in X --price-out Y]` expands a `batch.py`
  job file the way `batch.py` does (one job per run x mode x repeat). It multiplies by the
  measured calls per mode from `results/**` records and the tokens per call from the logs, and
  prints calls, tokens and dollars.
- **Check:** applied to the last sweep's job files, it should reproduce the logged call counts
  within 20%.

### P0.5 Free analyses that answer open questions with data already on disk
These need no new runs. Each feeds a question below.
- **Q2.1:** split the 81 incidents with no single culprit by cause, using the existing
  incident copies (attack text in 2+ retrieved memories, attacker page read live, or neither).
- **Q2.2:** simulate combined detectors (LLM content classifier AND counterfactual, OR, or
  classifier as a pre-filter) from `results/diagnosis_n3/*/diagnosis.json`, with precision,
  recall and model calls per incident.
- **Q4.1:** score no-oracle source attribution. For each heal run, compare the sources heal's
  immunize blocked with the true poisoned pages (precision and recall per run).
- **Q5.1:** explain why Mem0 is attacked less, from the mirror rows: attack text written, kept,
  rewritten or deleted per run, and whether it is retrieved at the target.

## P1: cheap paid runs that close proposal gaps (each needs a yes on its estimate)

| Item | What it closes | Runs | Rough calls | Rough cost |
|---|---|---|---|---|
| P1.1 Fair reinfection pair | RQ4 for healing | `incident +immunize --reinfect` on the same 77 runs heal already has | about 7,700 | **about $8** |
| P1.2 Finish `+verify`, main suite, replay 1 only | H4 gap (no-oracle 7% vs oracle 1% attack success) | about 400 missing configurations | about 20,000 | **about $21** |
| P1.3 Finish `+verify`, main suite, all 3 replays | same, at full repetition | about 1,200 configurations | about 60,000 | about $64 |
| P1.4 Finish `+verify` on external and adaptive suites | generalisation of the verify loop | about 340 configurations | about 17,000 | about $18 |

Recommended order: P1.1, then P1.2. Do P1.3 and P1.4 only if P1.2 leaves an interval that
matters. Skip the cross-model suite's `+verify` (gpt-5.4 and gpt-5.5 cost much more per call).

## Research questions to investigate

Each question has the evidence that motivates it, a method, a success criterion and a cost.

### RQ2 (diagnosis): explaining incidents with no single culprit

- **Q2.1 What causes the 49% of incidents where no single removal stops the attack?**
  - **Evidence:** 81 of 167 explained incidents have no single culprit (N=3). In the earlier
    sweep, live re-reads of the poisoned page and 2+ attack-carrying memories explained most of them.
  - **Method:** P0.5 split; then, only for the redundant-memory group, run pair/group and
    keep-only-one tests (already built: `--cf-groups`).
  - **Success:** each incident is assigned a cause, and the redundant ones are explained by a
    pair or group.
  - **Cost:** free split; group tests about $5-10 on the redundant subset.
- **Q2.2 Does combining the LLM content classifier with the counterfactual beat either one?**
  - **Evidence:** the classifier is precise on the retrieved set (0.98) but noisy store-wide
    (0.56); the counterfactual is precise (0.79-0.95) with low store-wide recall (0.28).
  - **Method:** P0.5 offline simulation from existing outputs. Only if it wins, wire it into
    `incident`/`heal` as a pre-filter.
  - **Success:** higher F1 at equal or fewer model calls per incident.
  - **Cost:** free.

### RQ3 / H4 (recovery trade-off): the no-oracle attack gap

- **Q3.1 Can no-oracle detection reach the 1% attack success of oracle ancestry cleanup?**
  - **Evidence:** heal and quarantine without the oracle leave 6.9-7.5% attack success; ancestry
    with the oracle root gets 1.1%. Heal + verify reached 0% on a 37-scenario subset.
  - **Method:** P1.2, then compare heal + verify, quarantine + verify and oracle ancestry on the
    same scenarios.
  - **Success:** heal + verify within 1-2 pp of oracle ancestry, keeping its preservation advantage.
  - **Cost:** about $21.
- **Q3.2 A preservation measure that does not use oracle labels.**
  - **Evidence:** BCP labels claims with the scenario's ground truth. BMP (benign probes) shows
    only +6 pp for healing, because probes are coarse.
  - **Method:** auto-generate one benign question per affected memory before recovery (from
    its non-attack claims) and check whether the recovered store still answers it.
  - **Success:** the label-free measure ranks modes the same way BCP does (rank correlation
    above 0.8 across modes).
  - **Cost:** question generation and answering are model calls: estimate first. Test on 20 runs
    (about $3).

### RQ4 (over time): reinfection without an oracle

- **Q4.1 Can MemHeal identify which sources to block without being told?**
  - **Evidence:** heal's no-oracle immunize blocked anything in only 55 of 77 runs, and store
    reinfection reached 35% vs 1% for oracle blocking.
  - **Method:** P0.5 scoring, then improve source attribution (block every page whose memories
    the culprits came from, plus pages they point to). Measure with P1.1.
  - **Success:** blocks the true poisoned page in at least 90% of runs with no clean page blocked,
    and store reinfection close to the oracle's.
  - **Cost:** free analysis; the rerun is part of P1.1.

### RQ1 (evolution): mechanisms

- **Q5.1 Why is Mem0 attacked less than our evolving store (40% vs 70%)?**
  - **Method:** P0.5 mirror analysis: how often Mem0 adds, rewrites or deletes attack text,
    and whether the attack memory is retrieved at the target.
  - **Success:** the gap is explained by write behaviour or by retrieval.
  - **Cost:** free.
- **Q5.2 Which memory operation is most dangerous, at N=3?**
  - **Method:** P0.2 per-operation table.
  - **Cost:** free.

### New threat: attacks aimed at the healer

- **Q6.1 Can poison be written to get through claim-level healing?**
  - **Evidence:** heal drops claims by name, source taint and claim-level counterfactual. Each
    has an obvious evasion: split the claim across pages with no change words, or pair it with a
    benign fact in one sentence so it cannot be dropped alone.
  - **Method:** two new adaptive templates in `memheal/scenarios/adaptive.py`; smoke on 10
    scenarios x clean/poisoned, then heal, heal + verify and quarantine.
  - **Success:** either heal still suppresses the attack, or the evasion is documented with a
    proposed fix.
  - **Cost:** about $5 for the smoke.

### Cost and practicality

- **Q7.1 How cheap can healing get?**
  - **Evidence:** heal costs about 8 more model calls than quarantine per recovery; claim probes
    and ddmin dominate.
  - **Method:**
    - use the candidate filter (45% fewer counterfactual calls at the same recall) before claim probes;
    - stop claim probes once the post-heal check passes;
    - measure on the smoke set.
  - **Success:** heal costs no more than 1.3x quarantine with the same BCP and attack success.
  - **Cost:** about $3 smoke.
  - **Note:** prompt caching will not help much: prompts average about 850 tokens, likely below
    the 1,024-token minimum for OpenAI's prompt caching.

### External validity (later, needs decisions)

- **Q8.1 Other model families.** Only OpenAI models were tested. A local open-weight model would
  remove API cost, but this laptop has no CUDA GPU, so start with a small model on CPU, smoke only.
- **Q8.2 MemSecBench.** Revisit if its authors release data.
- **Q8.3 Real tools.** Everything uses simulated tools by design. A sandboxed real-tool pilot
  needs its own safety plan first.

## Suggested order

1. P0.1 report, P0.3 merge, P0.4 cost estimator (all free).
2. P0.5 free analyses (Q2.1 split, Q2.2 combined detector, Q4.1 attribution score, Q5.1 Mem0)
   and P0.2 N=3 survival tables.
3. With approval: P1.1 fair reinfection (about $8) and P1.2 verify pairs (about $21).
4. With approval: Q4.1 improved source blocking (measured in a P1.1-sized rerun), Q6.1 adaptive
   attacks on the healer (about $5), Q7.1 cheaper healing (about $3).
5. Decide on P1.3/P1.4, Q3.2 and Q8.x based on what steps 2-4 show.

Total for steps 3-4 at the rough rate: **about $40**, each item approved separately.
