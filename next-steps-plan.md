# MemHeal: next to-dos and open research questions

Written November 16, 2026, after the N=3 sweep (`main` at `ab6ec01`). It covers what is left
from the proposal and the research questions the results opened, in the order to do them.

**Do Step 0 first.** Every model call from here on goes through OpenRouter, using free
(`:free`) models only. Step 0 is a prerequisite for every item below that calls a model.

## Where things stand

| Area | Status |
|---|---|
| Proposal Weeks 1-12, Section 9.3 ablations, stronger and cross-model runs, LongMemEval and MemBench, MS extensions (except LongMemEval-V2) | Done, at 3 history runs per scenario, on OpenAI models |
| Healing (`heal`, `heal_rebuild`, `heal_oracle`) | Done. With the same detection, heal keeps 77% vs 38% of benign facts at equal attack success (6.9% vs 7.5%), on gpt-5.4-mini |
| Heal and quarantine with the verify loop (`+verify`, 3 replays per check) | **Partial**: the run stopped when API credits ran out (main suite about 90 of 693 run x replay pairs) |
| Reinfection for healing | **Not comparable yet**: heal (no oracle) was compared with oracle selective cleanup |
| MemSecBench (Section 7.6) | **Blocked**: no public data release |
| Final report PDF | **Not written**: figures and tables exist (`results/final/`, `results/analysis/`) |

## Ground rules for everything below

1. **Free models only.** Every model call goes through OpenRouter with a model id ending in
   `:free`. The harness refuses anything else (Step 0.3). A paid model needs explicit approval
   from the project owner for that run.
2. **Estimate requests and time first.** Free models cost $0 but are rate-limited, so run the
   estimator (P0.4) before any model-calling step and post the request count and calendar time.
3. **Smoke first.** Every model-calling step runs on the 4-8 run smoke set before the full set,
   and each run is logged in `docs/remaining-work-plan.md` (command, scope, run folders, outcome).
4. **Local only.** Simulated tools, `.example` attacker hosts, no network in tests.
   `scripts/checks.sh` enforces this before every push and in CI.
5. **Scenario is the unit.** Report scenario-level means with scenario-bootstrap intervals
   (`analyze_main.py`); call a difference real only when its interval excludes 0.
6. **A new model is a new cell.** Results from the free model are never pooled with the
   gpt-5.4-mini results. Any comparison re-runs both arms (and its `none` baseline) on the free
   model.

## Step 0 (mandatory first): migrate to OpenRouter free models

### 0.1 What OpenRouter's free tier allows (verify before relying on it)

These figures come from third-party write-ups, and OpenRouter's own limits page should be
checked when you start (https://openrouter.ai/docs/api/reference/limits):

| Limit | Value |
|---|---|
| Requests per minute, all `:free` models together | **20**, per account (more API keys do not add throughput) |
| Requests per day, account with under $10 of lifetime credit purchases | **50** |
| Requests per day, account with at least $10 purchased | **1,000** |
| Over the limit | HTTP 429 (per-minute or daily); HTTP 402 if the credit balance is negative |

**What this means for the harness.**
- A history run makes about 60 model calls and a recovery configuration 12-55. At 1,000
  requests a day, that is roughly **16 history runs a day**.
- At 50 a day, about one history run a day: only smoke tests are feasible.
- Size every experiment below to a daily budget and plan in days, not hours.

### 0.2 Credentials and configuration

- The key is already in the environment as `OPENROUTER_API_KEY`; the harness reads it from
  there or from `.env`. Never write the key into code, logs, results or commits (check 1 of
  `scripts/checks.sh` scans for it).
- New settings in `memheal/llm.py`, overridable by env var and CLI flag:

| Setting | Value |
|---|---|
| `MEMHEAL_PROVIDER` | `openrouter` (default from now on); `openai` stays available only to reproduce old results |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `MEMHEAL_MODEL` | the chosen `:free` model id (0.4) |

- Send OpenRouter's optional attribution headers (`HTTP-Referer`, `X-Title: MemHeal`).
  They must contain no personal data.

### 0.3 Code changes

1. **`memheal/llm.py`: one client, two providers.**
   - Keep the `LLM.respond(...)` signature and the `LLMResponse` fields (`text`, `tool_calls`,
     `history_items`, `model`) so callers do not change.
   - Recommended: port the OpenRouter path to the **Chat Completions API**
     (`client.chat.completions.create` with the OpenAI SDK pointed at the OpenRouter base URL).
     OpenRouter's Responses API exists but is labelled beta and may change.
   - Mapping from today's Responses calls:

| Today (Responses API) | Chat Completions |
|---|---|
| `instructions` | first `system` message |
| `tools` | same function schemas, `type: "function"` |
| `tool_choice: "required"` | `tool_choice: "required"`. Not every free model honours it, so fall back to `"auto"` plus a retry when no tool is called |
| `text.format` json_schema | `response_format: {type: "json_schema", json_schema: {...}}`. Without schema support, use one forced tool whose parameters are the schema |
| `reasoning.effort` | omit |
| `history_items` | the assistant message with its `tool_calls` |
| `function_call_output` | a `role: "tool"` message with `tool_call_id` |

   - Send `provider: {"require_parameters": true}` so OpenRouter only routes to endpoints that
     support tools and structured outputs.

2. **`memheal/agent.py`:** stop building `{"type": "function_call_output"}` items itself. Ask
   the LLM for a provider-neutral tool-result item (e.g. `llm.tool_result(call, output)`), so
   the loop works with either provider.
3. **Free-only guard.** With the `openrouter` provider, `LLM.__init__` raises unless the model
   id ends in `:free`. It also raises if the base URL is `api.openai.com` while the provider is
   `openrouter`.
4. **Every construction site uses the provider settings and the free default model:**
   - `run_demo.py`, `recover.py` (`default_llm_factory`), `diagnose.py` (LLM baselines),
     `measure.py` (LLM judge), `chat.py`, `probe_recovery.py`;
   - `memheal/backends/mem0_backend.py` (its provider forwards to `memheal.llm.LLM`);
   - `batch.py` (passes `--model`).
5. **Records and logs:** record `provider`, the requested model and the returned model on every
   run (`experiment_runs`), every `llm.jsonl` line and every recovery record. Analysis groups by
   them.
6. **Temperature 0:** request it, record what the endpoint reports, and treat replay noise as
   possibly higher on free endpoints (measure it in 0.6).

### 0.4 Choosing the free model

- List candidates from OpenRouter's public model list (`GET https://openrouter.ai/api/v1/models`;
  no key needed). Keep only models whose id ends in `:free`, whose prompt and completion
  prices are 0, and whose `supported_parameters` include both `tools` and `structured_outputs`
  (or `response_format`).
- Pick one by a 10-scenario smoke (clean and poisoned):
  - tool calls must parse;
  - clean runs must reach the correct destination in at least 80% of runs;
  - poisoned runs must sometimes reach the attacker. A model that never acts or never gets
    attacked cannot measure recovery.
- Pin the exact id in `memheal/llm.py` (`DEFAULT_MODEL`) and in `docs/remaining-work-plan.md`.
  Record a second free model as a fallback in case the first is withdrawn.

### 0.5 Rate limits: make the harness patient, not fast

Free models are slower and more limited, so test runs need larger rate-limit allowances.

| Change | Where | Value |
|---|---|---|
| Client-side limiter shared by every thread in the process | `memheal/llm.py` (token bucket) | at most **18 requests/min** (headroom under 20); configurable `MEMHEAL_RPM` |
| Retries on 429, 5xx and timeouts | `memheal/llm.py` | honour `Retry-After`; otherwise backoff 2 s to **120 s** cap with jitter; up to **40** attempts for 429 (today: 14 attempts, 30 s cap) |
| Request timeout | `memheal/llm.py` | **180 s** (free endpoints can queue) |
| Daily cap reached (429 that says the daily limit is used, or `free_model_daily_requests.remaining == 0` when reported) | `memheal/llm.py` raises a distinct `DailyQuotaExhausted`; `batch.py` stops cleanly and prints the resume command | never retried in a loop |
| Batch concurrency | `batch.py` default `--workers 2` for OpenRouter; `--cf-workers 1-2` | more threads only queue behind the limiter |
| Daily budget | `batch.py --max-requests N` | stop before N requests (e.g. 900 of 1,000) so interactive use keeps some quota |

Every sweep must be resumable: the job generators already list only what is missing
(`data/resume_verify_jobs.txt` pattern), so a run stopped by the daily cap continues the next
UTC day.

### 0.6 Tests and checks (offline; no test may call OpenRouter)

- `scripts/checks.sh`: also export `OPENROUTER_API_KEY=checks-no-model-calls` and
  `OPENROUTER_BASE_URL=http://127.0.0.1:9`, so a stray call fails instead of using quota.
  Extend check 1 to scan for OpenRouter-style keys (`sk-or-...`).
- New unit tests with a fake Chat Completions client:
  - **Mapping:** tool calls, tool results and JSON-schema output are mapped correctly in both
    directions.
  - **Free-only guard:** the guard refuses non-`:free` ids and the OpenAI base URL.
  - **Retry and limiter:** the limiter never exceeds the configured RPM under 8 threads (fake
    clock); 429 handling honours `Retry-After`; the daily-cap error stops a batch cleanly.
  - **Construction sites:** every construction site from 0.3 picks up the provider and model
    from env and flags.
- Existing tests keep passing with `MEMHEAL_PROVIDER=openai` and with `openrouter`.

### 0.7 Re-baseline on the free model (model-calling; estimate first)

- **Smoke.** On the free model, run the 4-run smoke set: history clean and poisoned, then
  `none`, `incident` and `heal` recovery. Check that the full pipeline runs, tool calls parse,
  and the rate limiter and resume work.
- **Baseline cell.** 20 scenarios (5 per domain, all 3 attack categories), clean and poisoned:
  attack success, clean correctness, and replay noise (3 replays of `none`).
- **Size:** about 40 history runs and 20 x 3 recovery configurations, roughly 4,000 requests:
  **about 4 days at 1,000/day**, or not feasible at 50/day (smoke only).
- **Done when:**
  - the pipeline runs end to end on the free model;
  - noise and baseline attack success are recorded;
  - `docs/remaining-work-plan.md` names the pinned model.

### 0.8 Documentation

- README: setup now says `OPENROUTER_API_KEY` and the free-model rule, the daily-limit sizing,
  and how to resume a sweep the next day.
- `scripts/estimate_cost.py` (P0.4) reports requests and days for the free tier.

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
  (HPSize) and benign descendant fraction (BDF) have scenario-bootstrap intervals, plus a
  per-operation table (store, consolidate, reflect, Mem0 UPDATE/DELETE).

### P0.3 Documentation
- **What:** add the heal modes, `bcp.py`, `analyze_main.py` and `batch.py` to the README.

### P0.4 Request and time estimator (enforces rule 2)
- **What:** `scripts/estimate_cost.py JOBFILE [--rpm 18 --daily 1000]` expands a `batch.py` job
  file the way `batch.py` does (one job per run x mode x repeat). It multiplies by the
  measured calls per mode from `results/**` records and prints requests, days at the daily
  cap, and hours at the RPM limit. Optional `--price-in/--price-out` covers an approved paid
  run.
- **Measured call counts:** a history run makes about 60 calls; a recovery configuration about
  12 (quarantine), 20 (heal) or 50 (either with verify).
- **Check:** on the last sweep's job files it reproduces the logged call counts within 20%.

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

## P1: model-calling runs that close proposal gaps (after Step 0; estimate each first)

Sizes are requests on the free model, at about 1,000 requests a day. Every item re-runs both of
its arms on the free model (rule 6), so the counts are larger than the earlier gpt-5.4-mini
figures.

| Item | What it closes | Scope | Requests | Days at 1,000/day |
|---|---|---|---|---|
| P1.1 Fair reinfection pair | RQ4 for healing | `heal` and `incident`, both `+immunize --reinfect`, on 20 scenarios | about 4,000 | about 4 |
| P1.2 Verify loop, main question | H4 gap (no-oracle 7% vs oracle 1% attack success) | `heal +verify`, `incident +verify` and oracle `ancestry` on 20 scenarios, 1 replay | about 2,500 | about 3 |
| P1.3 Verify loop at full repetition | same, 3 replays | as P1.2 with `--repeat 1 2 3` | about 7,500 | about 8 |
| P1.4 Verify loop on external and adaptive suites | generalisation | 10 scenarios each | about 2,500 | about 3 |

Recommended order: P1.1, then P1.2. Do P1.3 and P1.4 only if P1.2 leaves an interval that
matters. The 20 scenarios should be the same across P1 items and Step 0.7, so the free-model
baseline is shared.

## Research questions to investigate

Each question has the evidence that motivates it, a method, a success criterion and a size.
Sizes in requests are on the free model.

### RQ2 (diagnosis): explaining incidents with no single culprit

- **Q2.1 What causes the 49% of incidents where no single removal stops the attack?**
  - **Evidence:** 81 of 167 explained incidents have no single culprit (N=3). In the earlier
    sweep, live re-reads of the poisoned page and 2+ attack-carrying memories explained most of
    them.
  - **Method:** P0.5 split; then, only for the redundant-memory group, run pair/group and
    keep-only-one tests (already built: `--cf-groups`).
  - **Success:** each incident is assigned a cause, and the redundant ones are explained by a
    pair or group.
  - **Size:** free split; group tests about 1,000 requests on 10 redundant incidents.
- **Q2.2 Does combining the LLM content classifier with the counterfactual beat either one?**
  - **Evidence:** the classifier is precise on the retrieved set (0.98) but noisy store-wide
    (0.56); the counterfactual is precise (0.79-0.95) with low store-wide recall (0.28).
  - **Method:** P0.5 offline simulation from existing outputs. Only if it wins, wire it into
    `incident`/`heal` as a pre-filter.
  - **Success:** higher F1 at equal or fewer model calls per incident.
  - **Size:** free.

### RQ3 / H4 (recovery trade-off): the no-oracle attack gap

- **Q3.1 Can no-oracle detection reach the 1% attack success of oracle ancestry cleanup?**
  - **Evidence:** heal and quarantine without the oracle leave 6.9-7.5% attack success; ancestry
    with the oracle root gets 1.1%. Heal + verify reached 0% on a 37-scenario subset.
  - **Method:** P1.2, then compare heal + verify, quarantine + verify and oracle ancestry on the
    same scenarios.
  - **Success:** heal + verify within 1-2 pp of oracle ancestry, keeping its preservation
    advantage.
  - **Size:** P1.2.
- **Q3.2 A preservation measure that does not use oracle labels.**
  - **Evidence:** BCP labels claims with the scenario's ground truth. BMP (benign probes) shows
    only +6 pp for healing, because probes are coarse.
  - **Method:** auto-generate one benign question per affected memory before recovery (from its
    non-attack claims) and check whether the recovered store still answers it.
  - **Success:** the label-free measure ranks modes the same way BCP does (rank correlation above
    0.8 across modes).
  - **Size:** about 1,000 requests on 10 runs.

### RQ4 (over time): reinfection without an oracle

- **Q4.1 Can MemHeal identify which sources to block without being told?**
  - **Evidence:** heal's no-oracle immunize blocked anything in only 55 of 77 runs, and store
    reinfection reached 35% vs 1% for oracle blocking.
  - **Method:** P0.5 scoring, then improve source attribution (block every page whose memories
    the culprits came from, plus pages they point to). Measure with P1.1.
  - **Success:** blocks the true poisoned page in at least 90% of runs with no clean page
    blocked, and store reinfection close to the oracle's.
  - **Size:** free analysis; the rerun is part of P1.1.

### RQ1 (evolution): mechanisms

- **Q5.1 Why is Mem0 attacked less than our evolving store (40% vs 70%)?**
  - **Method:** P0.5 mirror analysis: how often Mem0 adds, rewrites or deletes attack text, and
    whether the attack memory is retrieved at the target.
  - **Success:** the gap is explained by write behaviour or by retrieval.
  - **Size:** free.
- **Q5.2 Which memory operation is most dangerous, at N=3?**
  - **Method:** P0.2 per-operation table.
  - **Size:** free.

### New threat: attacks aimed at the healer

- **Q6.1 Can poison be written to get through claim-level healing?**
  - **Evidence:** heal drops claims by name, source taint and claim-level counterfactual. Each
    has an obvious evasion: split the claim across pages with no change words, or pair it with a
    benign fact in one sentence so it cannot be dropped alone.
  - **Method:** two new adaptive templates in `memheal/scenarios/adaptive.py`; smoke on 10
    scenarios x clean/poisoned, then heal, heal + verify and quarantine.
  - **Success:** either heal still suppresses the attack, or the evasion is documented with a
    proposed fix.
  - **Size:** about 2,000 requests (2 days).

### Cost and practicality

- **Q7.1 How cheap can healing get?**
  - **Evidence:** heal costs about 8 more model calls than quarantine per recovery; claim probes
    and ddmin dominate. On the free tier, calls are the scarce resource.
  - **Method:**
    - use the candidate filter (45% fewer counterfactual calls at the same recall) before claim
      probes;
    - stop claim probes once the post-heal check passes;
    - measure on the smoke set.
  - **Success:** heal costs no more than 1.3x quarantine in requests, with the same BCP and
    attack success.
  - **Size:** about 500 requests.

### External validity (later, needs decisions)

- **Q8.1 Other model families.** The free OpenRouter model from Step 0 already adds one
  non-OpenAI family. Comparing two free families on the 20-scenario baseline cell would test
  whether the findings depend on the model.
- **Q8.2 MemSecBench.** Revisit if its authors release data.
- **Q8.3 Real tools.** Everything uses simulated tools by design. A sandboxed real-tool pilot
  needs its own safety plan first.

## Suggested order

1. **Step 0** (0.1-0.6 offline; 0.7 smoke and baseline on the free model).
2. Free work alongside Step 0:
   - P0.1 report, P0.3 README, P0.4 estimator;
   - P0.5 analyses (Q2.1, Q2.2, Q4.1, Q5.1);
   - P0.2 N=3 survival tables.
3. Model calls, in daily-sized batches: P1.1 fair reinfection, then P1.2 verify loop.
4. Then Q4.1 improved source blocking (inside a P1.1-sized rerun), Q6.1 attacks on the healer,
   Q7.1 cheaper healing.
5. Decide on P1.3/P1.4, Q3.2 and Q8.x based on what steps 3-4 show.

At about 1,000 free requests a day, steps 1.7, 3 and 4 together take roughly **3-4 weeks** of
daily batches. At 50 a day only smoke tests are practical, so buying the $10 of credits that
raises the daily cap is worth raising with the project owner before step 3.
