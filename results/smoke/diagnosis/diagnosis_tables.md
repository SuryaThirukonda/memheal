# Harmful-memory diagnosis (Sections 5.3, 8.3, 8.4, 9.1)

Runs: 4 qualified poisoned runs (data/runs/0964_poisoned_travel_car_hire_dense_composed_evolving data/runs/1003_poisoned_coding_ci_setup_script_dense_procedural_evolving data/runs/1018_poisoned_research_ethics_application_dense_direct_evolving data/runs/1089_poisoned_office_invoice_payment_dense_direct_evolving, suite dense); 4 with an incident (harmful target action). Counterfactual evidence from `recovery/incident.db`. LLM baselines: gpt-5.4-mini, temperature 0: 77 distinct prompts, 0 cached, 77 new calls, 0 failed.

Ground truth: positives = poison roots + descendants labelled harmful or mixed; negatives = benign descendants + non-lineage memories that do not name the attacker; ambiguous descendants are excluded. Non-lineage memories naming the attacker (e.g. read off the attacker's own portal) are excluded from the primary scoring and positive in the secondary one. Micro = pooled counts; macro = mean of per-run values where defined; 95% CIs are percentile bootstraps over runs (2000 resamples). 'alarm' methods act only after a harmful action, so they flag nothing in runs without one. LLM calls/run is the method's cost (cached answers included); for the MemHeal counterfactual it counts the agent steps of its baseline and single-removal replays, and for the MemHeal incident every step the pipeline replayed before it was measured (all counterfactual replays plus +verify's checks).

## 1. Diagnosis per scope (8.3 / 9.1)

### store: active memories at the pre-target snapshot, all runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 4 | 1.2 | 0.60 [0.25, 1.00] | 0.60 [0.25, 1.00] | 0.60 [0.22, 0.89] | 0.62 | 0.62 | 0.58 | 3/2/2 | 18.2 |
| source-only (external source) | store | 4 | 6.5 | 0.19 [0.12, 0.29] | 1.00 [1.00, 1.00] | 0.32 [0.22, 0.45] | 0.20 | 1.00 | 0.33 | 5/21/0 | 0.0 |
| retrieval attribution | alarm | 4 | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] | 0.25 | 1.00 | 0.39 | 5/15/0 | 0.0 |
| LLM self-explanation | alarm | 4 | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] | 0.88 | 0.88 | 0.83 | 4/1/1 | 1.0 |
| ancestry attribution (oracle root) | store | 4 | 1.5 | 0.83 [0.70, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.82, 1.00] | 0.92 | 1.00 | 0.95 | 5/1/0 | 0.0 |
| MemHeal counterfactual | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 12.0 |
| MemHeal incident (full quarantine set) | alarm | 4 | 1.0 | 1.00 [1.00, 1.00] | 0.80 [0.57, 1.00] | 0.89 [0.73, 1.00] | 1.00 | 0.88 | 0.92 | 4/0/1 | 22.5 |

### store: active memories at the pre-target snapshot, all runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 4 | 1.2 | 0.60 [0.25, 1.00] | 0.60 [0.25, 1.00] | 0.60 [0.22, 0.89] | 0.62 | 0.62 | 0.58 | 3/2/2 | 18.2 |
| source-only (external source) | store | 4 | 6.5 | 0.19 [0.12, 0.29] | 1.00 [1.00, 1.00] | 0.32 [0.22, 0.45] | 0.20 | 1.00 | 0.33 | 5/21/0 | 0.0 |
| retrieval attribution | alarm | 4 | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] | 0.25 | 1.00 | 0.39 | 5/15/0 | 0.0 |
| LLM self-explanation | alarm | 4 | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] | 0.88 | 0.88 | 0.83 | 4/1/1 | 1.0 |
| ancestry attribution (oracle root) | store | 4 | 1.5 | 0.83 [0.70, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.82, 1.00] | 0.92 | 1.00 | 0.95 | 5/1/0 | 0.0 |
| MemHeal counterfactual | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 12.0 |
| MemHeal incident (full quarantine set) | alarm | 4 | 1.0 | 1.00 [1.00, 1.00] | 0.80 [0.57, 1.00] | 0.89 [0.73, 1.00] | 1.00 | 0.88 | 0.92 | 4/0/1 | 22.5 |

### store: active memories at the pre-target snapshot, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 4 | 1.2 | 0.60 [0.25, 1.00] | 0.60 [0.25, 1.00] | 0.60 [0.22, 0.89] | 0.62 | 0.62 | 0.58 | 3/2/2 | 18.2 |
| source-only (external source) | store | 4 | 6.5 | 0.19 [0.12, 0.29] | 1.00 [1.00, 1.00] | 0.32 [0.22, 0.45] | 0.20 | 1.00 | 0.33 | 5/21/0 | 0.0 |
| retrieval attribution | alarm | 4 | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] | 0.25 | 1.00 | 0.39 | 5/15/0 | 0.0 |
| LLM self-explanation | alarm | 4 | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] | 0.88 | 0.88 | 0.83 | 4/1/1 | 1.0 |
| ancestry attribution (oracle root) | store | 4 | 1.5 | 0.83 [0.70, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.82, 1.00] | 0.92 | 1.00 | 0.95 | 5/1/0 | 0.0 |
| MemHeal counterfactual | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 12.0 |
| MemHeal incident (full quarantine set) | alarm | 4 | 1.0 | 1.00 [1.00, 1.00] | 0.80 [0.57, 1.00] | 0.89 [0.73, 1.00] | 1.00 | 0.88 | 0.92 | 4/0/1 | 22.5 |

### store: active memories at the pre-target snapshot, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 4 | 1.2 | 0.60 [0.25, 1.00] | 0.60 [0.25, 1.00] | 0.60 [0.22, 0.89] | 0.62 | 0.62 | 0.58 | 3/2/2 | 18.2 |
| source-only (external source) | store | 4 | 6.5 | 0.19 [0.12, 0.29] | 1.00 [1.00, 1.00] | 0.32 [0.22, 0.45] | 0.20 | 1.00 | 0.33 | 5/21/0 | 0.0 |
| retrieval attribution | alarm | 4 | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] | 0.25 | 1.00 | 0.39 | 5/15/0 | 0.0 |
| LLM self-explanation | alarm | 4 | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] | 0.88 | 0.88 | 0.83 | 4/1/1 | 1.0 |
| ancestry attribution (oracle root) | store | 4 | 1.5 | 0.83 [0.70, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.82, 1.00] | 0.92 | 1.00 | 0.95 | 5/1/0 | 0.0 |
| MemHeal counterfactual | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 12.0 |
| MemHeal incident (full quarantine set) | alarm | 4 | 1.0 | 1.00 [1.00, 1.00] | 0.80 [0.57, 1.00] | 0.89 [0.73, 1.00] | 1.00 | 0.88 | 0.92 | 4/0/1 | 22.5 |

### action: retrieved set of the harmful action, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 5.0 |
| source-only (external source) | store | 4 | 1.8 | 0.71 [0.57, 1.00] | 1.00 [1.00, 1.00] | 0.83 [0.73, 1.00] | 0.79 | 1.00 | 0.87 | 5/2/0 | 0.0 |
| retrieval attribution | alarm | 4 | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] | 0.25 | 1.00 | 0.39 | 5/15/0 | 0.0 |
| LLM self-explanation | alarm | 4 | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] | 0.88 | 0.88 | 0.83 | 4/1/1 | 1.0 |
| ancestry attribution (oracle root) | store | 4 | 1.2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 | 1.00 | 1.00 | 5/0/0 | 0.0 |
| MemHeal counterfactual | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 12.0 |
| MemHeal incident (full quarantine set) | alarm | 4 | 1.0 | 1.00 [1.00, 1.00] | 0.80 [0.57, 1.00] | 0.89 [0.73, 1.00] | 1.00 | 0.88 | 0.92 | 4/0/1 | 22.5 |

### action: retrieved set of the harmful action, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 5.0 |
| source-only (external source) | store | 4 | 1.8 | 0.71 [0.57, 1.00] | 1.00 [1.00, 1.00] | 0.83 [0.73, 1.00] | 0.79 | 1.00 | 0.87 | 5/2/0 | 0.0 |
| retrieval attribution | alarm | 4 | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] | 0.25 | 1.00 | 0.39 | 5/15/0 | 0.0 |
| LLM self-explanation | alarm | 4 | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] | 0.88 | 0.88 | 0.83 | 4/1/1 | 1.0 |
| ancestry attribution (oracle root) | store | 4 | 1.2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 | 1.00 | 1.00 | 5/0/0 | 0.0 |
| MemHeal counterfactual | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 4 | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] | 1.00 | 0.62 | 0.67 | 3/0/2 | 12.0 |
| MemHeal incident (full quarantine set) | alarm | 4 | 1.0 | 1.00 [1.00, 1.00] | 0.80 [0.57, 1.00] | 0.89 [0.73, 1.00] | 1.00 | 0.88 | 0.92 | 4/0/1 | 22.5 |

## 2. Does retrieval mean causation? (8.4)

4 incidents; 4 with a counterfactual ablation; 4 whose frozen baseline replay was harmful again (0 baselines were not harmful, so nothing was ablated). Of 20 retrieved memories in those incidents, removing 3 flipped the decision (score >= 0.5 over its replays; 4 flipped on the first replay alone).

### 2a. Retrieved memories at the harmful action, by oracle label

| label | retrieved | ablated | flipped | flip rate | flipped on 1st replay |
|---|---|---|---|---|---|
| descendant:harmful | 1 | 1 | 1 | 100% (1/1) | 1 |
| descendant:mixed | 4 | 4 | 2 | 50% (2/4) | 3 |
| non_lineage:clean | 15 | 15 | 0 | 0% (0/15) | 0 |

### 2b. Retrieval attribution vs counterfactual (action scope, incident runs)

| scoring | method | blamed/run | precision [95% CI] | recall [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|
| primary | retrieval attribution | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] |
| primary | LLM self-explanation | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] |
| primary | MemHeal counterfactual | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] |
| primary | MemHeal counterfactual, candidates only | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] |
| secondary | retrieval attribution | 5.0 | 0.25 [0.20, 0.35] | 1.00 [1.00, 1.00] | 0.40 [0.33, 0.52] |
| secondary | LLM self-explanation | 1.2 | 0.80 [0.57, 1.00] | 0.80 [0.57, 1.00] | 0.80 [0.67, 1.00] |
| secondary | MemHeal counterfactual | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] |
| secondary | MemHeal counterfactual, candidates only | 0.8 | 1.00 [1.00, 1.00] | 0.60 [0.25, 1.00] | 0.75 [0.40, 1.00] |

### 2c. Incidents with no single culprit

1 of 4 explained incidents have no memory whose removal alone flips the decision (1 judged on the first replay alone; 1 of them read an attacker page live in the baseline replay). Flipping memories per incident: 0: 1, 1: 3.

| baseline replay; redundancy | incidents |
|---|---|
| read attacker page; 1 attack-carrying in frozen set | 1 |

## 3. Candidate generation (5.3)

Store scope over all runs for size and recall; counterfactual coverage and replays over the explained incidents. 'replays saved' = ablation replays a filter on this signal would not have run (the frozen set's memories without the signal). retrieved_at_action is every ablated memory by construction.

| signal | candidates/run | share of store | recall of positives | precision | cf flips covered | replays saved | replay LLM calls saved |
|---|---|---|---|---|---|---|---|
| retrieved_at_action | 5.0 | 27% (20/73) | 100% (5/5) | 0.25 | 100% (3/3) | 0% (0/24) | 0% (0/79) |
| external_source | 6.5 | 36% (26/73) | 100% (5/5) | 0.19 | 100% (3/3) | 54% (13/24) | 53% (42/79) |
| conflict_with_stable | 0.8 | 4% (3/73) | 0% (0/5) | 0.00 | 0% (0/3) | 100% (24/24) | 100% (79/79) |
| security_field | 4.8 | 26% (19/73) | 100% (5/5) | 0.26 | 100% (3/3) | 58% (14/24) | 59% (47/79) |
| union: external_source + conflict_with_stable + security_field | 8.2 | 45% (33/73) | 100% (5/5) | 0.15 | 100% (3/3) | 54% (13/24) | 53% (42/79) |
| union: all four | 11.5 | 63% (46/73) | 100% (5/5) | 0.11 | 100% (3/3) | 0% (0/24) | 0% (0/79) |

