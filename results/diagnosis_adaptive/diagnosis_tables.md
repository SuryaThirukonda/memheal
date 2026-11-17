# Harmful-memory diagnosis (Sections 5.3, 8.3, 8.4, 9.1)

Runs: 7 qualified poisoned runs (data/runs/1443_poisoned_coding_backup_restore_dense_adaptive_evade_evolving data/runs/1443_poisoned_coding_backup_restore_dense_adaptive_survive_evolving data/runs/1444_poisoned_coding_ci_setup_script_dense_adaptive_evade_evolving data/runs/1444_poisoned_coding_ci_setup_script_dense_adaptive_survive_evolving data/runs/1527_poisoned_office_benefits_enrolment_dense_adaptive_evade_evolving data/runs/1527_poisoned_office_benefits_enrolment_dense_adaptive_survive_evolving data/runs/1527_poisoned_research_conference_abstract_dense_adaptive_survive_evolving, suite dense-adaptive); 6 with an incident (harmful target action). Counterfactual evidence from `recovery/incident.db`. LLM baselines: gpt-5.4-mini, temperature 0: 111 distinct prompts, 0 cached, 111 new calls, 0 failed.

Ground truth: positives = poison roots + descendants labelled harmful or mixed; negatives = benign descendants + non-lineage memories that do not name the attacker; ambiguous descendants are excluded. Non-lineage memories naming the attacker (e.g. read off the attacker's own portal) are excluded from the primary scoring and positive in the secondary one. Micro = pooled counts; macro = mean of per-run values where defined; 95% CIs are percentile bootstraps over runs (2000 resamples). 'alarm' methods act only after a harmful action, so they flag nothing in runs without one. LLM calls/run is the method's cost (cached answers included); for the MemHeal counterfactual it counts the agent steps of its baseline and single-removal replays, and for the MemHeal incident every step the pipeline replayed before it was measured (all counterfactual replays plus +verify's checks).

## 1. Diagnosis per scope (8.3 / 9.1)

### store: active memories at the pre-target snapshot, all runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 7 | 1.0 | 0.43 [0.09, 1.00] | 0.43 [0.12, 0.71] | 0.43 [0.11, 0.67] | 0.50 | 0.42 | 0.33 | 3/4/4 | 18.9 |
| source-only (external source) | store | 7 | 5.6 | 0.18 [0.10, 0.30] | 1.00 [1.00, 1.00] | 0.30 [0.19, 0.47] | 0.22 | 1.00 | 0.34 | 7/32/0 | 0.0 |
| retrieval attribution | alarm | 7 | 4.3 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] | 0.23 | 1.00 | 0.37 | 7/23/0 | 0.0 |
| LLM self-explanation | alarm | 7 | 1.1 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 0.9 |
| ancestry attribution (oracle root) | store | 7 | 1.3 | 0.78 [0.64, 1.00] | 1.00 [1.00, 1.00] | 0.88 [0.78, 1.00] | 0.86 | 1.00 | 0.91 | 7/2/0 | 0.0 |
| MemHeal counterfactual | alarm | 7 | 0.1 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 25.0 |
| MemHeal counterfactual, candidates only | alarm | 7 | 0.1 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 13.9 |
| MemHeal incident (full quarantine set) | alarm | 7 | 0.7 | 1.00 [1.00, 1.00] | 0.71 [0.40, 1.00] | 0.83 [0.57, 1.00] | 1.00 | 0.75 | 0.78 | 5/0/2 | 25.0 |

### store: active memories at the pre-target snapshot, all runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 7 | 1.0 | 0.43 [0.09, 1.00] | 0.43 [0.12, 0.71] | 0.43 [0.11, 0.67] | 0.50 | 0.42 | 0.33 | 3/4/4 | 18.9 |
| source-only (external source) | store | 7 | 5.6 | 0.18 [0.10, 0.30] | 1.00 [1.00, 1.00] | 0.30 [0.19, 0.47] | 0.22 | 1.00 | 0.34 | 7/32/0 | 0.0 |
| retrieval attribution | alarm | 7 | 4.3 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] | 0.23 | 1.00 | 0.37 | 7/23/0 | 0.0 |
| LLM self-explanation | alarm | 7 | 1.1 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 0.9 |
| ancestry attribution (oracle root) | store | 7 | 1.3 | 0.78 [0.64, 1.00] | 1.00 [1.00, 1.00] | 0.88 [0.78, 1.00] | 0.86 | 1.00 | 0.91 | 7/2/0 | 0.0 |
| MemHeal counterfactual | alarm | 7 | 0.1 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 25.0 |
| MemHeal counterfactual, candidates only | alarm | 7 | 0.1 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 13.9 |
| MemHeal incident (full quarantine set) | alarm | 7 | 0.7 | 1.00 [1.00, 1.00] | 0.71 [0.40, 1.00] | 0.83 [0.57, 1.00] | 1.00 | 0.75 | 0.78 | 5/0/2 | 25.0 |

### store: active memories at the pre-target snapshot, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 6 | 1.0 | 0.50 [0.14, 1.00] | 0.43 [0.14, 0.71] | 0.46 [0.14, 0.73] | 0.62 | 0.42 | 0.39 | 3/3/4 | 19.0 |
| source-only (external source) | store | 6 | 5.5 | 0.21 [0.14, 0.35] | 1.00 [1.00, 1.00] | 0.35 [0.24, 0.52] | 0.26 | 1.00 | 0.40 | 7/26/0 | 0.0 |
| retrieval attribution | alarm | 6 | 5.0 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] | 0.23 | 1.00 | 0.37 | 7/23/0 | 0.0 |
| LLM self-explanation | alarm | 6 | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 1.0 |
| ancestry attribution (oracle root) | store | 6 | 1.5 | 0.78 [0.64, 1.00] | 1.00 [1.00, 1.00] | 0.88 [0.78, 1.00] | 0.86 | 1.00 | 0.91 | 7/2/0 | 0.0 |
| MemHeal counterfactual | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 29.2 |
| MemHeal counterfactual, candidates only | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 16.2 |
| MemHeal incident (full quarantine set) | alarm | 6 | 0.8 | 1.00 [1.00, 1.00] | 0.71 [0.43, 1.00] | 0.83 [0.60, 1.00] | 1.00 | 0.75 | 0.78 | 5/0/2 | 29.2 |

### store: active memories at the pre-target snapshot, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 6 | 1.0 | 0.50 [0.14, 1.00] | 0.43 [0.14, 0.71] | 0.46 [0.14, 0.73] | 0.62 | 0.42 | 0.39 | 3/3/4 | 19.0 |
| source-only (external source) | store | 6 | 5.5 | 0.21 [0.14, 0.35] | 1.00 [1.00, 1.00] | 0.35 [0.24, 0.52] | 0.26 | 1.00 | 0.40 | 7/26/0 | 0.0 |
| retrieval attribution | alarm | 6 | 5.0 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] | 0.23 | 1.00 | 0.37 | 7/23/0 | 0.0 |
| LLM self-explanation | alarm | 6 | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 1.0 |
| ancestry attribution (oracle root) | store | 6 | 1.5 | 0.78 [0.64, 1.00] | 1.00 [1.00, 1.00] | 0.88 [0.78, 1.00] | 0.86 | 1.00 | 0.91 | 7/2/0 | 0.0 |
| MemHeal counterfactual | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 29.2 |
| MemHeal counterfactual, candidates only | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 16.2 |
| MemHeal incident (full quarantine set) | alarm | 6 | 0.8 | 1.00 [1.00, 1.00] | 0.71 [0.43, 1.00] | 0.83 [0.60, 1.00] | 1.00 | 0.75 | 0.78 | 5/0/2 | 29.2 |

### action: retrieved set of the harmful action, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 6 | 0.5 | 1.00 [1.00, 1.00] | 0.43 [0.14, 0.71] | 0.60 [0.25, 0.83] | 1.00 | 0.42 | 0.44 | 3/0/4 | 5.0 |
| source-only (external source) | store | 6 | 1.8 | 0.64 [0.46, 0.88] | 1.00 [1.00, 1.00] | 0.78 [0.63, 0.93] | 0.75 | 1.00 | 0.83 | 7/4/0 | 0.0 |
| retrieval attribution | alarm | 6 | 5.0 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] | 0.23 | 1.00 | 0.37 | 7/23/0 | 0.0 |
| LLM self-explanation | alarm | 6 | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 1.0 |
| ancestry attribution (oracle root) | store | 6 | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 0.0 |
| MemHeal counterfactual | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 29.2 |
| MemHeal counterfactual, candidates only | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 16.2 |
| MemHeal incident (full quarantine set) | alarm | 6 | 0.8 | 1.00 [1.00, 1.00] | 0.71 [0.43, 1.00] | 0.83 [0.60, 1.00] | 1.00 | 0.75 | 0.78 | 5/0/2 | 29.2 |

### action: retrieved set of the harmful action, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 6 | 0.5 | 1.00 [1.00, 1.00] | 0.43 [0.14, 0.71] | 0.60 [0.25, 0.83] | 1.00 | 0.42 | 0.44 | 3/0/4 | 5.0 |
| source-only (external source) | store | 6 | 1.8 | 0.64 [0.46, 0.88] | 1.00 [1.00, 1.00] | 0.78 [0.63, 0.93] | 0.75 | 1.00 | 0.83 | 7/4/0 | 0.0 |
| retrieval attribution | alarm | 6 | 5.0 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] | 0.23 | 1.00 | 0.37 | 7/23/0 | 0.0 |
| LLM self-explanation | alarm | 6 | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 1.0 |
| ancestry attribution (oracle root) | store | 6 | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.92 | 1.00 | 0.94 | 7/1/0 | 0.0 |
| MemHeal counterfactual | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 29.2 |
| MemHeal counterfactual, candidates only | alarm | 6 | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] | 1.00 | 0.17 | 0.17 | 1/0/6 | 16.2 |
| MemHeal incident (full quarantine set) | alarm | 6 | 0.8 | 1.00 [1.00, 1.00] | 0.71 [0.43, 1.00] | 0.83 [0.60, 1.00] | 1.00 | 0.75 | 0.78 | 5/0/2 | 29.2 |

## 2. Does retrieval mean causation? (8.4)

6 incidents; 6 with a counterfactual ablation; 6 whose frozen baseline replay was harmful again (0 baselines were not harmful, so nothing was ablated). Of 30 retrieved memories in those incidents, removing 1 flipped the decision (score >= 0.5 over its replays; 3 flipped on the first replay alone).

### 2a. Retrieved memories at the harmful action, by oracle label

| label | retrieved | ablated | flipped | flip rate | flipped on 1st replay |
|---|---|---|---|---|---|
| root | 1 | 1 | 0 | 0% (0/1) | 0 |
| descendant:harmful | 3 | 3 | 1 | 33% (1/3) | 1 |
| descendant:mixed | 3 | 3 | 0 | 0% (0/3) | 0 |
| descendant:benign | 1 | 1 | 0 | 0% (0/1) | 0 |
| non_lineage:clean | 22 | 22 | 0 | 0% (0/22) | 2 |

### 2b. Retrieval attribution vs counterfactual (action scope, incident runs)

| scoring | method | blamed/run | precision [95% CI] | recall [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|
| primary | retrieval attribution | 5.0 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] |
| primary | LLM self-explanation | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] |
| primary | MemHeal counterfactual | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] |
| primary | MemHeal counterfactual, candidates only | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] |
| secondary | retrieval attribution | 5.0 | 0.23 [0.20, 0.30] | 1.00 [1.00, 1.00] | 0.38 [0.33, 0.46] |
| secondary | LLM self-explanation | 1.3 | 0.88 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] |
| secondary | MemHeal counterfactual | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] |
| secondary | MemHeal counterfactual, candidates only | 0.2 | 1.00 [1.00, 1.00] | 0.14 [0.00, 0.50] | 0.25 [0.00, 0.67] |

### 2c. Incidents with no single culprit

5 of 6 explained incidents have no memory whose removal alone flips the decision (4 judged on the first replay alone; 5 of them read an attacker page live in the baseline replay). Flipping memories per incident: 0: 5, 1: 1.

| baseline replay; redundancy | incidents |
|---|---|
| read attacker page; 1 attack-carrying in frozen set | 5 |

## 3. Candidate generation (5.3)

Store scope over all runs for size and recall; counterfactual coverage and replays over the explained incidents. 'replays saved' = ablation replays a filter on this signal would not have run (the frozen set's memories without the signal). retrieved_at_action is every ablated memory by construction.

| signal | candidates/run | share of store | recall of positives | precision | cf flips covered | replays saved | replay LLM calls saved |
|---|---|---|---|---|---|---|---|
| retrieved_at_action | 5.0 | 27% (35/132) | 100% (7/7) | 0.20 | 100% (1/1) | 0% (0/42) | 0% (0/152) |
| external_source | 5.6 | 30% (39/132) | 100% (7/7) | 0.18 | 100% (1/1) | 55% (23/42) | 51% (78/152) |
| conflict_with_stable | 0.9 | 5% (6/132) | 0% (0/7) | 0.00 | 0% (0/1) | 100% (42/42) | 100% (152/152) |
| security_field | 4.1 | 22% (29/132) | 71% (5/7) | 0.17 | 100% (1/1) | 69% (29/42) | 66% (101/152) |
| union: external_source + conflict_with_stable + security_field | 7.7 | 41% (54/132) | 100% (7/7) | 0.13 | 100% (1/1) | 55% (23/42) | 51% (78/152) |
| union: all four | 11.0 | 58% (77/132) | 100% (7/7) | 0.09 | 100% (1/1) | 0% (0/42) | 0% (0/152) |

