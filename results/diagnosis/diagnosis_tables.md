# Harmful-memory diagnosis (Sections 5.3, 8.3, 8.4, 9.1)

Runs: 93 qualified poisoned runs (0959-1197:*_dense_*_evolving, suite dense); 67 with an incident (harmful target action). Counterfactual evidence from `recovery/incident.db`. LLM baselines: gpt-5.4-mini, temperature 0: 947 distinct prompts, 0 cached, 947 new calls, 0 failed.

Ground truth: positives = poison roots + descendants labelled harmful or mixed; negatives = benign descendants + non-lineage memories that do not name the attacker; ambiguous descendants are excluded. Non-lineage memories naming the attacker (e.g. read off the attacker's own portal) are excluded from the primary scoring and positive in the secondary one. Micro = pooled counts; macro = mean of per-run values where defined; 95% CIs are percentile bootstraps over runs (2000 resamples). 'alarm' methods act only after a harmful action, so they flag nothing in runs without one. LLM calls/run is the method's cost (cached answers included); for the MemHeal counterfactual it counts the agent steps of its baseline and single-removal replays, and for the MemHeal incident every step the pipeline replayed before it was measured (all counterfactual replays plus +verify's checks).

## 1. Diagnosis per scope (8.3 / 9.1)

### store: active memories at the pre-target snapshot, all runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 93 | 1.7 | 0.54 [0.47, 0.61] | 0.78 [0.71, 0.85] | 0.64 [0.57, 0.70] | 0.64 | 0.81 | 0.66 | 83/72/23 | 18.4 |
| source-only (external source) | store | 93 | 6.1 | 0.19 [0.17, 0.22] | 1.00 [1.00, 1.00] | 0.32 [0.29, 0.36] | 0.22 | 1.00 | 0.34 | 106/447/0 | 0.0 |
| retrieval attribution | alarm | 93 | 3.6 | 0.19 [0.17, 0.21] | 0.58 [0.49, 0.69] | 0.28 [0.25, 0.31] | 0.19 | 0.62 | 0.23 | 62/269/44 | 0.0 |
| LLM self-explanation | alarm | 93 | 1.0 | 0.66 [0.53, 0.80] | 0.58 [0.48, 0.68] | 0.61 [0.52, 0.71] | 0.82 | 0.61 | 0.59 | 61/32/45 | 0.7 |
| ancestry attribution (oracle root) | store | 93 | 1.3 | 0.91 [0.86, 0.96] | 1.00 [1.00, 1.00] | 0.95 [0.93, 0.98] | 0.95 | 1.00 | 0.96 | 106/10/0 | 0.0 |
| MemHeal counterfactual | alarm | 93 | 0.4 | 0.73 [0.53, 0.93] | 0.25 [0.18, 0.34] | 0.38 [0.28, 0.49] | 0.83 | 0.29 | 0.29 | 27/10/79 | 16.2 |
| MemHeal counterfactual, candidates only | alarm | 93 | 0.3 | 0.96 [0.88, 1.00] | 0.25 [0.18, 0.34] | 0.40 [0.30, 0.51] | 0.96 | 0.29 | 0.29 | 27/1/79 | 8.7 |
| MemHeal incident (full quarantine set) | alarm | 93 | 0.8 | 0.87 [0.75, 0.97] | 0.61 [0.52, 0.71] | 0.72 [0.64, 0.80] | 0.92 | 0.66 | 0.66 | 65/10/41 | 16.2 |

### store: active memories at the pre-target snapshot, all runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 93 | 1.7 | 0.54 [0.47, 0.62] | 0.77 [0.70, 0.84] | 0.63 [0.57, 0.70] | 0.64 | 0.81 | 0.66 | 84/72/25 | 18.4 |
| source-only (external source) | store | 93 | 6.1 | 0.20 [0.17, 0.23] | 1.00 [1.00, 1.00] | 0.33 [0.29, 0.37] | 0.22 | 1.00 | 0.34 | 109/447/0 | 0.0 |
| retrieval attribution | alarm | 93 | 3.6 | 0.19 [0.17, 0.22] | 0.59 [0.50, 0.69] | 0.29 [0.25, 0.33] | 0.19 | 0.62 | 0.23 | 64/269/45 | 0.0 |
| LLM self-explanation | alarm | 93 | 1.0 | 0.66 [0.54, 0.81] | 0.58 [0.49, 0.68] | 0.62 [0.53, 0.71] | 0.82 | 0.61 | 0.59 | 63/32/46 | 0.7 |
| ancestry attribution (oracle root) | store | 93 | 1.3 | 0.91 [0.86, 0.96] | 0.97 [0.92, 1.00] | 0.94 [0.91, 0.97] | 0.95 | 1.00 | 0.96 | 106/10/3 | 0.0 |
| MemHeal counterfactual | alarm | 93 | 0.4 | 0.73 [0.53, 0.93] | 0.25 [0.17, 0.34] | 0.37 [0.27, 0.48] | 0.83 | 0.29 | 0.29 | 27/10/82 | 16.2 |
| MemHeal counterfactual, candidates only | alarm | 93 | 0.3 | 0.96 [0.88, 1.00] | 0.25 [0.17, 0.34] | 0.39 [0.29, 0.50] | 0.96 | 0.29 | 0.29 | 27/1/82 | 8.7 |
| MemHeal incident (full quarantine set) | alarm | 93 | 0.8 | 0.87 [0.75, 0.97] | 0.60 [0.50, 0.70] | 0.71 [0.63, 0.79] | 0.92 | 0.66 | 0.66 | 65/10/44 | 16.2 |

### store: active memories at the pre-target snapshot, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 67 | 1.7 | 0.56 [0.47, 0.65] | 0.80 [0.71, 0.88] | 0.66 [0.58, 0.73] | 0.67 | 0.82 | 0.68 | 63/50/16 | 18.0 |
| source-only (external source) | store | 67 | 5.7 | 0.21 [0.18, 0.24] | 1.00 [1.00, 1.00] | 0.35 [0.31, 0.39] | 0.23 | 1.00 | 0.37 | 79/297/0 | 0.0 |
| retrieval attribution | alarm | 67 | 5.0 | 0.19 [0.17, 0.21] | 0.78 [0.70, 0.88] | 0.30 [0.27, 0.33] | 0.19 | 0.82 | 0.30 | 62/269/17 | 0.0 |
| LLM self-explanation | alarm | 67 | 1.4 | 0.66 [0.53, 0.80] | 0.77 [0.69, 0.87] | 0.71 [0.61, 0.81] | 0.82 | 0.81 | 0.79 | 61/32/18 | 1.0 |
| ancestry attribution (oracle root) | store | 67 | 1.3 | 0.91 [0.85, 0.96] | 1.00 [1.00, 1.00] | 0.95 [0.92, 0.98] | 0.94 | 1.00 | 0.96 | 79/8/0 | 0.0 |
| MemHeal counterfactual | alarm | 67 | 0.6 | 0.73 [0.53, 0.94] | 0.34 [0.24, 0.45] | 0.47 [0.34, 0.59] | 0.83 | 0.38 | 0.39 | 27/10/52 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 67 | 0.4 | 0.96 [0.88, 1.00] | 0.34 [0.24, 0.45] | 0.50 [0.38, 0.62] | 0.96 | 0.38 | 0.39 | 27/1/52 | 12.1 |
| MemHeal incident (full quarantine set) | alarm | 67 | 1.1 | 0.87 [0.75, 0.97] | 0.82 [0.74, 0.90] | 0.84 [0.77, 0.92] | 0.92 | 0.88 | 0.88 | 65/10/14 | 22.5 |

### store: active memories at the pre-target snapshot, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 67 | 1.7 | 0.56 [0.47, 0.65] | 0.78 [0.70, 0.87] | 0.65 [0.58, 0.72] | 0.67 | 0.81 | 0.67 | 64/50/18 | 18.0 |
| source-only (external source) | store | 67 | 5.7 | 0.22 [0.19, 0.25] | 1.00 [1.00, 1.00] | 0.36 [0.31, 0.41] | 0.24 | 1.00 | 0.37 | 82/297/0 | 0.0 |
| retrieval attribution | alarm | 67 | 5.0 | 0.19 [0.17, 0.22] | 0.78 [0.70, 0.87] | 0.31 [0.27, 0.35] | 0.19 | 0.82 | 0.30 | 64/269/18 | 0.0 |
| LLM self-explanation | alarm | 67 | 1.4 | 0.66 [0.54, 0.80] | 0.77 [0.69, 0.86] | 0.71 [0.62, 0.81] | 0.82 | 0.81 | 0.79 | 63/32/19 | 1.0 |
| ancestry attribution (oracle root) | store | 67 | 1.3 | 0.91 [0.85, 0.96] | 0.96 [0.90, 1.00] | 0.93 [0.90, 0.97] | 0.94 | 0.99 | 0.96 | 79/8/3 | 0.0 |
| MemHeal counterfactual | alarm | 67 | 0.6 | 0.73 [0.53, 0.94] | 0.33 [0.23, 0.45] | 0.45 [0.33, 0.58] | 0.83 | 0.38 | 0.39 | 27/10/55 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 67 | 0.4 | 0.96 [0.88, 1.00] | 0.33 [0.23, 0.45] | 0.49 [0.36, 0.61] | 0.96 | 0.38 | 0.39 | 27/1/55 | 12.1 |
| MemHeal incident (full quarantine set) | alarm | 67 | 1.1 | 0.87 [0.75, 0.97] | 0.79 [0.69, 0.90] | 0.83 [0.75, 0.91] | 0.92 | 0.88 | 0.88 | 65/10/17 | 22.5 |

### action: retrieved set of the harmful action, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 67 | 0.8 | 0.96 [0.91, 1.00] | 0.84 [0.75, 0.92] | 0.90 [0.84, 0.95] | 0.97 | 0.85 | 0.84 | 52/2/10 | 5.0 |
| source-only (external source) | store | 67 | 1.5 | 0.63 [0.56, 0.71] | 1.00 [1.00, 1.00] | 0.78 [0.72, 0.83] | 0.70 | 1.00 | 0.77 | 62/36/0 | 0.0 |
| retrieval attribution | alarm | 67 | 5.0 | 0.19 [0.17, 0.21] | 1.00 [1.00, 1.00] | 0.32 [0.28, 0.35] | 0.19 | 1.00 | 0.31 | 62/269/0 | 0.0 |
| LLM self-explanation | alarm | 67 | 1.4 | 0.66 [0.53, 0.80] | 0.98 [0.95, 1.00] | 0.79 [0.69, 0.88] | 0.82 | 0.99 | 0.83 | 61/32/1 | 1.0 |
| ancestry attribution (oracle root) | store | 67 | 1.0 | 0.97 [0.92, 1.00] | 1.00 [1.00, 1.00] | 0.98 [0.96, 1.00] | 0.98 | 1.00 | 0.99 | 62/2/0 | 0.0 |
| MemHeal counterfactual | alarm | 67 | 0.6 | 0.73 [0.53, 0.94] | 0.44 [0.31, 0.56] | 0.55 [0.41, 0.67] | 0.83 | 0.46 | 0.44 | 27/10/35 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 67 | 0.4 | 0.96 [0.88, 1.00] | 0.44 [0.31, 0.56] | 0.60 [0.47, 0.71] | 0.96 | 0.46 | 0.46 | 27/1/35 | 12.1 |
| MemHeal incident (full quarantine set) | alarm | 67 | 1.0 | 0.84 [0.71, 0.97] | 0.87 [0.79, 0.95] | 0.86 [0.77, 0.94] | 0.91 | 0.90 | 0.87 | 54/10/8 | 22.5 |

### action: retrieved set of the harmful action, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 67 | 0.8 | 0.96 [0.91, 1.00] | 0.81 [0.70, 0.92] | 0.88 [0.81, 0.94] | 0.97 | 0.85 | 0.84 | 52/2/12 | 5.0 |
| source-only (external source) | store | 67 | 1.5 | 0.64 [0.57, 0.72] | 1.00 [1.00, 1.00] | 0.78 [0.72, 0.84] | 0.70 | 1.00 | 0.77 | 64/36/0 | 0.0 |
| retrieval attribution | alarm | 67 | 5.0 | 0.19 [0.17, 0.22] | 1.00 [1.00, 1.00] | 0.32 [0.29, 0.36] | 0.19 | 1.00 | 0.31 | 64/269/0 | 0.0 |
| LLM self-explanation | alarm | 67 | 1.4 | 0.66 [0.54, 0.80] | 0.98 [0.95, 1.00] | 0.79 [0.70, 0.89] | 0.82 | 0.99 | 0.83 | 63/32/1 | 1.0 |
| ancestry attribution (oracle root) | store | 67 | 1.0 | 0.97 [0.92, 1.00] | 0.97 [0.91, 1.00] | 0.97 [0.93, 1.00] | 0.98 | 0.99 | 0.98 | 62/2/2 | 0.0 |
| MemHeal counterfactual | alarm | 67 | 0.6 | 0.73 [0.53, 0.94] | 0.42 [0.30, 0.56] | 0.53 [0.40, 0.66] | 0.83 | 0.46 | 0.44 | 27/10/37 | 22.5 |
| MemHeal counterfactual, candidates only | alarm | 67 | 0.4 | 0.96 [0.88, 1.00] | 0.42 [0.30, 0.56] | 0.59 [0.45, 0.71] | 0.96 | 0.46 | 0.46 | 27/1/37 | 12.1 |
| MemHeal incident (full quarantine set) | alarm | 67 | 1.0 | 0.84 [0.71, 0.97] | 0.84 [0.75, 0.94] | 0.84 [0.75, 0.93] | 0.91 | 0.89 | 0.87 | 54/10/10 | 22.5 |

## 2. Does retrieval mean causation? (8.4)

67 incidents; 67 with a counterfactual ablation; 62 whose frozen baseline replay was harmful again (5 baselines were not harmful, so nothing was ablated). Of 310 retrieved memories in those incidents, removing 37 flipped the decision (score >= 0.5 over its replays; 40 flipped on the first replay alone).

### 2a. Retrieved memories at the harmful action, by oracle label

| label | retrieved | ablated | flipped | flip rate | flipped on 1st replay |
|---|---|---|---|---|---|
| root | 2 | 2 | 0 | 0% (0/2) | 0 |
| descendant:harmful | 29 | 29 | 14 | 48% (14/29) | 14 |
| descendant:mixed | 29 | 29 | 13 | 45% (13/29) | 12 |
| descendant:ambiguous | 1 | 1 | 0 | 0% (0/1) | 0 |
| descendant:benign | 2 | 2 | 1 | 50% (1/2) | 1 |
| non_lineage:names_attacker | 2 | 2 | 0 | 0% (0/2) | 0 |
| non_lineage:clean | 245 | 245 | 9 | 4% (9/245) | 13 |

### 2b. Retrieval attribution vs counterfactual (action scope, incident runs)

| scoring | method | blamed/run | precision [95% CI] | recall [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|
| primary | retrieval attribution | 5.0 | 0.19 [0.17, 0.21] | 1.00 [1.00, 1.00] | 0.32 [0.28, 0.35] |
| primary | LLM self-explanation | 1.4 | 0.66 [0.53, 0.80] | 0.98 [0.95, 1.00] | 0.79 [0.69, 0.88] |
| primary | MemHeal counterfactual | 0.6 | 0.73 [0.53, 0.94] | 0.44 [0.31, 0.56] | 0.55 [0.41, 0.67] |
| primary | MemHeal counterfactual, candidates only | 0.4 | 0.96 [0.88, 1.00] | 0.44 [0.31, 0.56] | 0.60 [0.47, 0.71] |
| secondary | retrieval attribution | 5.0 | 0.19 [0.17, 0.22] | 1.00 [1.00, 1.00] | 0.32 [0.29, 0.36] |
| secondary | LLM self-explanation | 1.4 | 0.66 [0.54, 0.80] | 0.98 [0.95, 1.00] | 0.79 [0.70, 0.89] |
| secondary | MemHeal counterfactual | 0.6 | 0.73 [0.53, 0.94] | 0.42 [0.30, 0.56] | 0.53 [0.40, 0.66] |
| secondary | MemHeal counterfactual, candidates only | 0.4 | 0.96 [0.88, 1.00] | 0.42 [0.30, 0.56] | 0.59 [0.45, 0.71] |

### 2c. Incidents with no single culprit

30 of 62 explained incidents have no memory whose removal alone flips the decision (32 judged on the first replay alone; 17 of them read an attacker page live in the baseline replay). Flipping memories per incident: 0: 30, 1: 29, 2: 2, 4: 1.

| baseline replay; redundancy | incidents |
|---|---|
| read attacker page; 1 attack-carrying in frozen set | 12 |
| no attacker page read; 1 attack-carrying in frozen set | 11 |
| read attacker page; 0 attack-carrying in frozen set | 3 |
| read attacker page; 2+ attack-carrying in frozen set | 2 |
| no attacker page read; 2+ attack-carrying in frozen set | 1 |
| no attacker page read; 0 attack-carrying in frozen set | 1 |

## 3. Candidate generation (5.3)

Store scope over all runs for size and recall; counterfactual coverage and replays over the explained incidents. 'replays saved' = ablation replays a filter on this signal would not have run (the frozen set's memories without the signal). retrieved_at_action is every ablated memory by construction.

| signal | candidates/run | share of store | recall of positives | precision | cf flips covered | replays saved | replay LLM calls saved |
|---|---|---|---|---|---|---|---|
| retrieved_at_action | 5.0 | 27% (465/1708) | 68% (72/106) | 0.16 | 100% (37/37) | 0% (0/396) | 0% (0/1304) |
| external_source | 6.1 | 33% (563/1708) | 100% (106/106) | 0.19 | 76% (28/37) | 61% (240/396) | 58% (753/1304) |
| conflict_with_stable | 0.7 | 4% (66/1708) | 3% (3/106) | 0.05 | 0% (0/37) | 99% (392/396) | 99% (1290/1304) |
| security_field | 4.5 | 24% (414/1708) | 97% (103/106) | 0.25 | 76% (28/37) | 64% (253/396) | 61% (799/1304) |
| union: external_source + conflict_with_stable + security_field | 7.9 | 43% (736/1708) | 100% (106/106) | 0.15 | 76% (28/37) | 57% (225/396) | 54% (699/1304) |
| union: all four | 11.2 | 61% (1037/1708) | 100% (106/106) | 0.10 | 100% (37/37) | 0% (0/396) | 0% (0/1304) |

