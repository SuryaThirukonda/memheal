# Benign claim preservation (BCP) and attack claim removal (ACR)

Computed post hoc from the stores by `bcp.py` (metric: `memheal/preservation.py`); no model calls. For each record, the affected memories are the oracle poison lineage of the pre-target snapshot plus every memory the recovery quarantined or replaced, counted when 'retrievable' before recovery. Their claims (deterministic sentence/clause split, deduplicated) are labelled with the scenario's AttackSpec: attack (names the attacker), ambiguous (undermines the legitimate system without naming a replacement; in neither rate) or benign. A claim is kept if its words appear in an active memory after recovery, or if an active claim with the same label has cosine >= 0.85 (all-MiniLM-L6-v2). BCP = kept benign / benign; ACR = attack claims no longer kept / attack claims. 'Exact only' drops the cosine route: stricter for BCP, more lenient for ACR. 'Lineage claims only' counts benign claims of lineage memories alone, the same claims for every configuration of a run, so it compares modes on one denominator.

Rates pool claims over runs. Wilson intervals treat claims as independent; the run-bootstrap intervals resample runs and are the honest ones (claims of one run move together). Cells over fewer than 10 runs say 'small sample'.

## Inputs and coverage

| records file | state | records | checkpoint 0 | later checkpoints (not measured) | measured | missing copy | missing snapshot | other failures | reinfected copies undone | --depth stores rebuilt | copy quarantine != record n_quarantined | copy named by record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| results/recovery/records.jsonl | ok | 3069 | 1581 | 1488 | 1581 | 0 | 0 | 0 | 371 | 0 | 0 | 0 |
| results/claims/records.jsonl | ok | 40 | 40 | 0 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

A copy whose quarantine count differs from its record's n_quarantined may have been overwritten by a later sweep of the same configuration (or quarantined memories a verify round had itself written, which the copy count leaves out).

## claims: poisoned runs

| configuration | runs | BCP (Wilson 95%) | BCP run-bootstrap 95% | BCP exact only | BCP lineage claims only | ACR (Wilson 95%) | ACR run-bootstrap 95% | ACR exact only | ambiguous claims removed | not measured |
|---|---|---|---|---|---|---|---|---|---|---|
| selective[cf] | 8 (small sample) | 0% (0/9) [0, 30] | [0, 0] | 0% (0/9) [0, 30] | 0% (0/9) [0, 30] | 100% (10/10) [72, 100] | [100, 100] | 100% (10/10) [72, 100] | 100% (5/5) [57, 100] |  |
| selective[oracle] | 8 (small sample) | 0% (0/9) [0, 30] | [0, 0] | 0% (0/9) [0, 30] | 0% (0/9) [0, 30] | 100% (10/10) [72, 100] | [100, 100] | 100% (10/10) [72, 100] | 100% (5/5) [57, 100] |  |
| selective_recon[oracle] | 8 (small sample) | 33% (3/9) [12, 65] | [9, 60] | 0% (0/9) [0, 30] | 33% (3/9) [12, 65] | 100% (10/10) [72, 100] | [100, 100] | 100% (10/10) [72, 100] | 100% (5/5) [57, 100] |  |
| selective_claims[cf] | 8 (small sample) | 89% (8/9) [56, 98] | [70, 100] | 56% (5/9) [27, 81] | 89% (8/9) [56, 98] | 90% (9/10) [60, 98] | [73, 100] | 90% (9/10) [60, 98] | 0% (0/5) [0, 43] |  |
| selective_claims[oracle] | 8 (small sample) | 89% (8/9) [56, 98] | [70, 100] | 56% (5/9) [27, 81] | 89% (8/9) [56, 98] | 100% (10/10) [72, 100] | [100, 100] | 100% (10/10) [72, 100] | 100% (5/5) [57, 100] |  |

### claims: poisoned runs, paired differences against selective[oracle]

| A | B | runs | BCP A-B (pp) | BCP lineage-only A-B (pp) | ACR A-B (pp) |
|---|---|---|---|---|---|
| selective[cf] | selective[oracle] | 8 (small sample) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| selective_recon[oracle] | selective[oracle] | 8 (small sample) | +33.3 [+9.1, +60.0] | +33.3 [+9.1, +60.0] | +0.0 [+0.0, +0.0] |
| selective_claims[cf] | selective[oracle] | 8 (small sample) | +88.9 [+70.0, +100.0] | +88.9 [+70.0, +100.0] | -10.0 [-27.3, +0.0] |
| selective_claims[oracle] | selective[oracle] | 8 (small sample) | +88.9 [+70.0, +100.0] | +88.9 [+70.0, +100.0] | +0.0 [+0.0, +0.0] |

A minus B on the runs both measured, percentile bootstrap 95% over runs.

## recovery: poisoned runs

| configuration | runs | BCP (Wilson 95%) | BCP run-bootstrap 95% | BCP exact only | BCP lineage claims only | ACR (Wilson 95%) | ACR run-bootstrap 95% | ACR exact only | ambiguous claims removed | not measured |
|---|---|---|---|---|---|---|---|---|---|---|
| none (live) | 93 | 100% (137/137) [97, 100] | [100, 100] | 100% (137/137) [97, 100] | 100% (137/137) [97, 100] | 0% (0/118) [0, 3] | [0, 0] | 0% (0/118) [0, 3] | 0% (0/49) [0, 7] |  |
| none | 93 | 100% (137/137) [97, 100] | [100, 100] | 100% (137/137) [97, 100] | 100% (137/137) [97, 100] | 0% (0/118) [0, 3] | [0, 0] | 0% (0/118) [0, 3] | 0% (0/49) [0, 7] |  |
| reset (live) | 93 | 0% (0/2242) [0, 0] | [0, 0] | 0% (0/2242) [0, 0] | 0% (0/137) [0, 3] | 100% (121/121) [97, 100] | [100, 100] | 100% (121/121) [97, 100] | 100% (50/50) [93, 100] |  |
| reset | 93 | 0% (0/2242) [0, 0] | [0, 0] | 0% (0/2242) [0, 0] | 0% (0/137) [0, 3] | 100% (121/121) [97, 100] | [100, 100] | 100% (121/121) [97, 100] | 100% (50/50) [93, 100] |  |
| root | 93 | 99% (135/137) [95, 100] | [96, 100] | 99% (135/137) [95, 100] | 99% (135/137) [95, 100] | 3% (3/118) [1, 7] | [0, 6] | 3% (4/118) [1, 8] | 4% (2/49) [1, 14] |  |
| ancestry | 93 | 1% (1/137) [0, 4] | [0, 3] | 0% (0/137) [0, 3] | 1% (1/137) [0, 4] | 99% (117/118) [95, 100] | [98, 100] | 100% (118/118) [97, 100] | 100% (49/49) [93, 100] |  |
| selective[cf] | 93 | 48% (66/137) [40, 56] | [36, 59] | 47% (65/137) [39, 56] | 48% (66/137) [40, 56] | 68% (80/118) [59, 76] | [58, 78] | 69% (81/118) [60, 76] | 67% (33/49) [53, 79] |  |
| selective[oracle] | 93 | 23% (32/137) [17, 31] | [14, 32] | 23% (31/137) [16, 30] | 23% (32/137) [17, 31] | 99% (117/118) [95, 100] | [98, 100] | 100% (118/118) [97, 100] | 92% (45/49) [81, 97] |  |
| selective[oracle]+immunize | 93 | 23% (32/137) [17, 31] | [14, 32] | 23% (31/137) [16, 30] | 23% (32/137) [17, 31] | 99% (117/118) [95, 100] | [98, 100] | 100% (118/118) [97, 100] | 92% (45/49) [81, 97] |  |
| selective_recon[oracle] | 93 | 36% (49/137) [28, 44] | [26, 44] | 25% (34/137) [18, 33] | 36% (49/137) [28, 44] | 99% (117/118) [95, 100] | [98, 100] | 100% (118/118) [97, 100] | 90% (44/49) [78, 96] |  |
| unmerge[cf] | 93 | 57% (78/137) [49, 65] | [47, 66] | 50% (69/137) [42, 59] | 57% (78/137) [49, 65] | 68% (80/118) [59, 76] | [58, 78] | 69% (81/118) [60, 76] | 65% (32/49) [51, 77] |  |
| unmerge[oracle] | 93 | 39% (54/137) [32, 48] | [30, 48] | 28% (38/137) [21, 36] | 39% (54/137) [32, 48] | 99% (117/118) [95, 100] | [97, 100] | 100% (118/118) [97, 100] | 90% (44/49) [78, 96] |  |
| incident[cf] | 93 | 46% (68/148) [38, 54] | [36, 56] | 45% (67/148) [37, 53] | 50% (68/137) [41, 58] | 65% (77/118) [56, 73] | [55, 75] | 66% (78/118) [57, 74] | 61% (30/49) [47, 74] |  |
| incident[cf]+verify | 93 | 37% (58/157) [30, 45] | [27, 47] | 36% (57/157) [29, 44] | 42% (58/137) [34, 51] | 70% (83/118) [62, 78] | [61, 79] | 71% (84/118) [62, 79] | 65% (32/49) [51, 77] |  |

### recovery: poisoned runs, paired differences against selective[oracle]

| A | B | runs | BCP A-B (pp) | BCP lineage-only A-B (pp) | ACR A-B (pp) |
|---|---|---|---|---|---|
| none | selective[oracle] | 93 | +76.6 [+67.6, +86.0] | +76.6 [+67.6, +86.0] | -99.2 [-100.0, -97.5] |
| reset | selective[oracle] | 93 | -23.4 [-32.4, -14.0] | -23.4 [-32.4, -14.0] | +0.8 [+0.0, +2.5] |
| root | selective[oracle] | 93 | +75.2 [+66.7, +84.4] | +75.2 [+66.7, +84.4] | -96.6 [-99.2, -93.4] |
| ancestry | selective[oracle] | 93 | -22.6 [-31.9, -13.1] | -22.6 [-31.9, -13.1] | +0.0 [+0.0, +0.0] |
| selective[cf] | selective[oracle] | 93 | +24.8 [+16.3, +34.3] | +24.8 [+16.3, +34.3] | -31.4 [-40.7, -21.8] |
| selective[oracle]+immunize | selective[oracle] | 93 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| selective_recon[oracle] | selective[oracle] | 93 | +12.4 [+7.6, +17.8] | +12.4 [+7.6, +17.8] | +0.0 [+0.0, +0.0] |
| unmerge[cf] | selective[oracle] | 93 | +33.6 [+25.2, +42.7] | +33.6 [+25.2, +42.7] | -31.4 [-40.7, -21.8] |
| unmerge[oracle] | selective[oracle] | 93 | +16.1 [+10.6, +22.1] | +16.1 [+10.6, +22.1] | +0.0 [+0.0, +0.0] |
| incident[cf] | selective[oracle] | 93 | +22.6 [+13.3, +32.6] | +26.3 [+17.8, +35.9] | -33.9 [-43.5, -24.3] |
| incident[cf]+verify | selective[oracle] | 93 | +13.6 [+6.1, +22.7] | +19.0 [+11.7, +27.3] | -28.8 [-37.6, -19.5] |

A minus B on the runs both measured, percentile bootstrap 95% over runs.

## recovery: clean runs (false-alarm control: nothing is attack, every loss is collateral)

| configuration | runs | BCP (Wilson 95%) | BCP run-bootstrap 95% | BCP exact only | BCP lineage claims only | ACR (Wilson 95%) | ACR run-bootstrap 95% | ACR exact only | ambiguous claims removed | not measured |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 93 | - | - | - | - | - | - | - | - |  |
| selective[cf]+verify | 93 | 2% (6/302) [1, 4] | [1, 4] | 0% (0/302) [0, 1] | - | - | - | - | 100% (4/4) [51, 100] |  |
| incident[cf]+verify | 93 | 2% (6/296) [1, 4] | [1, 4] | 0% (0/296) [0, 1] | - | - | - | - | 100% (4/4) [51, 100] |  |

