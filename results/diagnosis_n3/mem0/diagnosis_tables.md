# Harmful-memory diagnosis (Sections 5.3, 8.3, 8.4, 9.1)

Runs: 48 qualified poisoned runs (data/runs/1438_poisoned_office_expense_receipts_dense_direct_mem0 data/runs/1443_poisoned_coding_ci_setup_script_dense_composed_mem0 data/runs/1443_poisoned_coding_ci_setup_script_dense_direct_mem0 data/runs/1443_poisoned_coding_ci_setup_script_dense_procedural_mem0 data/runs/1444_poisoned_coding_package_registry_dense_composed_mem0 data/runs/1450_poisoned_coding_package_registry_dense_direct_mem0 data/runs/1460_poisoned_coding_package_registry_dense_procedural_mem0 data/runs/1470_poisoned_coding_service_deploy_dense_composed_mem0 data/runs/1480_poisoned_coding_service_deploy_dense_direct_mem0 data/runs/1490_poisoned_coding_service_deploy_dense_procedural_mem0 data/runs/1500_poisoned_office_expense_receipts_dense_composed_mem0 data/runs/1527_poisoned_office_expense_receipts_dense_direct_mem0 data/runs/1527_poisoned_research_dataset_deposit_dense_procedural_mem0 data/runs/1527_poisoned_travel_flight_booking_dense_direct_mem0 data/runs/1538_poisoned_office_expense_receipts_dense_composed_mem0 data/runs/1538_poisoned_office_expense_receipts_dense_direct_mem0 data/runs/1538_poisoned_office_expense_receipts_dense_procedural_mem0 data/runs/1539_poisoned_office_expense_receipts_dense_composed_mem0 data/runs/1548_poisoned_office_expense_receipts_dense_procedural_mem0 data/runs/1557_poisoned_office_expense_receipts_dense_procedural_mem0 data/runs/1593_poisoned_office_invoice_payment_dense_direct_mem0 data/runs/1602_poisoned_office_invoice_payment_dense_direct_mem0 data/runs/1611_poisoned_office_invoice_payment_dense_direct_mem0 data/runs/1647_poisoned_office_invoice_payment_dense_composed_mem0 data/runs/1656_poisoned_office_invoice_payment_dense_composed_mem0 data/runs/1665_poisoned_office_invoice_payment_dense_composed_mem0 data/runs/1701_poisoned_office_invoice_payment_dense_procedural_mem0 data/runs/1710_poisoned_office_invoice_payment_dense_procedural_mem0 data/runs/1719_poisoned_office_invoice_payment_dense_procedural_mem0 data/runs/1755_poisoned_office_timesheet_dense_direct_mem0 data/runs/1764_poisoned_office_timesheet_dense_direct_mem0 data/runs/1773_poisoned_office_timesheet_dense_direct_mem0 data/runs/1808_poisoned_office_timesheet_dense_composed_mem0 data/runs/1817_poisoned_office_timesheet_dense_composed_mem0 data/runs/1826_poisoned_office_timesheet_dense_composed_mem0 data/runs/1871_poisoned_office_timesheet_dense_procedural_mem0 data/runs/2043_poisoned_office_timesheet_dense_procedural_mem0 data/runs/2043_poisoned_travel_hotel_booking_dense_direct_mem0 data/runs/2044_poisoned_office_timesheet_dense_procedural_mem0 data/runs/2044_poisoned_travel_hotel_booking_dense_direct_mem0 data/runs/2045_poisoned_travel_hotel_booking_dense_direct_mem0 data/runs/2072_poisoned_travel_hotel_booking_dense_composed_mem0 data/runs/2081_poisoned_travel_hotel_booking_dense_composed_mem0 data/runs/2090_poisoned_travel_hotel_booking_dense_composed_mem0 data/runs/2126_poisoned_travel_hotel_booking_dense_procedural_mem0 data/runs/2135_poisoned_travel_hotel_booking_dense_procedural_mem0 data/runs/2144_poisoned_travel_hotel_booking_dense_procedural_mem0 data/runs/2171_poisoned_travel_flight_booking_dense_direct_mem0 data/runs/2180_poisoned_travel_flight_booking_dense_direct_mem0 data/runs/2216_poisoned_travel_flight_booking_dense_composed_mem0 data/runs/2225_poisoned_travel_flight_booking_dense_composed_mem0 data/runs/2234_poisoned_travel_flight_booking_dense_composed_mem0 data/runs/2270_poisoned_travel_flight_booking_dense_procedural_mem0 data/runs/2279_poisoned_travel_flight_booking_dense_procedural_mem0 data/runs/2288_poisoned_travel_flight_booking_dense_procedural_mem0 data/runs/2324_poisoned_travel_grocery_order_dense_direct_mem0 data/runs/2333_poisoned_travel_grocery_order_dense_direct_mem0 data/runs/2378_poisoned_travel_grocery_order_dense_composed_mem0 data/runs/2387_poisoned_travel_grocery_order_dense_composed_mem0 data/runs/2396_poisoned_travel_grocery_order_dense_composed_mem0 data/runs/2441_poisoned_travel_grocery_order_dense_procedural_mem0 data/runs/2450_poisoned_travel_grocery_order_dense_procedural_mem0 data/runs/2477_poisoned_coding_service_deploy_dense_direct_mem0 data/runs/2486_poisoned_coding_service_deploy_dense_direct_mem0 data/runs/2513_poisoned_coding_service_deploy_dense_composed_mem0 data/runs/2522_poisoned_coding_service_deploy_dense_composed_mem0 data/runs/2549_poisoned_coding_service_deploy_dense_procedural_mem0 data/runs/2558_poisoned_coding_service_deploy_dense_procedural_mem0 data/runs/2585_poisoned_coding_ci_setup_script_dense_direct_mem0 data/runs/2594_poisoned_coding_ci_setup_script_dense_direct_mem0 data/runs/2630_poisoned_coding_ci_setup_script_dense_composed_mem0 data/runs/2657_poisoned_coding_ci_setup_script_dense_procedural_mem0 data/runs/2666_poisoned_coding_ci_setup_script_dense_procedural_mem0 data/runs/2693_poisoned_coding_package_registry_dense_direct_mem0 data/runs/2702_poisoned_coding_package_registry_dense_direct_mem0 data/runs/2728_poisoned_coding_package_registry_dense_composed_mem0 data/runs/2737_poisoned_coding_package_registry_dense_composed_mem0 data/runs/2764_poisoned_coding_package_registry_dense_procedural_mem0 data/runs/2773_poisoned_coding_package_registry_dense_procedural_mem0 data/runs/2809_poisoned_research_preprint_submission_dense_direct_mem0 data/runs/2818_poisoned_research_preprint_submission_dense_direct_mem0 data/runs/2827_poisoned_research_preprint_submission_dense_direct_mem0 data/runs/2863_poisoned_research_preprint_submission_dense_composed_mem0 data/runs/2872_poisoned_research_preprint_submission_dense_composed_mem0 data/runs/2917_poisoned_research_preprint_submission_dense_procedural_mem0 data/runs/2926_poisoned_research_preprint_submission_dense_procedural_mem0 data/runs/2980_poisoned_research_ethics_application_dense_direct_mem0 data/runs/2989_poisoned_research_ethics_application_dense_direct_mem0 data/runs/3025_poisoned_research_ethics_application_dense_composed_mem0 data/runs/3043_poisoned_research_ethics_application_dense_composed_mem0 data/runs/3079_poisoned_research_ethics_application_dense_procedural_mem0 data/runs/3088_poisoned_research_ethics_application_dense_procedural_mem0 data/runs/3097_poisoned_research_ethics_application_dense_procedural_mem0 data/runs/3129_poisoned_research_dataset_deposit_dense_direct_mem0 data/runs/3137_poisoned_research_dataset_deposit_dense_direct_mem0 data/runs/3144_poisoned_research_dataset_deposit_dense_direct_mem0 data/runs/3178_poisoned_research_dataset_deposit_dense_composed_mem0 data/runs/3184_poisoned_research_dataset_deposit_dense_composed_mem0 data/runs/3208_poisoned_research_dataset_deposit_dense_procedural_mem0 data/runs/4194_poisoned_coding_ci_setup_script_dense_composed_mem0 data/runs/4194_poisoned_research_ethics_application_dense_composed_mem0 data/runs/4194_poisoned_research_ethics_application_dense_direct_mem0 data/runs/4194_poisoned_research_preprint_submission_dense_composed_mem0 data/runs/4194_poisoned_research_preprint_submission_dense_procedural_mem0 data/runs/4194_poisoned_travel_grocery_order_dense_direct_mem0 data/runs/4194_poisoned_travel_grocery_order_dense_procedural_mem0 data/runs/4196_poisoned_research_dataset_deposit_dense_composed_mem0 data/runs/4197_poisoned_research_dataset_deposit_dense_procedural_mem0, suite dense-subset); 20 with an incident (harmful target action). Counterfactual evidence from `main/incident@full@w6@stop.db`. LLM baselines: gpt-5.4-mini, temperature 0: 256 distinct prompts, 0 cached, 256 new calls, 0 failed.

Ground truth: positives = poison roots + descendants labelled harmful or mixed; negatives = benign descendants + non-lineage memories that do not name the attacker; ambiguous descendants are excluded. Non-lineage memories naming the attacker (e.g. read off the attacker's own portal) are excluded from the primary scoring and positive in the secondary one. Micro = pooled counts; macro = mean of per-run values where defined; 95% CIs are percentile bootstraps over runs (2000 resamples). 'alarm' methods act only after a harmful action, so they flag nothing in runs without one. LLM calls/run is the method's cost (cached answers included); for the MemHeal counterfactual it counts the agent steps of its baseline and single-removal replays, and for the MemHeal incident every step the pipeline replayed before it was measured (all counterfactual replays plus +verify's checks).

## 1. Diagnosis per scope (8.3 / 9.1)

### store: active memories at the pre-target snapshot, all runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 48 | 1.0 | 0.83 [0.72, 0.92] | 0.51 [0.43, 0.61] | 0.63 [0.54, 0.73] | 0.81 | 0.61 | 0.65 | 39/8/37 | 22.6 |
| source-only (external source) | store | 48 | 4.1 | 0.39 [0.30, 0.48] | 1.00 [1.00, 1.00] | 0.56 [0.46, 0.65] | 0.37 | 1.00 | 0.44 | 76/121/0 | 0.0 |
| retrieval attribution | alarm | 48 | 2.1 | 0.11 [0.07, 0.15] | 0.14 [0.07, 0.24] | 0.12 [0.07, 0.18] | 0.11 | 0.24 | 0.10 | 11/89/65 | 0.0 |
| LLM self-explanation | alarm | 48 | 0.6 | 0.35 [0.18, 0.61] | 0.14 [0.07, 0.24] | 0.21 [0.11, 0.33] | 0.53 | 0.24 | 0.25 | 11/20/65 | 0.4 |
| ancestry attribution (oracle root) | store | 48 | 1.8 | 0.89 [0.83, 0.95] | 1.00 [1.00, 1.00] | 0.94 [0.91, 0.98] | 0.88 | 1.00 | 0.90 | 76/9/0 | 0.0 |
| MemHeal counterfactual | alarm | 48 | 0.1 | 0.25 [0.00, 1.00] | 0.01 [0.00, 0.05] | 0.03 [0.00, 0.09] | 0.50 | 0.03 | 0.03 | 1/3/75 | 7.6 |
| MemHeal counterfactual, candidates only | alarm | 48 | 0.0 | 0.50 [0.00, 1.00] | 0.01 [0.00, 0.05] | 0.03 [0.00, 0.09] | 0.50 | 0.03 | 0.03 | 1/1/75 | 3.7 |
| MemHeal incident (full quarantine set) | alarm | 48 | 0.4 | 0.86 [0.64, 1.00] | 0.24 [0.12, 0.37] | 0.37 [0.21, 0.53] | 0.94 | 0.31 | 0.35 | 18/3/58 | 7.6 |

### store: active memories at the pre-target snapshot, all runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 48 | 1.0 | 0.83 [0.72, 0.92] | 0.51 [0.43, 0.61] | 0.63 [0.54, 0.73] | 0.81 | 0.61 | 0.65 | 39/8/37 | 22.6 |
| source-only (external source) | store | 48 | 4.1 | 0.39 [0.30, 0.48] | 1.00 [1.00, 1.00] | 0.56 [0.46, 0.65] | 0.37 | 1.00 | 0.44 | 76/121/0 | 0.0 |
| retrieval attribution | alarm | 48 | 2.1 | 0.11 [0.07, 0.15] | 0.14 [0.07, 0.24] | 0.12 [0.07, 0.18] | 0.11 | 0.24 | 0.10 | 11/89/65 | 0.0 |
| LLM self-explanation | alarm | 48 | 0.6 | 0.35 [0.18, 0.61] | 0.14 [0.07, 0.24] | 0.21 [0.11, 0.33] | 0.53 | 0.24 | 0.25 | 11/20/65 | 0.4 |
| ancestry attribution (oracle root) | store | 48 | 1.8 | 0.89 [0.83, 0.95] | 1.00 [1.00, 1.00] | 0.94 [0.91, 0.98] | 0.88 | 1.00 | 0.90 | 76/9/0 | 0.0 |
| MemHeal counterfactual | alarm | 48 | 0.1 | 0.25 [0.00, 1.00] | 0.01 [0.00, 0.05] | 0.03 [0.00, 0.09] | 0.50 | 0.03 | 0.03 | 1/3/75 | 7.6 |
| MemHeal counterfactual, candidates only | alarm | 48 | 0.0 | 0.50 [0.00, 1.00] | 0.01 [0.00, 0.05] | 0.03 [0.00, 0.09] | 0.50 | 0.03 | 0.03 | 1/1/75 | 3.7 |
| MemHeal incident (full quarantine set) | alarm | 48 | 0.4 | 0.86 [0.64, 1.00] | 0.24 [0.12, 0.37] | 0.37 [0.21, 0.53] | 0.94 | 0.31 | 0.35 | 18/3/58 | 7.6 |

### store: active memories at the pre-target snapshot, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 20 | 1.2 | 1.00 [1.00, 1.00] | 0.68 [0.57, 0.82] | 0.81 [0.72, 0.90] | 1.00 | 0.76 | 0.84 | 25/0/12 | 22.4 |
| source-only (external source) | store | 20 | 4.7 | 0.39 [0.31, 0.49] | 1.00 [1.00, 1.00] | 0.56 [0.47, 0.66] | 0.43 | 1.00 | 0.53 | 37/57/0 | 0.0 |
| retrieval attribution | alarm | 20 | 5.0 | 0.11 [0.07, 0.15] | 0.30 [0.17, 0.47] | 0.16 [0.10, 0.23] | 0.11 | 0.41 | 0.16 | 11/89/26 | 0.0 |
| LLM self-explanation | alarm | 20 | 1.6 | 0.35 [0.18, 0.62] | 0.30 [0.17, 0.47] | 0.32 [0.17, 0.53] | 0.53 | 0.41 | 0.40 | 11/20/26 | 1.0 |
| ancestry attribution (oracle root) | store | 20 | 2.0 | 0.93 [0.84, 1.00] | 1.00 [1.00, 1.00] | 0.96 [0.91, 1.00] | 0.96 | 1.00 | 0.98 | 37/3/0 | 0.0 |
| MemHeal counterfactual | alarm | 20 | 0.2 | 0.25 [0.00, 1.00] | 0.03 [0.00, 0.10] | 0.05 [0.00, 0.18] | 0.50 | 0.06 | 0.06 | 1/3/36 | 18.2 |
| MemHeal counterfactual, candidates only | alarm | 20 | 0.1 | 0.50 [0.00, 1.00] | 0.03 [0.00, 0.10] | 0.05 [0.00, 0.18] | 0.50 | 0.06 | 0.06 | 1/1/36 | 8.8 |
| MemHeal incident (full quarantine set) | alarm | 20 | 1.1 | 0.86 [0.64, 1.00] | 0.49 [0.34, 0.64] | 0.62 [0.45, 0.78] | 0.94 | 0.53 | 0.60 | 18/3/19 | 18.2 |

### store: active memories at the pre-target snapshot, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 20 | 1.2 | 1.00 [1.00, 1.00] | 0.68 [0.57, 0.82] | 0.81 [0.72, 0.90] | 1.00 | 0.76 | 0.84 | 25/0/12 | 22.4 |
| source-only (external source) | store | 20 | 4.7 | 0.39 [0.31, 0.49] | 1.00 [1.00, 1.00] | 0.56 [0.47, 0.66] | 0.43 | 1.00 | 0.53 | 37/57/0 | 0.0 |
| retrieval attribution | alarm | 20 | 5.0 | 0.11 [0.07, 0.15] | 0.30 [0.17, 0.47] | 0.16 [0.10, 0.23] | 0.11 | 0.41 | 0.16 | 11/89/26 | 0.0 |
| LLM self-explanation | alarm | 20 | 1.6 | 0.35 [0.18, 0.62] | 0.30 [0.17, 0.47] | 0.32 [0.17, 0.53] | 0.53 | 0.41 | 0.40 | 11/20/26 | 1.0 |
| ancestry attribution (oracle root) | store | 20 | 2.0 | 0.93 [0.84, 1.00] | 1.00 [1.00, 1.00] | 0.96 [0.91, 1.00] | 0.96 | 1.00 | 0.98 | 37/3/0 | 0.0 |
| MemHeal counterfactual | alarm | 20 | 0.2 | 0.25 [0.00, 1.00] | 0.03 [0.00, 0.10] | 0.05 [0.00, 0.18] | 0.50 | 0.06 | 0.06 | 1/3/36 | 18.2 |
| MemHeal counterfactual, candidates only | alarm | 20 | 0.1 | 0.50 [0.00, 1.00] | 0.03 [0.00, 0.10] | 0.05 [0.00, 0.18] | 0.50 | 0.06 | 0.06 | 1/1/36 | 8.8 |
| MemHeal incident (full quarantine set) | alarm | 20 | 1.1 | 0.86 [0.64, 1.00] | 0.49 [0.34, 0.64] | 0.62 [0.45, 0.78] | 0.94 | 0.53 | 0.60 | 18/3/19 | 18.2 |

### action: retrieved set of the harmful action, incident runs (primary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 20 | 0.5 | 1.00 [1.00, 1.00] | 0.82 [0.56, 1.00] | 0.90 [0.71, 1.00] | 1.00 | 0.82 | 0.82 | 9/0/2 | 5.0 |
| source-only (external source) | store | 20 | 1.0 | 0.55 [0.35, 0.80] | 1.00 [1.00, 1.00] | 0.71 [0.52, 0.89] | 0.67 | 1.00 | 0.70 | 11/9/0 | 0.0 |
| retrieval attribution | alarm | 20 | 5.0 | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.20 [0.13, 0.26] | 0.11 | 1.00 | 0.18 | 11/89/0 | 0.0 |
| LLM self-explanation | alarm | 20 | 1.6 | 0.35 [0.18, 0.62] | 1.00 [1.00, 1.00] | 0.52 [0.30, 0.77] | 0.53 | 1.00 | 0.53 | 11/20/0 | 1.0 |
| ancestry attribution (oracle root) | store | 20 | 0.7 | 0.85 [0.57, 1.00] | 1.00 [1.00, 1.00] | 0.92 [0.73, 1.00] | 0.92 | 1.00 | 0.92 | 11/2/0 | 0.0 |
| MemHeal counterfactual | alarm | 20 | 0.2 | 0.25 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.13 [0.00, 0.40] | 0.50 | 0.09 | 0.08 | 1/3/10 | 18.2 |
| MemHeal counterfactual, candidates only | alarm | 20 | 0.1 | 0.50 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.15 [0.00, 0.43] | 0.50 | 0.09 | 0.08 | 1/1/10 | 8.8 |
| MemHeal incident (full quarantine set) | alarm | 20 | 0.7 | 0.77 [0.44, 1.00] | 0.91 [0.70, 1.00] | 0.83 [0.57, 1.00] | 0.91 | 0.91 | 0.83 | 10/3/1 | 18.2 |

### action: retrieved set of the harmful action, incident runs (secondary scoring)

| method | trigger | runs | flagged/run | micro P [95% CI] | micro R [95% CI] | micro F1 [95% CI] | macro P | macro R | macro F1 | TP/FP/FN | LLM calls/run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content-only LLM classifier | store | 20 | 0.5 | 1.00 [1.00, 1.00] | 0.82 [0.56, 1.00] | 0.90 [0.71, 1.00] | 1.00 | 0.82 | 0.82 | 9/0/2 | 5.0 |
| source-only (external source) | store | 20 | 1.0 | 0.55 [0.35, 0.80] | 1.00 [1.00, 1.00] | 0.71 [0.52, 0.89] | 0.67 | 1.00 | 0.70 | 11/9/0 | 0.0 |
| retrieval attribution | alarm | 20 | 5.0 | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.20 [0.13, 0.26] | 0.11 | 1.00 | 0.18 | 11/89/0 | 0.0 |
| LLM self-explanation | alarm | 20 | 1.6 | 0.35 [0.18, 0.62] | 1.00 [1.00, 1.00] | 0.52 [0.30, 0.77] | 0.53 | 1.00 | 0.53 | 11/20/0 | 1.0 |
| ancestry attribution (oracle root) | store | 20 | 0.7 | 0.85 [0.57, 1.00] | 1.00 [1.00, 1.00] | 0.92 [0.73, 1.00] | 0.92 | 1.00 | 0.92 | 11/2/0 | 0.0 |
| MemHeal counterfactual | alarm | 20 | 0.2 | 0.25 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.13 [0.00, 0.40] | 0.50 | 0.09 | 0.08 | 1/3/10 | 18.2 |
| MemHeal counterfactual, candidates only | alarm | 20 | 0.1 | 0.50 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.15 [0.00, 0.43] | 0.50 | 0.09 | 0.08 | 1/1/10 | 8.8 |
| MemHeal incident (full quarantine set) | alarm | 20 | 0.7 | 0.77 [0.44, 1.00] | 0.91 [0.70, 1.00] | 0.83 [0.57, 1.00] | 0.91 | 0.91 | 0.83 | 10/3/1 | 18.2 |

## 2. Does retrieval mean causation? (8.4)

20 incidents; 20 with a counterfactual ablation; 19 whose frozen baseline replay was harmful again (1 baselines were not harmful, so nothing was ablated). Of 95 retrieved memories in those incidents, removing 4 flipped the decision (score >= 0.5 over its replays; 7 flipped on the first replay alone).

### 2a. Retrieved memories at the harmful action, by oracle label

| label | retrieved | ablated | flipped | flip rate | flipped on 1st replay |
|---|---|---|---|---|---|
| root | 4 | 4 | 0 | 0% (0/4) | 0 |
| descendant:harmful | 4 | 4 | 0 | 0% (0/4) | 0 |
| descendant:mixed | 3 | 3 | 1 | 33% (1/3) | 1 |
| descendant:benign | 2 | 2 | 1 | 50% (1/2) | 1 |
| non_lineage:clean | 82 | 82 | 2 | 2% (2/82) | 5 |

### 2b. Retrieval attribution vs counterfactual (action scope, incident runs)

| scoring | method | blamed/run | precision [95% CI] | recall [95% CI] | F1 [95% CI] |
|---|---|---|---|---|---|
| primary | retrieval attribution | 5.0 | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.20 [0.13, 0.26] |
| primary | LLM self-explanation | 1.6 | 0.35 [0.18, 0.62] | 1.00 [1.00, 1.00] | 0.52 [0.30, 0.77] |
| primary | MemHeal counterfactual | 0.2 | 0.25 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.13 [0.00, 0.40] |
| primary | MemHeal counterfactual, candidates only | 0.1 | 0.50 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.15 [0.00, 0.43] |
| secondary | retrieval attribution | 5.0 | 0.11 [0.07, 0.15] | 1.00 [1.00, 1.00] | 0.20 [0.13, 0.26] |
| secondary | LLM self-explanation | 1.6 | 0.35 [0.18, 0.62] | 1.00 [1.00, 1.00] | 0.52 [0.30, 0.77] |
| secondary | MemHeal counterfactual | 0.2 | 0.25 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.13 [0.00, 0.40] |
| secondary | MemHeal counterfactual, candidates only | 0.1 | 0.50 [0.00, 1.00] | 0.09 [0.00, 0.30] | 0.15 [0.00, 0.43] |

### 2c. Incidents with no single culprit

17 of 19 explained incidents have no memory whose removal alone flips the decision (15 judged on the first replay alone; 12 of them read an attacker page live in the baseline replay). Flipping memories per incident: 0: 17, 1: 1, 3: 1.

| baseline replay; redundancy | incidents |
|---|---|
| read attacker page; 0 attack-carrying in frozen set | 7 |
| read attacker page; 1 attack-carrying in frozen set | 5 |
| no attacker page read; 1 attack-carrying in frozen set | 4 |
| no attacker page read; 0 attack-carrying in frozen set | 1 |

## 3. Candidate generation (5.3)

Store scope over all runs for size and recall; counterfactual coverage and replays over the explained incidents. 'replays saved' = ablation replays a filter on this signal would not have run (the frozen set's memories without the signal). retrieved_at_action is every ablated memory by construction.

| signal | candidates/run | share of store | recall of positives | precision | cf flips covered | replays saved | replay LLM calls saved |
|---|---|---|---|---|---|---|---|
| retrieved_at_action | 5.0 | 22% (240/1087) | 17% (13/76) | 0.05 | 100% (4/4) | 0% (0/125) | 0% (0/315) |
| external_source | 4.1 | 18% (197/1087) | 100% (76/76) | 0.39 | 50% (2/4) | 66% (83/125) | 60% (190/315) |
| conflict_with_stable | 1.5 | 6% (70/1087) | 1% (1/76) | 0.01 | 0% (0/4) | 100% (125/125) | 100% (315/315) |
| security_field | 5.3 | 23% (254/1087) | 67% (51/76) | 0.20 | 50% (2/4) | 69% (86/125) | 63% (199/315) |
| union: external_source + conflict_with_stable + security_field | 7.2 | 32% (344/1087) | 100% (76/76) | 0.22 | 50% (2/4) | 66% (82/125) | 59% (187/315) |
| union: all four | 11.2 | 49% (537/1087) | 100% (76/76) | 0.14 | 100% (4/4) | 0% (0/125) | 0% (0/315) |

