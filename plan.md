# Credit Risk Modelling — PD Model
### Home Credit Default Risk | WoE Scorecard + LightGBM Benchmark

**Dataset:** Home Credit Default Risk (Kaggle)
**Target:** `TARGET` — 1 = defaulted, 0 = repaid (8.07% default rate, 307k rows)
**Goal:** Build a production-style PD model. Primary output is a WoE scorecard; LightGBM is a benchmark to understand the interpretability trade-off.

---

## Project Structure

```
Credit-risk-PD/
├── Data/
│   ├── Raw/                        # gitignored
│   └── Processed/                  # gitignored — all notebooks read from here
├── Notebooks/
│   ├── 00_data_reduction.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_woe_scorecard.ipynb
│   ├── 04_lgbm_benchmark.ipynb
│   └── 05_validation.ipynb
├── Outputs/                        # gitignored — plots, tables, model files
├── requirements.txt
└── .gitignore
```

---

## Raw Data Files (Data/Raw/)

| File | Size | Rows | Keep |
|---|---|---|---|
| `application_train.csv` | 166 MB | 307k | Yes — primary table |
| `bureau.csv` | 170 MB | 1.7M | Yes |
| `previous_application.csv` | 405 MB | 1.7M | Yes |
| `installments_payments.csv` | 723 MB | 13M | Yes |
| `credit_card_balance.csv` | 425 MB | 3.8M | Yes |
| `HomeCredit_columns_description.csv` | — | — | Keep — column reference, consult while implementing |

> Note: actual filename is `installments_payments.csv` (double-l), not `instalments_payments.csv`

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Skip `bureau_balance` and `POS_CASH_balance` | Too large for marginal learning value |
| Out-of-time split, not random | How banks validate — earlier data trains, later data tests |
| 15–20 features only | Industry norm; stability matters more than lift |
| WoE scorecard as primary model | Basel IRB standard; interpretable; a real bank artefact |
| LightGBM as benchmark only | Shows the interpretability trade-off, not just one approach |
| `DAYS_EMPLOYED = 365243` → unemployed flag | Sentinel value — numeric treatment would corrupt the feature |
| Missing values get their own WoE bin | Missingness is informative — no bureau history = higher risk |
| Paths via `pathlib.Path` | Portable across OS |

---

## Validation Benchmarks (retail unsecured)

| Metric | Acceptable | Good |
|---|---|---|
| Gini | > 0.35 | > 0.50 |
| KS | > 0.25 | > 0.40 |
| PSI | < 0.10 stable | < 0.25 monitor |

---

## Implementation Checklist

### Notebook 00 — Data Reduction

- [x] Read `application_train.csv`, verify shape (307511, 122) and default rate (~8%)
- [x] Drop document flags (`FLAG_DOCUMENT_2–21`), contact flags, housing detail columns
- [x] Keep 33 essential columns → save `application_slim.parquet`
- [x] Aggregate `bureau.csv` → `bureau_agg.parquet`
  - [x] `bureau_loan_count`, `bureau_active_count`, `bureau_total_debt`, `bureau_total_credit`
  - [x] `bureau_total_overdue`, `bureau_max_overdue`, `bureau_avg_days`, `bureau_cnt_prolong`
  - [x] Derived: `bureau_debt_ratio = bureau_total_debt / bureau_total_credit`
- [x] Aggregate `previous_application.csv` (most recent 3 per customer) → `prev_agg.parquet`
  - [x] `prev_count`, `prev_approved`, `prev_refused`, `prev_avg_credit`, `prev_avg_annuity`, `prev_days_last`
  - [x] Derived: `prev_approval_rate = prev_approved / prev_count`
- [x] Aggregate `installments_payments.csv` (last 12 months only) → `inst_agg.parquet`
  - [x] Compute `payment_diff = AMT_PAYMENT - AMT_INSTALMENT`, `days_late`, `is_late`
  - [x] `inst_count`, `inst_late_count`, `inst_avg_diff`, `inst_max_late`
  - [x] Derived: `inst_late_rate = inst_late_count / inst_count`
- [x] Aggregate `credit_card_balance.csv` (last 6 months only) → `cc_agg.parquet`
  - [x] `cc_avg_balance`, `cc_avg_limit`, `cc_avg_payment`, `cc_max_dpd`, `cc_max_dpd_def`
  - [x] Derived: `cc_utilisation = cc_avg_balance / cc_avg_limit`
- [x] Left-join all 4 aggregates onto application_slim → `master.parquet`
- [x] Verify: 307511 rows, 61 columns, 8.07% default rate ✓

---

### Notebook 01 — EDA

- [x] Load `master.parquet`
- [x] Default rate by key segments: `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`
  - [x] Horizontal bar charts → `Outputs/dr_<col>.png` (4 files confirmed)
- [x] Distributions split by target for: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `AMT_CREDIT`, `DAYS_BIRTH`
  - [x] Overlapping histograms → `Outputs/dist_<col>.png` (5 files confirmed)
- [x] Missing value bar chart → `Outputs/missing_rates.png` ✓
- [ ] **PENDING** — Document observations in NB01 Section 4 markdown cell (left blank intentionally — fill in after reviewing charts)

---

### Notebook 02 — Feature Engineering & Selection

- [x] Load `master.parquet`
- [x] Create affordability ratios: `credit_annuity_ratio`, `income_annuity_ratio`, `income_credit_ratio`, `credit_goods_ratio`
- [x] Create age & employment features: `age_years`, `is_unemployed`, `employed_years`, `employed_age_ratio`
- [x] Create external score composites: `ext_source_mean`, `ext_source_min`
- [x] Create bureau signal: `bureau_overdue_rate`
- [x] Create address & inquiry features: `address_mismatch`, `recent_inquiries`
- [x] ~~`scorecardpy.iv()`~~ → **CHANGED** replaced with custom `compute_iv()` using pandas groupby (same formula, runs in seconds vs 30+ min)
- [x] IV filter applied — auto-selects features with IV > 0.05, excludes superseded raw columns
- [x] **CHANGED** — 25 features selected (plan said 15–20); all pass IV > 0.05 threshold
- [x] **ADDED** — saves `selected_features.json` so feature list flows automatically into NB03/04/05
- [x] Save `master_features.parquet` ✓

---

### Notebook 03 — WoE Scorecard

- [x] Load `master_features.parquet` + `selected_features.json`, sort by `SK_ID_CURR`
- [x] 80/20 out-of-time split
- [x] Print train/OOT sizes and default rates
- [x] Fit `BinningProcess`: `min_bin_size=0.05`, `max_n_bins=8`; auto-detects categorical columns
- [x] Print binning table per variable
- [x] Transform train and OOT to WoE
- [x] Fit `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)`
- [x] Print coefficient table
- [x] Convert to scorecard points (PDO=20, base=600, odds=50:1)
- [x] Score distribution plot → `Outputs/score_distribution.png` ✓
- [x] Calibration curve → `Outputs/calibration.png` ✓
- [x] Save `Outputs/lr_model.pkl` and `Outputs/binning.pkl` ✓

---

### Notebook 04 — LightGBM Benchmark

- [x] Load `master_features.parquet` + `selected_features.json`, same 80/20 split
- [x] Cast object columns to `category` dtype before creating `lgb.Dataset` (fix for dtype error)
- [x] Train: `objective=binary`, `metric=auc`, `lr=0.05`, `num_leaves=31`, `min_child_samples=200`
- [x] Early stopping (patience=30), log every 50 rounds
- [x] Save `Outputs/lgbm_model.txt` ✓
- [x] ~~SHAP values + summary/dependence plots~~ → **CHANGED** replaced with LightGBM built-in `feature_importance(importance_type='gain')` due to shap/numba/numpy version conflict
- [x] Feature importance bar chart → `Outputs/lgbm_feature_importance.png` ✓

---

### Notebook 05 — Validation

**Results: Scorecard AUC 0.758 / Gini 0.516 | LightGBM AUC 0.773 / Gini 0.545 — both exceed "Good" threshold (> 0.50)**

- [x] Load data, models, split; generate all predictions in setup cell
- [x] **Discrimination** — AUC, Gini, KS printed for both models
- [x] Dual ROC curve → `Outputs/roc_curve.png` ✓
- [x] **Calibration** — Hosmer-Lemeshow decile table printed
- [x] **PSI** — train vs OOT comparison, Stable/Monitor/Unstable label printed
- [x] **Rank-ordering** — decile table → `Outputs/decile_table.csv` ✓
- [x] Final comparison table → `Outputs/model_comparison.csv` ✓
- [ ] **PENDING** — Write reflection markdown cell in NB05 Section 6 (fill in after reviewing results)

---

## Changes from Original Plan

| What | Original | Actual |
|---|---|---|
| IV calculation | `scorecardpy.iv()` | Custom `compute_iv()` — same math, seconds not minutes |
| Feature count | 15–20 | 25 (all IV > 0.05, auto-selected) |
| Feature list passing | Manual copy-paste | `selected_features.json` auto-loaded by NB03/04/05 |
| NB04 explainability | SHAP summary + dependence plots | LightGBM built-in gain importance (shap incompatible with numpy version) |
| Output files | `shap_summary.png`, `shap_dependence.png` | `lgbm_feature_importance.png` |
| `scorecardpy` | In requirements | Removed — replaced by custom function |

---

## Processed File Inventory

| File | Created in | Used in |
|---|---|---|
| `application_slim.parquet` | 00 | 00 (merge) |
| `bureau_agg.parquet` | 00 | 00 (merge) |
| `prev_agg.parquet` | 00 | 00 (merge) |
| `inst_agg.parquet` | 00 | 00 (merge) |
| `cc_agg.parquet` | 00 | 00 (merge) |
| `master.parquet` | 00 | 01, 02 |
| `master_features.parquet` | 02 | 03, 04, 05 |
| `selected_features.json` | 02 | 03, 04, 05 |

## Output File Inventory

| File | Created in | Status |
|---|---|---|
| `dr_<col>.png` (×4) | 01 | ✓ |
| `dist_<col>.png` (×5) | 01 | ✓ |
| `missing_rates.png` | 01 | ✓ |
| `score_distribution.png` | 03 | ✓ |
| `calibration.png` | 03 | ✓ |
| `lr_model.pkl` | 03 | ✓ |
| `binning.pkl` | 03 | ✓ |
| `lgbm_model.txt` | 04 | ✓ |
| `lgbm_feature_importance.png` | 04 | ✓ |
| `roc_curve.png` | 05 | ✓ |
| `decile_table.csv` | 05 | ✓ |
| `model_comparison.csv` | 05 | ✓ |
