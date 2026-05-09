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
| `bureau_balance.csv` | 376 MB | 27M | **Delete** — too large, 3 cols, low marginal value |
| `POS_CASH_balance.csv` | 393 MB | 10M | **Delete** — similar signal to instalments |
| `application_test.csv` | — | — | **Delete** — no TARGET column |
| `sample_submission.csv` | — | — | **Delete** — Kaggle artefact |

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

- [ ] Read `application_train.csv`, verify shape (307511, 122) and default rate (~8%)
- [ ] Drop document flags (`FLAG_DOCUMENT_2–21`), contact flags, housing detail columns
- [ ] Keep 33 essential columns → save `application_slim.parquet`
- [ ] Aggregate `bureau.csv` → `bureau_agg.parquet`
  - [ ] `bureau_loan_count`, `bureau_active_count`, `bureau_total_debt`, `bureau_total_credit`
  - [ ] `bureau_total_overdue`, `bureau_max_overdue`, `bureau_avg_days`, `bureau_cnt_prolong`
  - [ ] Derived: `bureau_debt_ratio = bureau_total_debt / bureau_total_credit`
- [ ] Aggregate `previous_application.csv` (most recent 3 per customer) → `prev_agg.parquet`
  - [ ] `prev_count`, `prev_approved`, `prev_refused`, `prev_avg_credit`, `prev_avg_annuity`, `prev_days_last`
  - [ ] Derived: `prev_approval_rate = prev_approved / prev_count`
- [ ] Aggregate `installments_payments.csv` (last 12 months only) → `inst_agg.parquet`
  - [ ] Compute `payment_diff = AMT_PAYMENT - AMT_INSTALMENT`, `days_late`, `is_late`
  - [ ] `inst_count`, `inst_late_count`, `inst_avg_diff`, `inst_max_late`
  - [ ] Derived: `inst_late_rate = inst_late_count / inst_count`
- [ ] Aggregate `credit_card_balance.csv` (last 6 months only) → `cc_agg.parquet`
  - [ ] `cc_avg_balance`, `cc_avg_limit`, `cc_avg_payment`, `cc_max_dpd`, `cc_max_dpd_def`
  - [ ] Derived: `cc_utilisation = cc_avg_balance / cc_avg_limit`
- [ ] Left-join all 4 aggregates onto application_slim → `master.parquet`
- [ ] Verify: ~307k rows, ~55 columns, ~8% default rate

---

### Notebook 01 — EDA

- [ ] Load `master.parquet`
- [ ] Default rate by key segments: `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`
  - [ ] Horizontal bar charts, save to `Outputs/dr_<col>.png`
- [ ] Distributions split by target for: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `AMT_CREDIT`, `DAYS_BIRTH`
  - [ ] Overlapping histograms, save to `Outputs/dist_<col>.png`
- [ ] Missing value bar chart → save `Outputs/missing_rates.png`
- [ ] Document observations in a markdown cell:
  - [ ] `EXT_SOURCE_2` and `EXT_SOURCE_3` clearly separate defaulters
  - [ ] `DAYS_EMPLOYED = 365243` is a sentinel for unemployed
  - [ ] `EXT_SOURCE_1` has ~56% missing — missingness itself is a risk signal
  - [ ] Bureau-joined columns have many NaNs (no bureau history = higher risk)
  - [ ] Class imbalance ~8:92 — do not oversample; WoE handles it implicitly

---

### Notebook 02 — Feature Engineering & Selection

- [ ] Load `master.parquet`
- [ ] Create affordability ratios:
  - [ ] `credit_annuity_ratio = AMT_CREDIT / AMT_ANNUITY`
  - [ ] `income_annuity_ratio = AMT_INCOME_TOTAL / AMT_ANNUITY`
  - [ ] `income_credit_ratio = AMT_INCOME_TOTAL / AMT_CREDIT`
  - [ ] `credit_goods_ratio = AMT_CREDIT / AMT_GOODS_PRICE`
- [ ] Create age & employment features:
  - [ ] `age_years = DAYS_BIRTH / -365`
  - [ ] `is_unemployed = (DAYS_EMPLOYED == 365243).astype(int)`
  - [ ] `employed_years = DAYS_EMPLOYED.where(DAYS_EMPLOYED != 365243) / -365`
  - [ ] `employed_age_ratio = employed_years / age_years`
- [ ] Create external score composites:
  - [ ] `ext_source_mean` (mean of EXT_SOURCE_1/2/3)
  - [ ] `ext_source_min` (min of EXT_SOURCE_1/2/3)
- [ ] Create bureau signal: `bureau_overdue_rate = bureau_total_overdue / bureau_total_credit`
- [ ] Create address & inquiry features:
  - [ ] `address_mismatch = REG_CITY_NOT_LIVE_CITY + REG_CITY_NOT_WORK_CITY`
  - [ ] `recent_inquiries = AMT_REQ_CREDIT_BUREAU_MON + AMT_REQ_CREDIT_BUREAU_QRT`
- [ ] Run `scorecardpy.iv()` on all features, print ranked IV table
- [ ] Apply IV filter: keep features with IV > 0.05, covering distinct dimensions
- [ ] Define final `selected_features` list (target: 15–20 features)
- [ ] Save `master_features.parquet`

---

### Notebook 03 — WoE Scorecard

- [ ] Load `master_features.parquet`, sort by `SK_ID_CURR`
- [ ] 80/20 out-of-time split (first 80% trains, last 20% tests)
- [ ] Print train/OOT sizes and default rates (should both be ~8%)
- [ ] Fit `BinningProcess` on train: `min_bin_size=0.05`, `max_n_bins=8`
- [ ] For each variable, print binning table — verify WoE is monotone and intuitive
- [ ] Transform train and OOT to WoE
- [ ] Fit `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)`
- [ ] Print coefficient table — verify signs align with business logic (higher score = lower risk)
- [ ] Convert probabilities to scorecard points (PDO=20, base score=600, base odds=50:1)
- [ ] Plot score distribution by outcome → save `Outputs/score_distribution.png`
- [ ] Plot calibration curve (OOT) → save `Outputs/calibration.png`
- [ ] Save `Outputs/lr_model.pkl` and `Outputs/binning.pkl`

---

### Notebook 04 — LightGBM Benchmark

- [ ] Load `master_features.parquet`, same 80/20 split as notebook 03
- [ ] Create `lgb.Dataset` for train and OOT
- [ ] Train with params: `objective=binary`, `metric=auc`, `lr=0.05`, `num_leaves=31`, `min_child_samples=200`
- [ ] Use early stopping (patience=30), log every 50 rounds
- [ ] Save `Outputs/lgbm_model.txt`
- [ ] Compute SHAP values on OOT
- [ ] SHAP summary plot → save `Outputs/shap_summary.png`
- [ ] SHAP dependence plot for top feature (likely `EXT_SOURCE_2`) → save `Outputs/shap_dependence.png`

---

### Notebook 05 — Validation

- [ ] Load `master_features.parquet`, same split; load both saved models
- [ ] Generate predictions: `y_pred_lr` (scorecard), `y_pred_lgb` (LightGBM)
- [ ] **Discrimination** — for both models print AUC, Gini, KS
- [ ] Plot dual ROC curve → save `Outputs/roc_curve.png`
- [ ] **Calibration** — Hosmer-Lemeshow decile table (scorecard only)
  - [ ] Group by predicted PD decile; compare observed vs expected default rate
- [ ] **PSI** — compare train vs OOT score distributions
  - [ ] Print: Stable (< 0.10) / Monitor (0.10–0.25) / Unstable (> 0.25)
- [ ] **Rank-ordering** — decile table with cumulative % defaults captured
  - [ ] Save `Outputs/decile_table.csv`
- [ ] Final comparison table (AUC, Gini, interpretability, regulatory fit)
  - [ ] Save `Outputs/model_comparison.csv`
- [ ] Write reflection markdown cell:
  - [ ] What is the Gini gap between the two models?
  - [ ] Why does a bank accept lower Gini for the scorecard?
  - [ ] What does interpretability buy in a regulated environment?

---

## Processed File Inventory

| File | Created in | Used in |
|---|---|---|
| `application_slim.parquet` | 00 | 00 (merge step) |
| `bureau_agg.parquet` | 00 | 00 (merge step) |
| `prev_agg.parquet` | 00 | 00 (merge step) |
| `inst_agg.parquet` | 00 | 00 (merge step) |
| `cc_agg.parquet` | 00 | 00 (merge step) |
| `master.parquet` | 00 | 01, 02 |
| `master_features.parquet` | 02 | 03, 04, 05 |

## Output File Inventory

| File | Created in |
|---|---|
| `dr_<col>.png` (×4) | 01 |
| `dist_<col>.png` (×5) | 01 |
| `missing_rates.png` | 01 |
| `score_distribution.png` | 03 |
| `calibration.png` | 03 |
| `lr_model.pkl` | 03 |
| `binning.pkl` | 03 |
| `lgbm_model.txt` | 04 |
| `shap_summary.png` | 04 |
| `shap_dependence.png` | 04 |
| `roc_curve.png` | 05 |
| `decile_table.csv` | 05 |
| `model_comparison.csv` | 05 |
