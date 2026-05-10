# PD Model — Interpretation & Findings

---

## 1. Dataset

- **Source:** Home Credit Default Risk
- **Population:** 307,511 loan applications
- **Target:** `TARGET` — 1 = defaulted, 0 = repaid
- **Default rate:** 8.07%
- **Validation approach:** Out-of-time split — first 80% trains (246,008), last 20% tests (61,503)

---

## 2. Features Selected (25)

Selected automatically: IV > 0.05, raw columns superseded by derived ones excluded.

| Group | Features |
|---|---|
| External scores | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `ext_source_mean`, `ext_source_min` |
| Credit card | `cc_utilisation`, `cc_avg_balance`, `cc_avg_payment` |
| Instalment behaviour | `inst_late_rate`, `inst_late_count`, `inst_max_late`, `inst_avg_diff` |
| Bureau | `bureau_debt_ratio`, `bureau_avg_days` |
| Affordability | `credit_annuity_ratio`, `credit_goods_ratio`, `AMT_GOODS_PRICE` |
| Employment & age | `employed_years`, `employed_age_ratio`, `age_years` |
| Categorical | `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE` |
| Assets | `OWN_CAR_AGE` |

---

## 3. EDA Observations

*(fill in after reviewing charts in Outputs/)*

**Default rate by segment:**

**Score distributions (EXT_SOURCE, AMT_CREDIT, DAYS_BIRTH):**

**Missing values:**

---

## 4. Model Performance

| Model | AUC | Gini | Benchmark |
|---|---|---|---|
| WoE Scorecard | 0.758 | 0.516 | > 0.50 ✓ Good |
| LightGBM | 0.773 | 0.545 | > 0.50 ✓ Good |
| **Gap** | 0.015 | 0.029 | — |

Both models exceed the "Good" threshold for retail unsecured lending (Gini > 0.50).

---

## 5. Scorecard Rank-Ordering

Decile 1 = highest predicted risk, Decile 10 = lowest.

| Decile | Count | Default rate | Avg PD | Cumulative defaults captured |
|---|---|---|---|---|
| 1 (riskiest) | 6,151 | 25.9% | 26.6% | 100% |
| 2 | 6,150 | 15.1% | 14.6% | 67.4% |
| 3 | 6,150 | 10.7% | 10.4% | 48.4% |
| 4 | 6,150 | 8.3% | 7.9% | 34.9% |
| 5 | 6,150 | 5.9% | 6.1% | 24.4% |
| 6 | 6,151 | 4.5% | 4.8% | 16.9% |
| 7 | 6,150 | 3.0% | 3.8% | 11.3% |
| 8 | 6,150 | 2.8% | 3.0% | 7.5% |
| 9 | 6,150 | 1.7% | 2.2% | 4.0% |
| 10 (safest) | 6,151 | 1.5% | 1.3% | 1.9% |

---

## 6. Calibration & Stability

**Calibration (Hosmer-Lemeshow):** *(note whether observed ≈ expected default rate per decile)*

**PSI:** *(note value and Stable/Monitor/Unstable label from NB05 output)*

---

## 7. Scorecard vs LightGBM — The Trade-off

**Gini gap:** 0.029 (LightGBM higher)

*(fill in: why does a bank accept lower Gini for the scorecard? what does interpretability buy in a regulated environment?)*

---

## 8. Known Limitations

- `bureau_balance.csv` and `POS_CASH_balance.csv` were excluded — may contain additional signal
- EXT_SOURCE features are external scores whose construction is opaque — treated as black-box inputs
- Out-of-time split uses customer ID order as a proxy for time — not a true temporal split
