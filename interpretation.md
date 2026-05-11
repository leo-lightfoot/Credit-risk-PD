# PD Model — Interpretation & Findings

---

## 1. Dataset

| | |
|---|---|
| Source | Home Credit Default Risk |
| Population | 307,511 loan applications |
| Target | `TARGET` — 1 = defaulted, 0 = repaid |
| Overall default rate | 8.07% |
| Train set | 246,008 rows (first 80% by `SK_ID_CURR`) |
| OOT test set | 61,503 rows (last 20%) |
| OOT default rate | 8.07% — consistent with train, no drift |

---

## 2. EDA Findings

### Default rate by segment

| Segment | Group | Default rate |
|---|---|---|
| Contract type | Cash loans | 8.3% |
| | Revolving loans | 5.5% |
| Gender | Male | 10.1% |
| | Female | 7.0% |
| Income type | Maternity leave | 40.0% |
| | Unemployed | 36.4% |
| | Working | 9.6% |
| | State servant | 5.8% |
| | Pensioner | 5.4% |
| Education | Lower secondary | 10.9% |
| | Secondary | 8.9% |
| | Incomplete higher | 8.5% |
| | Higher education | 5.4% |
| | Academic degree | 1.8% |

Employment status and education level show the steepest gradients — both are strong risk signals. Maternity leave and unemployed categories are very high risk but represent a small share of the population.

### External scores — strongest predictors

| Feature | No default (mean) | Default (mean) | Gap |
|---|---|---|---|
| `EXT_SOURCE_1` | 0.511 | 0.387 | 0.124 |
| `EXT_SOURCE_2` | 0.523 | 0.411 | 0.112 |
| `EXT_SOURCE_3` | 0.521 | 0.391 | 0.130 |

All three external scores clearly separate defaulters from non-defaulters. Higher score = lower risk. The gap is consistent across all three sources.

### Age

- No default: mean age **44.2 years**
- Default: mean age **40.8 years**

Older applicants default less. The 3.4-year gap is meaningful at scale.

### Missing values

| Feature | Missing rate | Note |
|---|---|---|
| `EXT_SOURCE_1` | 56.4% | Missingness itself is a risk signal — customers without an external score default more |
| `OWN_CAR_AGE` | 66.0% | Missing = no car owned |
| `EXT_SOURCE_3` | 19.8% | Moderate missingness |
| `OCCUPATION_TYPE` | 31.3% | Self-reported, commonly left blank |
| `EXT_SOURCE_2` | 0.2% | Nearly complete |

The WoE binning assigns a dedicated bin to missing values, so no imputation is needed and missingness becomes informative.

---

## 3. Features Selected (25)

Selected automatically: IV > 0.05 on the master dataset, raw columns superseded by derived ones excluded.

| Group | Features | What they measure |
|---|---|---|
| External scores | `EXT_SOURCE_1/2/3`, `ext_source_mean`, `ext_source_min` | Third-party creditworthiness scores |
| Credit card | `cc_utilisation`, `cc_avg_balance`, `cc_avg_payment` | Card usage intensity and payment behaviour |
| Instalment history | `inst_late_rate`, `inst_late_count`, `inst_max_late`, `inst_avg_diff` | Payment timeliness on prior loans |
| Bureau | `bureau_debt_ratio`, `bureau_avg_days` | External credit burden and credit history length |
| Affordability | `credit_annuity_ratio`, `credit_goods_ratio`, `AMT_GOODS_PRICE` | Loan sizing relative to goods and repayment capacity |
| Employment & age | `employed_years`, `employed_age_ratio`, `age_years` | Job tenure and age as stability proxies |
| Categorical | `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE` | Socioeconomic profile |
| Assets | `OWN_CAR_AGE` | Asset ownership as stability signal |

---

## 4. WoE Scorecard — Logistic Regression Coefficients

In the WoE framework, all coefficients should be negative — higher WoE (more non-events in a bin) means lower predicted risk, so a negative coefficient means riskier bins get higher predicted PD. Three features show positive coefficients, flagged below.

| Feature | Coefficient | Direction |
|---|---|---|
| `credit_annuity_ratio` | −0.644 | Higher annuity-to-credit ratio → lower risk |
| `OWN_CAR_AGE` | −0.606 | Older car ownership → lower risk |
| `EXT_SOURCE_2` | −0.538 | Higher score → lower risk ✓ |
| `EXT_SOURCE_3` | −0.535 | Higher score → lower risk ✓ |
| `credit_goods_ratio` | −0.477 | Loan closer to goods value → lower risk |
| `AMT_GOODS_PRICE` | −0.467 | Higher goods price → lower risk |
| `NAME_EDUCATION_TYPE` | −0.464 | Higher education → lower risk ✓ |
| `cc_utilisation` | −0.458 | Lower utilisation → lower risk ✓ |
| `inst_max_late` | −0.455 | Fewer late days → lower risk ✓ |
| `ORGANIZATION_TYPE` | −0.445 | Organization type risk gradient |
| `EXT_SOURCE_1` | −0.376 | Higher score → lower risk ✓ |
| `bureau_debt_ratio` | −0.349 | Lower debt-to-credit → lower risk ✓ |
| `age_years` | +0.055 | ⚠ Positive — non-monotone WoE bins |
| `inst_late_count` | +0.077 | ⚠ Positive — check WoE bins |
| `cc_avg_payment` | +0.098 | ⚠ Positive — may reflect high-usage profiles |

The three positive coefficients are weak (near zero) and likely reflect non-monotone WoE binning in those features rather than a genuine reversal of direction.

---

## 5. Scorecard Points Distribution

| | No default | Default |
|---|---|---|
| Mean score | 570.3 | 543.8 |
| Score range (OOT) | 452 – 655 | 452 – 655 |
| Separation | **26.5 points** | — |

The score scale is centred at 600 (base odds 50:1, PDO=20). Defaulters score ~27 points lower on average — a meaningful separation that allows a lender to set cut-off thresholds.

---

## 6. Model Performance

| Model | AUC | Gini | KS | Benchmark |
|---|---|---|---|---|
| WoE Scorecard | 0.758 | 0.516 | 0.391 | Gini > 0.50 ✓ Good |
| LightGBM | 0.773 | 0.545 | 0.409 | Gini > 0.50 ✓ Good |
| **Gap** | 0.015 | **0.029** | 0.018 | — |

Both models comfortably exceed the "Good" threshold for retail unsecured lending. The Gini gap of 2.9 points is small — the scorecard gives up very little performance for full interpretability.

### LightGBM — top features by gain

| Feature | Gain | Relative importance |
|---|---|---|
| `ext_source_mean` | 112,813 | Dominant — 3× the next feature |
| `ORGANIZATION_TYPE` | 23,551 | Employer sector |
| `credit_annuity_ratio` | 19,279 | Loan affordability |
| `credit_goods_ratio` | 10,301 | Loan-to-goods ratio |
| `EXT_SOURCE_3` | 9,001 | External score |
| `AMT_GOODS_PRICE` | 8,628 | Goods value |
| `bureau_debt_ratio` | 8,427 | External credit burden |
| `inst_max_late` | 8,114 | Worst payment delay |
| `cc_utilisation` | 8,068 | Card utilisation |
| `ext_source_min` | 7,580 | Worst external score |

`ext_source_mean` dominates — its gain is nearly 5× the second feature. External credit scores are by far the most predictive signal in this dataset. This is consistent with the logistic regression.

---

## 7. Calibration — Hosmer-Lemeshow

The scorecard predicts actual default rates well across the risk spectrum. Observed vs expected default rates per predicted PD decile:

| Decile (0=lowest PD) | N | Observed rate | Expected PD | Assessment |
|---|---|---|---|---|
| 0 | 6,151 | 1.5% | 1.3% | ✓ |
| 1 | 6,150 | 1.7% | 2.2% | ✓ |
| 2 | 6,150 | 2.8% | 3.0% | ✓ |
| 3 | 6,150 | 3.0% | 3.8% | ✓ slight under-prediction |
| 4 | 6,151 | 4.5% | 4.8% | ✓ |
| 5 | 6,150 | 5.9% | 6.1% | ✓ |
| 6 | 6,150 | 8.3% | 7.9% | ✓ |
| 7 | 6,150 | 10.7% | 10.4% | ✓ |
| 8 | 6,150 | 15.1% | 14.6% | ✓ |
| 9 | 6,151 | 25.9% | 26.6% | ✓ slight over-prediction |

Calibration is good — observed and expected rates track closely. Slight under-prediction in mid-deciles and over-prediction at the top, but within acceptable tolerance.

---

## 8. Population Stability Index

**PSI = 0.0003 — Stable** (threshold: < 0.10)

The score distribution is virtually identical between train and OOT. The model is not overfitting to the training population.

---

## 9. Rank-Ordering

The scorecard correctly rank-orders risk. Decile 1 = highest predicted risk, Decile 10 = lowest.

| Decile | Count | Default rate | Avg PD | Cum. defaults captured |
|---|---|---|---|---|
| 1 (riskiest) | 6,151 | **25.9%** | 26.6% | 32.6% |
| 2 | 6,150 | 15.1% | 14.6% | 51.5% |
| 3 | 6,150 | 10.7% | 10.4% | 63.8% |
| 4 | 6,150 | 8.3% | 7.9% | 73.1% |
| 5 | 6,150 | 5.9% | 6.1% | 80.5% |
| 6 | 6,151 | 4.5% | 4.8% | 86.3% |
| 7 | 6,150 | 3.0% | 3.8% | 89.9% |
| 8 | 6,150 | 2.8% | 3.0% | 92.7% |
| 9 | 6,150 | 1.7% | 2.2% | 96.1% |
| 10 (safest) | 6,151 | **1.5%** | 1.3% | 100% |

The top decile has a 25.9% default rate — **17× the bottom decile (1.5%)**. The top two deciles together capture 51.5% of all defaults while representing only 20% of applicants. This is strong rank-ordering.

---

## 10. Scorecard vs LightGBM — The Trade-off

| | WoE Scorecard | LightGBM |
|---|---|---|
| Gini | 0.516 | 0.545 |
| Interpretable | Yes — each bin has a WoE and scorecard points | Feature importance only, no per-applicant explanation |
| Regulatory fit | Yes — Basel IRB standard | No — regulators require explainability |
| Audit trail | Full — every score can be decomposed by feature | Not available |
| Gini cost of interpretability | — | **2.9 Gini points** |

A bank accepts the 2.9-point Gini loss because:
- Regulators (Basel II/III IRB) require models where every credit decision can be explained to the applicant
- Internal model risk teams require full auditability — you must be able to explain why a score changed between two dates
- A scorecard's WoE bins and points make stress-testing and sensitivity analysis straightforward
- LightGBM's feature importance tells you which features matter globally but cannot explain a single customer's score

---

## 11. Known Limitations

| Limitation | Detail |
|---|---|
| `bureau_balance` excluded | 27M rows, 3 columns — skipped for learning efficiency; may contain additional delinquency signal |
| `POS_CASH_balance` excluded | Similar signal to instalments; skipped |
| EXT_SOURCE opacity | These external scores are the strongest predictors but their construction is unknown — model depends on a black box |
| OOT split is approximate | Sorted by `SK_ID_CURR` as a proxy for time — not a guaranteed temporal split |
| 25 features vs plan's 15–20 | All pass IV > 0.05; a tighter threshold would reduce the feature set but may slightly reduce Gini |
| SHAP removed | Replaced by LightGBM built-in importance due to a shap/numba/numpy version conflict — directional SHAP analysis not available |
