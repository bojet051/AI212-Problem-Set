# AI212 Hatchery Linear Regression Report

## Objective

This report compares non-Bayesian and Bayesian linear regression for hatchery performance data. The required tasks were:

- compare OLS and Bayesian regression
- study weak versus informative priors
- report point estimates and confidence/credible intervals
- evaluate models using 5-fold cross-validation
- evaluate OLS models using 0.632 bootstrap
- study model complexity using AIC and WAIC

The executed notebook is:

`Bayesian_non_Bayesian/AI212_Hatch_Bayesian_OLS_Regression_Implementation.ipynb`

## Data And Cleaning

Data source:

- File: `Hatch_Model.xlsx`
- Sheet: `DB`
- Loaded columns: `B:P, AE`
- Raw selected shape: `128,066 x 16`
- Cleaned modeling shape: `71,773 x 20`

Cleaning pipeline:

- normalized headers
- removed summary rows where `HY` contains `GRAND TOTAL`
- converted `HATCH_DATE`, `DATE_SET`, and `PRODN_DATE` to datetime
- converted `AGE`, `GRP`, `S.P.`, `EGGSET`, and `CLEAN_CHICKS` to numeric
- kept rows with non-null core fields and `EGGSET > 0`
- removed duplicates
- derived:
  - `CLEAN_CHICKS_RATIO = CLEAN_CHICKS / EGGSET`
  - `CLEAN_CHICKS_PCT = CLEAN_CHICKS_RATIO * 100`
- kept target values in `[0, 1]`

## Response And Predictors

Response:

```text
Y = CLEAN_CHICKS_RATIO
```

Excluded to avoid leakage:

```text
EGGSET, CLEAN_CHICKS, CLEAN_CHICKS_RATIO, CLEAN_CHICKS_PCT
```

Raw candidate predictors:

```text
HY, LOCATION, WEEK_NO, HATCH_DATE, DATE_SET, FARM_SOURCE,
LINE, B, H, STRAIN, PRODN_DATE, AGE, GRP, S.P.
```

Categorical raw candidates not encoded in this run:

```text
HY, LOCATION, FARM_SOURCE, LINE, B, H, STRAIN
```

Date/week raw candidates not engineered in this run:

```text
WEEK_NO, HATCH_DATE, DATE_SET, PRODN_DATE
```

Selected raw numeric predictors:

```text
AGE, GRP, S.P.
```

This satisfies the assignment requirement to use 1 to 5 predictors. Since all feature engineering was removed, only the raw numeric predictors were eligible for direct use in the linear model.

Expected relationships:

- `AGE`: expected negative association with hatch ratio
- `GRP`: expected negative association and likely overlap with `AGE`
- `S.P.`: expected negative association because longer storage should reduce hatch performance

## Exploratory Data Analysis

Target summary:

| Variable | Mean | SD | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| `CLEAN_CHICKS_RATIO` | 0.6074 | 0.1773 | 0.0000 | 0.6419 | 1.0000 |

Predictor summaries:

| Variable | Mean | SD | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| `AGE` | 44.8089 | 11.2177 | 23.0000 | 44.0000 | 73.0000 |
| `GRP` | 4.7380 | 1.9043 | 2.0000 | 4.0000 | 9.0000 |
| `S.P.` | 10.5943 | 4.0577 | 1.0000 | 11.0000 | 26.0000 |

Correlations with the response:

| Predictor | Correlation with `CLEAN_CHICKS_RATIO` |
|---|---:|
| `AGE` | -0.6434 |
| `GRP` | -0.6398 |
| `S.P.` | -0.0987 |

Interpretation:

- `AGE` and `GRP` have the strongest negative relationships with the response.
- `AGE` and `GRP` are also strongly correlated with each other, so coefficient interpretation should acknowledge overlap.
- `S.P.` has a weaker but still negative association.

## Tools And Functions Used

| Library/function | Purpose | Inputs | Outputs |
|---|---|---|---|
| `pandas.read_excel` | Load source data | file path, sheet, column range | raw dataframe |
| `pandas` cleaning methods | Filter, convert types, derive target | raw dataframe | cleaned dataframe |
| `statsmodels.api.OLS` | OLS regression | `y`, design matrix `X` | coefficients, CIs, AIC |
| `sklearn.model_selection.KFold` | 5-fold CV | `n_splits=5`, shuffle, random state | train/test folds |
| `sklearn.pipeline.Pipeline` | Standardized CV workflow | `StandardScaler`, `LinearRegression` | fold predictions |
| `sklearn.metrics.mean_squared_error` | RMSE calculation | observed, predicted | RMSE |
| `scipy.stats.invgamma` | Bayesian variance draws | posterior `a`, `b` | `sigma^2` samples |
| `scipy.special.logsumexp` | WAIC calculation | log-likelihood draws | stable lppd values |

Cross-validation scheme:

- 5-fold cross-validation
- shuffled splits
- fixed `RANDOM_STATE = 212`
- reported mean RMSE and fold SD

Bootstrap scheme:

- 0.632 bootstrap for OLS models
- bootstrap sample with replacement
- out-of-bag validation
- error formula:

```text
0.632 error = 0.368 * resubstitution RMSE + 0.632 * OOB RMSE
```

## OLS Regression

Model:

```text
CLEAN_CHICKS_RATIO ~ AGE + GRP + S.P.
```

Training performance:

| Metric | Value |
|---|---:|
| RMSE | 0.131681 |
| RMSE in percentage points | 13.168 |
| R-squared | 0.4485 |
| Adjusted R-squared | 0.4485 |
| AIC | -87330.53 |

OLS coefficient estimates:

| Term | Estimate | 95% CI Low | 95% CI High |
|---|---:|---:|---:|
| Intercept | 1.096870 | 1.088685 | 1.105055 |
| `AGE` | -0.006513 | -0.006925 | -0.006101 |
| `GRP` | -0.024107 | -0.026535 | -0.021679 |
| `S.P.` | -0.007873 | -0.008112 | -0.007634 |

Interpretation:

- all three coefficients are negative
- all three intervals exclude zero
- the linear model explains about 44.9% of variation in `CLEAN_CHICKS_RATIO`

## Bayesian Regression With Weak Prior

Model:

```text
y | X, beta, sigma^2 ~ Normal(X beta, sigma^2 I)
beta | sigma^2 ~ Normal(0, sigma^2 * 1000^2 I)
sigma^2 ~ InverseGamma(0.001, 0.001)
```

Training performance:

| Metric | Value |
|---|---:|
| RMSE | 0.131681 |
| Posterior predictive 95% coverage | 0.9520 |
| Mean posterior predictive 95% width | 0.5162 |

Weak-prior posterior summaries on the original scale:

| Term | Posterior Mean | 95% Credible Low | 95% Credible High |
|---|---:|---:|---:|
| Intercept | 1.096833 | 1.089023 | 1.104845 |
| `AGE` | -0.006510 | -0.006920 | -0.006106 |
| `GRP` | -0.024129 | -0.026455 | -0.021737 |
| `S.P.` | -0.007873 | -0.008112 | -0.007635 |

Comparison with OLS:

- posterior means are nearly identical to OLS estimates
- credible intervals are very close to OLS confidence intervals
- this is expected under a very weak prior with a large sample

## Bayesian Regression With Informative Prior

Informative prior construction:

- prior mean centered at standardized OLS coefficients
- prior SD based on OLS standard errors
- same variance prior:

```text
sigma^2 ~ InverseGamma(0.001, 0.001)
```

Training performance:

| Metric | Value |
|---|---:|
| RMSE | 0.131681 |
| Posterior predictive 95% coverage | 0.9520 |
| Mean posterior predictive 95% width | 0.5162 |

Informative-prior posterior summaries on the original scale:

| Term | Posterior Mean | 95% Credible Low | 95% Credible High |
|---|---:|---:|---:|
| Intercept | 1.096911 | 1.090833 | 1.103208 |
| `AGE` | -0.006514 | -0.006773 | -0.006253 |
| `GRP` | -0.024099 | -0.025588 | -0.022562 |
| `S.P.` | -0.007876 | -0.008109 | -0.007644 |

Prior effect:

- posterior means remain very close to the weak-prior results
- informative-prior intervals narrow slightly
- the prior does not materially pull the coefficients away from the data because it is centered at the OLS solution and the sample is large

## OLS Cross-Validation And Bootstrap

OLS complexity sequence:

| k | Predictors | 5-fold CV RMSE | 0.632 Bootstrap RMSE | AIC |
|---:|---|---:|---:|---:|
| 1 | `AGE` | 0.135736 | 0.135655 | -82978.50 |
| 2 | `AGE + GRP` | 0.135446 | 0.135431 | -83287.59 |
| 3 | `AGE + GRP + S.P.` | 0.131687 | 0.131677 | -87330.53 |

Incremental OLS CV improvement:

| Step | Improvement |
|---|---:|
| Add `GRP` after `AGE` | 0.000290 |
| Add `S.P.` after `AGE + GRP` | 0.003759 |

Interpretation:

- adding `GRP` after `AGE` improves generalization only slightly
- adding `S.P.` provides the largest additional improvement
- within the no-feature-engineering constraint, the best OLS model is the full three-predictor model

## Bayesian Model Complexity And WAIC

Bayesian complexity sequence:

| k | Predictors | Bayesian 5-fold CV RMSE | WAIC |
|---:|---|---:|---:|
| 1 | `AGE` | 0.135736 | -82975.87 |
| 2 | `AGE + GRP` | 0.135446 | -83284.99 |
| 3 | `AGE + GRP + S.P.` | 0.131687 | -87327.59 |

Interpretation:

- Bayesian CV RMSE tracks OLS CV RMSE closely
- WAIC selects the same final model as OLS AIC and CV
- the three-predictor model is the best model under the current constraint set

## Final Model Selection

Best model by criterion:

| Criterion | Best k | Selected Model | Score |
|---|---:|---|---:|
| OLS 5-fold CV RMSE | 3 | `AGE + GRP + S.P.` | 0.131687 |
| OLS 0.632 Bootstrap RMSE | 3 | `AGE + GRP + S.P.` | 0.131677 |
| OLS AIC | 3 | `AGE + GRP + S.P.` | -87330.53 |
| Bayesian 5-fold CV RMSE | 3 | `AGE + GRP + S.P.` | 0.131687 |
| Bayesian WAIC | 3 | `AGE + GRP + S.P.` | -87327.59 |

Recommended model:

```text
CLEAN_CHICKS_RATIO ~ AGE + GRP + S.P.
```

## Deliverables

The main generated artifacts are in:

`Bayesian_non_Bayesian/hatch_outputs/`

Key files:

- `summary_report.md`
- `results.json`
- `eda_summary.csv`
- `correlations.csv`
- `ols_coefficients.csv`
- `ols_bayesian_comparison_original_scale.csv`
- `posterior_predictive_check_summary.csv`
- `model_complexity_ols.csv`
- `model_complexity_bayesian.csv`
- `model_selection_summary.csv`
- `ols_complexity_rmse.png`
- `ols_aic_vs_complexity.png`
- `bayesian_complexity_rmse.png`
- `bayesian_complexity_waic.png`

## Conclusion

This execution is internally consistent with the user-imposed constraint of no feature engineering. Under that constraint, the only usable raw numeric predictors are `AGE`, `GRP`, and `S.P.`, and the best model by every evaluation criterion is the full three-predictor model.

The Bayesian and OLS fits produce nearly identical predictions and very similar coefficient estimates. The informative prior mainly tightens uncertainty slightly rather than changing point estimates. The model is a reasonable interpretable baseline, but it is not the strongest possible modeling strategy because categorical variables and date/week information were intentionally left unused. A stronger next iteration would reintroduce one-hot encoding for categorical predictors and carefully engineered calendar features.
