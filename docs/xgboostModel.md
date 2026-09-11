# xgboostModel.py

XGBoost classifier on the day-30 features from `pipeline.py`. Trees are built one after another, each one fitting the errors the previous trees left behind.

## Running it

```
cd src
python xgboostModel.py     # this model only
python main.py             # both models on the same split
```

## Set up

`build()` returns an `XGBClassifier` with `tree_method="hist"` (bins the features first, much faster on 32k rows), `eval_metric="logloss"`, and the seed and `n_jobs` from `ModelConfig`.

### Early stopping

Instead of tuning the number of trees, `run()` holds out 15% of the train split (`VAL_SIZE`) as a validation slice and passes it as `eval_set`. `n_estimators` is set to 1000 as a ceiling; boosting stops once validation logloss has not improved for 20 rounds (`EARLY_STOPPING_ROUNDS`) and predictions use the best round.

The validation slice is fixed across every grid combo and every CV fold, so the stopping decision is always made on the same rows. The test set is never used for it. The tree count early stopping landed on is reported as `n_estimators` in `best_params`.

Grid searched with 3-fold CV, scored on roc_auc (18 combos, 54 fits):

| Parameter | Values | What it controls |
|---|---|---|
| `learning_rate` | 0.01, 0.1, 0.3 | Step size per tree. Lower needs more trees but generalizes better |
| `max_depth` | 3, 6, 10 | Max splits per tree |
| `reg_lambda` | 1.0, 5.0 | L2 penalty on leaf weights, higher = less overfit |

The grid ranges are the ones listed in the research doc. After the search, the best estimator is cross validated with 5 folds on the fit portion of the train split and then scored once on the 20% hold-out.

## Results

Full grid run, cutoff day 30, withdrawn included as at-risk, 6,519 test rows.

| Metric | 5-fold CV (train) | Hold-out (test) |
|---|---|---|
| Accuracy | 77.31% (+/- 0.68) | 78.55% |
| Precision | 82.32% (+/- 0.68) | 82.90% |
| Recall | 72.62% (+/- 0.86) | 74.81% |
| F1 | 77.16% (+/- 0.72) | 78.65% |
| ROC-AUC | 85.93% (+/- 0.61) | 86.83% |

Hold-out AUC of 0.868.

Confusion matrix on the hold-out:

|  | Not At Risk | At Risk |
|---|---|---|
| **Not At Risk** | 2546 | 531 |
| **At Risk** | 867 | 2575 |

### Feature importance

| Rank | Gain | Permutation |
|---|---|---|
| 1 | `unregistered_before_cutoff` 0.304 | `mean_score` 0.038 |
| 2 | `mean_score` 0.143 | `unregistered_before_cutoff` 0.029 |
| 3 | `studied_credits` 0.044 | `avg_days_before_due` 0.012 |
| 4 | `num_of_prev_attempts` 0.039 | `clicks_page` 0.012 |
| 5 | `avg_days_before_due` 0.035 | `studied_credits` 0.010 |

Gain importance is far more concentrated than the randomForest's Gini: `unregistered_before_cutoff` alone accounts for 30% and the top two for 45%. Permutation importance is more spread out and puts `mean_score` first, which agrees with the forest's ranking: the two models depend on the same handful of features (withdrawal, assessment scores, submission timing, `clicks_page`, `studied_credits`) in slightly different orders.

`region_Wales` and `region_Scotland` appear in XGBoost's permutation top 20. They are small (under 0.002) but they should be looked at.

## Next steps