# randomForest.py

Random Forest classifier on the day-32 features from `pipeline.py`.

## Running it

```
cd src
python randomForest.py     # this model only
python main.py             # both models on the same split
```

## Set up

`build()` returns a `RandomForestClassifier` with `class_weight="balanced"`, which weights each class inversely to how often it appears so the forest is not biased toward the larger class. Seed and `n_jobs` come from `ModelConfig`.

Grid searched with 3-fold CV, scored on roc_auc (18 combos, 54 fits):

| Parameter | Values | What it controls |
|---|---|---|
| `n_estimators` | 50, 200, 500 | Number of trees |
| `max_depth` | 5, 15, 20, 30, 40 | Max splits per tree |
| `min_samples_split` | 2, 10 | Min rows a node needs before it splits |

The grid ranges are the ones listed in the research doc. After the search, the best estimator is cross validated with 5 folds on the train split and then scored once on the 20% hold-out.

## Results

Full grid run, cutoff day 32, withdrawn included as at-risk, 6,519 test rows.

| Metric | 5-fold CV (train) | Hold-out (test) |
|---|---|---|
| Accuracy | 77.00% (+/- 0.70%) | 78.29% |
| Precision | 83.29% (+/- 0.59%) | 84.14% |
| Recall | 70.61% (+/- 1.23%) | 72.57% |
| F1 | 76.42% (+/- 0.85%) | 77.93% |
| ROC-AUC | 85.52% (+/- 0.40%) | 86.54% |

Hold-out AUC of 0.865

Confusion matrix on the hold-out:

|  | Not At Risk | At Risk |
|---|---|---|
| **Not At Risk** | 2606 | 471 |
| **At Risk** | 944 | 2498 |

944 at-risk students are missed (predicted positive) against 471 false positives. Precision is high because when the forest does flag someone it is usually right, but it leaves about 1 in 4 at-risk students unflagged at the default 0.5 threshold. `results/random_forest_precision_recall.png` shows the trade-off.

### Feature importance

Gini importance and permutation importance (how much test AUC drops when the column is shuffled) show slightly different results:

| Rank | Gini | Permutation |
|---|---|---|
| 1 | `mean_score` 0.095 | `unregistered_before_cutoff` 0.026 |
| 2 | `weighted_score` 0.076 | `mean_score` 0.014 |
| 3 | `avg_days_before_due` 0.071 | `weighted_score` 0.007 |
| 4 | `unregistered_before_cutoff` 0.059 | `clicks_page` 0.006 |
| 5 | `engagement_score` 0.045 | `clicks_url` 0.004 |

Gini spreads credit across the many correlated click columns (`total_clicks`, `active_days`, `clicks_homepage`, `engagement_score` all land in the top 10 at similar values) because any of them can be used for the same split. Permutation meanwhile, shuffles one click column and the others cover for it, so none of them individually moves AUC much. What cannot be covered for is `unregistered_before_cutoff` and the assessment scores.

The demographics (`imd_band`, `age_band`, `region_*`) sit at the bottom of both rankings. 

## Next steps
