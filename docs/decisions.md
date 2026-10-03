# Decisions

## Target and population

- **Binary target**: `at_risk = 1` for `Fail` and `Withdrawn`, `0` for `Pass` and `Distinction`. Withdrawal is non-completion just like failing, and both are what an early-warning flag is meant to prevent. Distinction is a pass, so it joins `Pass`.

- **Population at each cutoff**: students who unregistered on or before the cutoff day are dropped. They have already left, so there is nothing for the model to predict. 5,047 of the 5,055 dropped at week 4 end as `Withdrawn`, so keeping them lets the model read the label straight off the unregistration date.

- **`unregistered_before_cutoff` removed as a feature**: it acts similiar to the `Withdrawn` label. It was the #1 gain feature in XGBoost (0.30) before removal. It is still computed for the population filter.

- **Resulting class balance**: 44.1% at-risk at week 4, 42.0% at week 8, 40.2% at week 12, from 27,538 / 26,522 / 25,724 of 32,593 student-presentations.

## Time window and leakage

- **Prediction cutoffs**: end of weeks 4, 8 and 12 (days 28, 56, 84 from the presentation start). The default is day 28.

- **Leakage rule**: nothing dated after the cutoff is used. Clicks with `date > cutoff` are dropped, assessments due after the cutoff are dropped, and so are submissions made after it. `final_result` is used only to build `y`.

- **End-of-module exams excluded**: assessments with no due date (which are mostly final exams) fail the `date <= cutoff` test and drop out. Their timing is unknown, so they can't be placed before a cutoff.

- **Pre-course clicks kept**: clicks with negative dates (before day 0) count, since an instructor can see them by the cutoff.

## Missing values and encoding

- **No activity means zero**: a student with no click or submission rows before the cutoff gets 0 for the count features (`total_clicks`, `clicks_*`, `n_submitted`, `submission_rate`, `pct_banked`). No record means no activity, not unknown activity.

- **Missing stays missing in ratios**: `_safe_div` returns NaN when the numerator is missing and 0 only when the denominator is 0, so the imputer, not a hard-coded 0, handles truly unknown values.

- **Numeric imputation**: `IterativeImputer` (MICE-style), fit on the training split only and then applied to test, followed by `StandardScaler`.

- **Ordinal encoding** for `imd_band`, `age_band` and `highest_education`, since the bands have a natural order. Unmapped labels become NaN and are imputed.

- **`region`**: most-frequent imputation, then one-hot encoding. It is nominal, so there is no ordering to preserve.

## Data split and validation

- **Hold-out**: 80/20 split, stratified on `at_risk`, seed 42. The pipeline runs once per cutoff and every model gets the identical split.

- **Tuning**: `GridSearchCV` with 3 folds (per the research doc), selecting on F1. F1 balances catching at-risk students (recall) against false alarms (precision) on the positive class.

- **Reported CV**: the tuned model is re-scored with stratified, shuffled 5-fold CV on the training split, reporting the mean +/- std of accuracy, precision, recall, F1 and ROC-AUC.

- **Decision threshold**: 0.5, default threshold. The precision/recall-vs-threshold plots are saved so a different threshold points can be chosen  later.

## Class imbalance

- **Same correction for every model**: Random Forest and Logistic Regression use `class_weight='balanced'`. XGBoost uses `scale_pos_weight` = negatives/positives, computed from `y_train` at runtime (1.27 / 1.38 / 1.49 at weeks 4 / 8 / 12). Both give the positive class the same relative weight, so the comparison doesn't confound algorithm with imbalance handling.

## Model setups

- **Three models**: Logistic Regression (baseline), Random Forest and XGBoost.

- **RF grid**: `n_estimators` {50, 200, 500}, `max_depth` {5, 15, 20, 30}, `min_samples_split` {2, 10}. Selected `max_depth` 15 / 15 / 30 at weeks 4 / 8 / 12.

- **XGBoost grid**: `learning_rate` {0.01, 0.1, 0.3}, `max_depth` {3, 6, 10}, `reg_lambda` {1, 5} (see notes for more). Selected `max_depth=3` at every cutoff and `learning_rate` 0.3 / 0.1 / 0.3 at weeks 4 / 8 / 12 (75 / 199 / 74 trees). 
- **XGBoost early stopping (only for tuning)**: a stratified 15% validation slice is carved off the training split. Each fit allows up to 1,000 rounds and stops after 20 rounds without improvement in validation log-loss. Tuning picks the grid parameters and the tree count (`best_iteration + 1`).
- **XGBoost final fit**: refit on the full training split with the tree count fixed and early stopping off, so XGBoost trains on exactly the same rows as RF and LR. The tree count was chosen on 85% of the data and applied to 100%.
- **XGBoost `tree_method='hist'`**: histogram-binned splits, for speed on ~20k rows.

- **LR solver and penalty**: `saga` with `l1_ratio` elements {0 (L2), 1 (L1)}, `max_iter=5000`.
- **LR C grid**: {0.001, 0.01, 0.1, 1}. `C=10` was dropped because it hit `max_iter` without converging. At first I thought it was the "correlated click columns", but after `progress_fraction` was removed, `C=10` converges in about 240 iterations at weeks 4 and 8 for both L1 and L2, so the `progress_fraction` / `module_presentation_length` pair was the cause. Selected: `C=0.1` L1 at weeks 4 and 12, `C=1` L2 at week 8. The week 8 fit originally hit `max_iter` without converging. That was fixed by removing `progress_fraction`, and every fit now converges.

## Feature ranking

- **Native importance**: Gini is used for RF, gain for XGBoost, coefficient for LR. Features are standardized, so coefficient sizes are comparable and the top 20 are kept.

- **Permutation importance**: Top 20 are kept on the hold-out set, scored by F1, 5 repeats per feature. It is model-agnostic, so it is the like-for-like comparison across all three models.

- **Next: SHAP** (global summary plus local explanations) for all three models.

## Notes
- **XGBoost reg_lamda**: `reg_lambda` (L2 penalty on leaf weights) was added to give the search a direct overfitting control, and test stronger penalties. This addition held no meaningful impact to the original results which chose the default value 1, and `reg_lambda = 5` was only chosen in week 4 and never after with the additional cutoff handling. To be removed. 

## TODO Questions

- Should the operating threshold move off 0.5 based on the precision/recall plots?
