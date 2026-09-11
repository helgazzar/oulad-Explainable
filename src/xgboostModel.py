# xgboost, tuned by grid search on the pipeline output
# (file is not named xgboost.py so it doesn't shadow the xgboost package)
from __future__ import annotations

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import models
import plots

NAME = "xgboost"

# slice of the train split held out for early stopping
VAL_SIZE = 0.15
# stop adding trees once val logloss hasn't improved for n many rounds
EARLY_STOPPING_ROUNDS = 20

### grid params
PARAM_GRID: dict[str, list] = {
    "learning_rate": [0.01, 0.1, 0.3],  # step size, lower = slower but more stable
    "max_depth": [3, 6, 10],            # max splits per tree
    "reg_lambda": [1.0, 5.0],           # L2 penalty on leaf weights, higher = less overfit
}


### xgboost set up
def build(cfg: models.ModelConfig) -> XGBClassifier:
    """Raw XGBoost classifier carrying the seed and parallelism from cfg."""
    # n_estimators = max boosting rounds, set high and let early stopping pick the best count
    # tree_method = 'hist' bins the features first, faster on 32k rows
    # eval_metric = logloss for binary results, also what early stopping watches
    return XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        tree_method="hist",
        eval_metric="logloss",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


### train
# tune -> cross validate -> evaluate -> importance, pass data in to skip re-running the pipeline
def run(cfg: models.ModelConfig = models.ModelConfig(), data: dict | None = None) -> dict:
    """Tune, evaluate, and rank features for xgboost."""
    if data is None:
        data = models.load_data(cfg)

    # carve a validation slice off train for early stopping, test set stays untouched
    X_fit, X_val, y_fit, y_val = train_test_split(
        data["X_train"], data["y_train"],
        test_size=VAL_SIZE, random_state=cfg.random_state, stratify=data["y_train"],
    )
    # every fit watches the same val slice, verbose = False stops the per-round print
    fit_params = {"eval_set": [(X_val, y_val)], "verbose": False}

    search = models.tune(build(cfg), PARAM_GRID, X_fit, y_fit, cfg, **fit_params)
    # create model based on the 'best' parameters found
    model = search.best_estimator_

    # call cross_validate using the tuned model to find stats + std. dev
    cv_metrics = models.cross_validate_model(model, X_fit, y_fit, cfg, **fit_params)

    # score on the hold-out and build both importance tables
    metrics = models.evaluate(model, data["X_test"], data["y_test"], NAME)
    importance = models.native_importance(model, data["feature_names"], cfg)
    permutation = models.permutation_ranking(model, data["X_test"], data["y_test"], data["feature_names"], cfg)

    models.save_table(cv_metrics, f"{NAME}_cv_metrics.csv", cfg)
    models.save_table(importance, f"{NAME}_importance.csv", cfg)
    models.save_table(permutation, f"{NAME}_permutation_importance.csv", cfg)

    return {
        "name": NAME,
        "model": model,
        # best_iteration = round early stopping landed on, +1 for the tree count
        "best_params": {**search.best_params_, "n_estimators": model.best_iteration + 1},
        "cv_score": search.best_score_,
        "cv_metrics": cv_metrics,
        "metrics": metrics,
        "importance": importance,
        "permutation_importance": permutation,
    }


if __name__ == "__main__":
    cfg = models.ModelConfig()
    data = models.load_data(cfg)
    result = run(cfg, data)
    models.print_results(result)
    plots.plot_all([result], data, cfg)
