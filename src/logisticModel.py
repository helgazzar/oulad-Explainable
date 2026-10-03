# logistic regression, the interpretable baseline, tuned by grid search on the pipeline output
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

import models
import plots

NAME = "logistic_regression"

### grid params
PARAM_GRID: dict[str, list] = {
    # inverse penalty strength, lower = stronger penalty. stops at 1.0, C = 10 hit max_iter
    # without converging while progress_fraction was a feature (now removed, see docs/decisions.md)
    "C": [0.001, 0.01, 0.1, 1.0],
    "l1_ratio": [0.0, 1.0],
}


### logistic regression set up
def build(cfg: models.ModelConfig) -> LogisticRegression:
    """Raw logistic regression carrying the seed from cfg."""
    # class_weight = 'balanced' matches the forest and xgboost's scale_pos_weight
    # solver = 'saga' supports both L1 and L2, and features are already scaled by the pipeline
    # max_iter raised since saga converges slowly on 20k+ rows
    return LogisticRegression(
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=cfg.random_state,
    )


### train
# tune -> cross validate -> evaluate -> importance, pass data in to skip re-running the pipeline
def run(cfg: models.ModelConfig | None = None, data: dict | None = None) -> dict:
    """Tune, evaluate, and rank features for logistic regression."""
    # build cfg per call
    if cfg is None:
        cfg = models.ModelConfig()
    if data is None:
        data = models.load_data(cfg)

    search = models.tune(build(cfg), PARAM_GRID, data["X_train"], data["y_train"], cfg)
    # create model based on the 'best' parameters found
    model = search.best_estimator_

    # call cross_validate using the tuned model to find stats + std. dev
    cv_metrics = models.cross_validate_model(model, data["X_train"], data["y_train"], cfg)

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
        "best_params": search.best_params_,
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
    models.print_results(result, cfg)
    plots.plot_all([result], data, cfg)
