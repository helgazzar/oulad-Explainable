# random forest, tuned by grid search on the pipeline output
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

import models
import plots

NAME = "random_forest"

### grid params
PARAM_GRID: dict[str, list] = {
    "n_estimators": [50, 200, 500],  # number of trees
    "max_depth": [5, 15, 20, 30, 40],        # max splits per tree
    "min_samples_split": [2, 10],    # min rows needed before a node splits
}


### random forest set up
def build(cfg: models.ModelConfig) -> RandomForestClassifier:
    """Raw Random Forest carrying the seed and parallelism from cfg."""
    # class_weight = 'balanced' weights the at-risk class up so the forest
    # doesn't just predict pass for everyone
    return RandomForestClassifier(
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight="balanced",
    )


### train
# tune -> cross validate -> evaluate -> importance, pass data in to skip re-running the pipeline
def run(cfg: models.ModelConfig = models.ModelConfig(), data: dict | None = None) -> dict:
    """Tune, evaluate, and rank features for the random forest."""
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
    models.print_results(result)
    plots.plot_all([result], data, cfg)
