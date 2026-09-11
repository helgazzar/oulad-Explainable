# shared pieces for the model files, randomForest.py and xgboostModel.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from scipy import sparse

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import pipeline


# Use to set default values for the training run.
# passing a different values overwrites the defaults.
@dataclass(frozen=True)
class ModelConfig:

    # settings for pipeline.run(), defaults to PreprocessConfig
    preprocess: pipeline.PreprocessConfig = field(default_factory=pipeline.PreprocessConfig)

    # k folds for GridSearch, 3 per research doc
    cv_folds: int = 3

    # k folds for cross validating the tuned model, 5 like the knn assignment
    final_cv_folds: int = 5

    # metric GridSearch picks the best params by
    scoring: str = "f1"

    # number of shuffles per feature for permutation importance
    permutation_repeats: int = 5

    # static seed for reproducibility
    random_state: int = 42

    # -1 = use all cores
    n_jobs: int = -1

    # number of features kept in each importance table
    top_n_features: int = 20

    # TEST threshold for classifying a student as at-risk, 0.5 = default
    # threshold: float = 0.4

    # output folder for csv results
    results_dir: Path = Path(__file__).resolve().parent.parent / "results"


### load data
# run the pipeline once, both model files train on the same split
def load_data(cfg: ModelConfig) -> dict:
    """Preprocessed train/test split plus feature names, as one dict."""
    X_train, X_test, y_train, y_test, preprocessor, _ = pipeline.run(cfg.preprocess)

    # OneHotEncoder can return a sparse matrix, permutation_importance needs dense
    return {
        "X_train": _densify(X_train),
        "X_test": _densify(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names(preprocessor),
    }

def feature_names(preprocessor) -> list[str]:
    """Column names after the preprocessor, minus the numeric__ / categorical__ prefix."""
    return [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]

def _densify(X):
    """Convert sparse to dense."""
    return X.toarray() if sparse.issparse(X) else X


### grid search
# fit on x_train only, test set stays untouched until evaluate()
def tune(estimator, param_grid: dict[str, list], X_train, y_train, cfg: ModelConfig, **fit_params) -> GridSearchCV:
    """GridSearch the estimator, returns the fitted search. fit_params go to estimator.fit()."""
    # refit = True retrains on the full train set using the best params found
    search = GridSearchCV(
        estimator,
        param_grid,
        scoring=cfg.scoring,
        cv=cfg.cv_folds,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train, **fit_params)
    return search


### cross validation
# call cross_validate using the tuned model to find stats + std. dev
CV_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def cross_validate_model(model, X_train, y_train, cfg: ModelConfig, **fit_params) -> pd.DataFrame:
    """Mean and std of each metric across cfg.final_cv_folds on the train split."""
    # shuffle = True so folds aren't in file order, stratify keeps the class ratio per fold
    kfold = StratifiedKFold(n_splits=cfg.final_cv_folds, shuffle=True, random_state=cfg.random_state)
    results = cross_validate(model, X_train, y_train, cv=kfold, scoring=CV_METRICS,
                             n_jobs=cfg.n_jobs, params=fit_params)
    return pd.DataFrame({
        "metric": CV_METRICS,
        "mean": [results[f"test_{m}"].mean() for m in CV_METRICS],
        "std": [results[f"test_{m}"].std() for m in CV_METRICS],
    })


### evaluation
# same metrics for every model so they can be compared side by side
def evaluate(model, X_test, y_test, name: str) -> dict[str, float]:
    """Classification metrics for one fitted model on the hold-out."""
    # predicted labels for accuracy / precision / recall / f1
    y_pred = model.predict(X_test)
    # predicted probability of at-risk for roc_auc, [:, 1] = positive class column
    y_score = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


### feature importance
# two rankings per model, the model's own (gini for RF, gain for XGB)
# and permutation importance which works the same for any model
def native_importance(model, names: list[str], cfg: ModelConfig) -> pd.DataFrame:
    """Top features by the model's built-in feature_importances_."""
    out = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    # sort high to low, keep the top n
    return out.sort_values("importance", ascending=False).head(cfg.top_n_features).reset_index(drop=True)

def permutation_ranking(model, X_test, y_test, names: list[str], cfg: ModelConfig) -> pd.DataFrame:
    """Top features by how much the score drops when that column is shuffled."""
    # shuffle one column at a time, n_repeats times, and re-score the model
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=cfg.permutation_repeats,
        scoring=cfg.scoring,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    # mean drop = importance, std shows how stable it is across shuffles
    out = pd.DataFrame({
        "feature": names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    })
    return out.sort_values("importance", ascending=False).head(cfg.top_n_features).reset_index(drop=True)


### results output
# write tables to results/ so plots can be made without retraining
def save_table(table: pd.DataFrame, filename: str, cfg: ModelConfig) -> Path:
    """Write one table to cfg.results_dir, returns the path."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.results_dir / filename
    table.to_csv(path, index=False)
    return path

def print_results(result: dict) -> None:
    """Print best params, hold-out metrics, and top features for one model."""
    metrics = result["metrics"]
    print(f"\n{result['name']}")
    print("Best params:", result["best_params"])
    print(f"Best CV roc_auc: {result['cv_score']*100:.2f}%")
    # cross validation on the tuned model, mean (+/- std) per metric
    for row in result["cv_metrics"].itertuples():
        print(f"CV {row.metric}: {row.mean*100:.2f}% (+/- {row.std*100:.2f}%)")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"Test {metric}: {metrics[metric]*100:.2f}%")
    print("Top features (permutation):")
    print(result["permutation_importance"].head(10).to_string(index=False))
