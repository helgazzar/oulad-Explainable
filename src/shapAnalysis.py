from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

import models

# rows explained for random forest
MAX_EXPLAIN_ROWS = 1000
# background rows for LinearExplainer, sampled from the train split
BACKGROUND_ROWS = 1000
# features given their own dependence plot
DEPENDENCE_FEATURES: tuple[str, ...] = ("submission_rate", "mean_score", "active_days", "clicks_page")
# features shown in each plot
WATERFALL_DISPLAY = 12

# columns (click counts, score averages, region)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "demographic": ("imd_band_ordinal", "age_band_ordinal", "highest_education_ordinal",
                    "disability_flag", "gender_flag", "num_of_prev_attempts", "studied_credits"),
    "course": ("module_presentation_length",),
    "registration": ("registration_lead_days", "registered_late"),
    "vle": ("total_clicks", "active_days", "distinct_sites", "avg_clicks_per_active_day", "engagement_score"),
    "assessment": ("n_submitted", "mean_score", "weighted_score", "avg_days_before_due",
                   "pct_banked", "submission_rate"),
}


### readable feature values
# models see imputed + standardized columns
def unscaled_features(X, preprocessor) -> np.ndarray:
    """X with the numeric columns mapped back to their original units (imputed values stay imputed)."""
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    n_numeric = len(scaler.mean_)
    out = np.array(X, dtype=float, copy=True)
    # numeric columns first in the ColumnTransformer output, then one-hot columns
    out[:, :n_numeric] = out[:, :n_numeric] * scaler.scale_ + scaler.mean_
    return out


### explain
# background = train split, explained rows = hold-out, same split as permutation importance
def explain_rows(result: dict, data: dict, cfg: models.ModelConfig) -> np.ndarray:
    """Positions in X_test to explain, a stratified sample for the random forest, all rows otherwise."""
    n_test = len(data["y_test"])
    if result["name"] != "random_forest" or n_test <= MAX_EXPLAIN_ROWS:
        return np.arange(n_test)
    rows, _ = train_test_split(
        np.arange(n_test), train_size=MAX_EXPLAIN_ROWS,
        stratify=data["y_test"], random_state=cfg.random_state,
    )
    return np.sort(rows)


def build_explainer(result: dict, data: dict, cfg: models.ModelConfig):
    """SHAP explainer that matches the model type."""
    names = data["feature_names"]
    if result["name"] == "logistic_regression":
        # SHAP = coef * (x - background mean), in log-odds
        masker = shap.maskers.Independent(data["X_train"], max_samples=BACKGROUND_ROWS)
        return shap.LinearExplainer(result["model"], masker, feature_names=names)
    # xgboost output = log-odds, random forest output = probability
    return shap.TreeExplainer(result["model"], feature_names=names)


def explain(result: dict, data: dict, cfg: models.ModelConfig) -> tuple[shap.Explanation, np.ndarray]:
    """SHAP values for the at-risk class on hold-out rows, plus the X_test positions they belong to."""
    rows = explain_rows(result, data, cfg)
    X = data["X_test"][rows]
    raw = build_explainer(result, data, cfg)(X)

    values = raw.values
    base = np.asarray(raw.base_values)
    # random forest returns one column per class, keep the at-risk class
    if values.ndim == 3:
        values = values[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else np.full(len(rows), base[1])
    base = np.broadcast_to(base, (len(rows),)).astype(float)

    explanation = shap.Explanation(
        values=values,
        base_values=base,
        data=unscaled_features(X, data["preprocessor"]),
        feature_names=data["feature_names"],
    )
    return explanation, rows


### checks
# base value + sum of SHAP for every explained row
def model_output(result: dict, X) -> np.ndarray:
    """Raw output for models."""
    model = result["model"]
    if result["name"] == "random_forest":
        return model.predict_proba(X)[:, 1]
    if result["name"] == "xgboost":
        return model.predict(X, output_margin=True)
    return model.decision_function(X)


def additivity_error(result: dict, values: shap.Explanation, data: dict, rows: np.ndarray) -> float:
    """Largest difference between base + sum(SHAP) and the model output."""
    rebuilt = values.base_values + values.values.sum(axis=1)
    return float(np.abs(rebuilt - model_output(result, data["X_test"][rows])).max())


### tables
def shap_importance(values: shap.Explanation, cfg: models.ModelConfig) -> pd.DataFrame:
    """Top features by mean SHAP on the explained hold-out rows."""
    out = pd.DataFrame({
        "feature": values.feature_names,
        "importance": np.abs(values.values).mean(axis=0),
        # mean, positive = pushes toward at-risk on average
        "mean_shap": values.values.mean(axis=0),
    })
    return out.sort_values("importance", ascending=False).head(cfg.top_n_features).reset_index(drop=True)


def feature_group(name: str) -> str:
    """Model columns."""
    if name.startswith("region_"):
        return "region"
    if name.startswith("clicks_"):
        return "vle"
    for group, members in FEATURE_GROUPS.items():
        if name in members:
            return group
    return "other"


def grouped_importance(values: shap.Explanation) -> pd.DataFrame:
    """Mean SHAP per feature family, SHAP summed within the family per student first."""
    groups = np.array([feature_group(n) for n in values.feature_names])
    rows = []
    for group in sorted(set(groups)):
        per_student = values.values[:, groups == group].sum(axis=1)
        rows.append({
            "group": group,
            "n_features": int((groups == group).sum()),
            "importance": float(np.abs(per_student).mean()),
            "mean_shap": float(per_student.mean()),
        })
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


### plots
def _save(path) -> None:
    """Save the matplotlib fig."""
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")


def beeswarm_plot(values: shap.Explanation, name: str, cfg: models.ModelConfig) -> None:
    """Total summary."""
    shap.plots.beeswarm(values, max_display=cfg.top_n_features, show=False)
    plt.title(f"SHAP summary - {name}")
    _save(cfg.results_dir / f"{name}_shap_beeswarm.png")


def dependence_plots(values: shap.Explanation, name: str, cfg: models.ModelConfig) -> None:
    """SHAP value against feature value for early-warning features."""
    for feature in DEPENDENCE_FEATURES:
        if feature not in values.feature_names:
            continue
        shap.plots.scatter(values[:, feature], show=False)
        plt.title(f"SHAP dependence - {feature} - {name}")
        _save(cfg.results_dir / f"{name}_shap_dependence_{feature}.png")


def pick_students(values: shap.Explanation, rows: np.ndarray, data: dict, result: dict) -> dict[str, int]:
    """Explained-row positions for three local examples: most confident hit, most confident miss, borderline."""
    proba = result["model"].predict_proba(data["X_test"][rows])[:, 1]
    actual = data["y_test"].to_numpy()[rows]
    picks = {}

    # at-risk student for pass
    hits = np.where((actual == 1) & (proba >= 0.5))[0]
    if len(hits):
        picks["true_positive"] = int(hits[np.argmax(proba[hits])])

    # at-risk student for fail
    misses = np.where((actual == 1) & (proba < 0.5))[0]
    if len(misses):
        picks["false_negative"] = int(misses[np.argmin(proba[misses])])
    
    # student closest to the 0.5 threshold
    picks["borderline"] = int(np.argmin(np.abs(proba - 0.5)))
    return picks


def waterfall_plots(values: shap.Explanation, rows: np.ndarray, data: dict, result: dict,
                    cfg: models.ModelConfig) -> None:
    """Per-student breakdown of how each feature moved the prediction from the base value."""
    for label, i in pick_students(values, rows, data, result).items():
        shap.plots.waterfall(values[i], max_display=WATERFALL_DISPLAY, show=False)
        plt.title(f"{label.replace('_', ' ')} - {result['name']}")
        _save(cfg.results_dir / f"{result['name']}_shap_waterfall_{label}.png")


### run
def run(result: dict, data: dict, cfg: models.ModelConfig) -> dict:
    """SHAP tables and plots for a fitted model."""
    name = result["name"]
    values, rows = explain(result, data, cfg)

    importance = shap_importance(values, cfg)
    grouped = grouped_importance(values)
    models.save_table(importance, f"{name}_shap_importance.csv", cfg)
    models.save_table(grouped, f"{name}_shap_grouped_importance.csv", cfg)

    beeswarm_plot(values, name, cfg)
    dependence_plots(values, name, cfg)
    waterfall_plots(values, rows, data, result, cfg)

    return {
        "name": name,
        "values": values,
        "rows": rows,
        "importance": importance,
        "grouped_importance": grouped,
        "additivity_error": additivity_error(result, values, data, rows),
    }


def run_all(results: list[dict], data: dict, cfg: models.ModelConfig) -> list[dict]:
    """SHAP for every fitted model on one cutoff + print top features and additivity check."""
    out = []
    for result in results:
        shap_result = run(result, data, cfg)
        print(f"\n{result['name']} SHAP (rows explained: {len(shap_result['rows'])}, "
              f"additivity max error: {shap_result['additivity_error']:.2e})")
        print(shap_result["importance"].head(10).to_string(index=False))
        out.append(shap_result)
    return out

# standalone run 
if __name__ == "__main__":
    import logisticModel
    import main
    import randomForestModel
    import xgboostModel

    week = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = main.cutoff_config(models.ModelConfig(), week)
    data = models.load_data(cfg)
    results = [
        logisticModel.run(cfg, data),
        randomForestModel.run(cfg, data),
        xgboostModel.run(cfg, data),
    ]
    run_all(results, data, cfg)
