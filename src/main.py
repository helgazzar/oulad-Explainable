# entry point, for each cutoff week: preprocess once -> train all three models -> print and save results
from __future__ import annotations

from dataclasses import replace

import pandas as pd

import logisticModel
import models
import plots
import randomForestModel
import xgboostModel

# weeks into the presentation the prediction is made at, cutoff_day = week * 7
CUTOFF_WEEKS: tuple[int, ...] = (4, 8, 12)


def cutoff_config(base: models.ModelConfig, week: int) -> models.ModelConfig:
    """Copy of base with the cutoff moved to the end of week and its own results/week_XX folder."""
    preprocess = replace(base.preprocess, cutoff_day=week * 7)
    return replace(base, preprocess=preprocess, results_dir=base.results_dir / f"week_{week:02d}")


def run_cutoff(cfg: models.ModelConfig, week: int) -> pd.DataFrame:
    """Train Logistic Regression, Random Forest and XGBoost on one cutoff's split, save tables and plots, return its metrics."""
    # run the pipeline once and hand the same split to every model
    data = models.load_data(cfg)

    results = [
        logisticModel.run(cfg, data),
        randomForestModel.run(cfg, data),
        xgboostModel.run(cfg, data),
    ]

    print(f"\n===== week {week} (day {cfg.preprocess.cutoff_day}) =====")
    for result in results:
        models.print_results(result, cfg)

    # one row per model, the population shrinks as the cutoff moves later
    # (more students have already withdrawn) so record its size and at-risk rate too
    metrics = pd.DataFrame([r["metrics"] for r in results])
    metrics.insert(0, "cutoff_week", week)
    metrics["n_train"] = len(data["y_train"])
    metrics["n_test"] = len(data["y_test"])
    metrics["at_risk_rate"] = float(data["y_train"].mean())
    models.save_table(metrics, "model_metrics.csv", cfg)

    # confusion matrix + precision/recall per model, one roc curve with every model
    plots.plot_all(results, data, cfg)
    return metrics


def main() -> None:
    """Run every cutoff week and save one combined metrics table to results/."""
    base = models.ModelConfig()
    all_metrics = [run_cutoff(cutoff_config(base, week), week) for week in CUTOFF_WEEKS]
    models.save_table(pd.concat(all_metrics, ignore_index=True), "model_metrics_by_cutoff.csv", base)


if __name__ == "__main__":
    main()
