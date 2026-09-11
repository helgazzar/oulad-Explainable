# entry point, preprocess once -> train both models -> print and save results
from __future__ import annotations

import pandas as pd

import models
import plots
import randomForest
import xgboostModel


def main() -> None:
    """Train Random Forest and XGBoost on the same split, print and save metrics."""
    cfg = models.ModelConfig()
    # run the pipeline once and hand the same split to both models
    data = models.load_data(cfg)

    results = [
        randomForest.run(cfg, data),
        xgboostModel.run(cfg, data),
    ]

    for result in results:
        models.print_results(result)

    # one row per model
    models.save_table(pd.DataFrame([r["metrics"] for r in results]), "model_metrics.csv", cfg)

    # confusion matrix per model + roc curve for both
    plots.plot_all(results, data, cfg)


if __name__ == "__main__":
    main()
