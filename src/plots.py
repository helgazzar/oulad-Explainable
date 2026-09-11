# plots for the model results, confusion matrix heat map + roc curve
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

import models


# axis labels for the at_risk target, 0 = pass, 1 = fail/withdrawn
CLASS_LABELS = ["Not At Risk", "At Risk"]

### confusion matrix
# one heat map per model, saved to results/
def confusion_heatmap(result: dict, data: dict, cfg: models.ModelConfig) -> None:
    """Seaborn heat map of the confusion matrix on the hold-out."""
    y_pred = result["model"].predict(data["X_test"])
    cm = confusion_matrix(data["y_test"], y_pred)

    plt.figure()
    # annot = write the count in each cell, fmt = 'd' for whole numbers
    sn.heatmap(cm, annot=True, fmt="d", cmap="Blues",
               xticklabels=CLASS_LABELS,
               yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {result['name']}")
    plt.tight_layout()
    plt.savefig(cfg.results_dir / f"{result['name']}_confusion_matrix.png")


### roc curve
# every model on the same axes so the curves can be compared directly
def roc_plot(results: list[dict], data: dict, cfg: models.ModelConfig) -> None:
    """ROC curve for each model in results, with AUC in the legend."""
    plt.figure()
    plt.title("ROC Curve")

    for result in results:
        # roc_curve needs the probability of at-risk, not the 0/1 label,
        # [:, 1] = positive class column
        y_score = result["model"].predict_proba(data["X_test"])[:, 1]
        fpr, tpr, threshold = roc_curve(data["y_test"], y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{result['name']} AUC = {roc_auc:.2f}")

    # dashed diagonal = random guessing
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(cfg.results_dir / "roc_curve.png")


### precision / recall vs threshold
# one per model, shows where to move the 0.5 cutoff
def precision_recall_plot(result: dict, data: dict, cfg: models.ModelConfig) -> None:
    """Precision and recall at every prediction threshold, on the hold-out."""
    # get raw probabilities before thresholding
    y_score = result["model"].predict_proba(data["X_test"])[:, 1]

    # calculate precision and recall at every threshold
    # arrays are one longer than thresholds, so drop the last point
    precision, recall, thresholds = precision_recall_curve(data["y_test"], y_score)

    plt.figure()
    plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
    plt.plot(thresholds, recall[:-1], label="Recall", color="red")
    # default cutoff used by .predict()
    plt.axvline(0.5, color="gray", linestyle="--", label="Default 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs Threshold - {result['name']}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg.results_dir / f"{result['name']}_precision_recall.png")


### all plots
# heat map + precision/recall per model, one shared roc curve, then show
def plot_all(results: list[dict], data: dict, cfg: models.ModelConfig) -> None:
    """Draw and save every figure for the given results."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        confusion_heatmap(result, data, cfg)
        precision_recall_plot(result, data, cfg)
    roc_plot(results, data, cfg)
    plt.show()
