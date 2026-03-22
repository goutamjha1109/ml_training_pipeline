from pathlib import Path
from data_loader import load_splits, load_pipeline
from logger import logger
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from utils import get_validate_args, load_params


def evaluate(pipeline, X_test, y_test):
    logger.info("Starting evaluation.")
    y_test = np.array(y_test, dtype=int)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info(f"Metrics: { {k: v for k, v in metrics.items() if k != 'confusion_matrix'} }")
    return metrics


def load_previous_metrics(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def compare_metrics(current, previous):
    if previous is None:
        logger.info("No previous metrics found. Skipping comparison.")
        return
    logger.info("---- Metrics Comparison ----")
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    for key in keys:
        curr_val = current[key]
        prev_val = previous.get(key, None)
        if prev_val is not None:
            delta = round(curr_val - prev_val, 4)
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            logger.info(f"{key:12s}: {prev_val} → {curr_val}  {direction} ({delta:+.4f})")


def save_metrics(metrics, path, previous_metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {path}")

    md_path = path.parent / "metrics.md"
    cm = metrics["confusion_matrix"]
    with open(md_path, "w") as f:
        f.write("## Model Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                f.write(f"| {k} | {v} |\n")
        f.write("\n## Confusion Matrix\n\n")
        f.write("| | Predicted 0 | Predicted 1 |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Actual 0 | {cm[0][0]} | {cm[0][1]} |\n")
        f.write(f"| Actual 1 | {cm[1][0]} | {cm[1][1]} |\n")
        if previous_metrics:
            f.write("\n## Metrics Comparison vs Previous Run\n\n")
            f.write("| Metric | Previous | Current | Delta |\n")
            f.write("|--------|----------|---------|-------|\n")
            for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                prev = previous_metrics.get(key, "N/A")
                curr = metrics[key]
                delta = round(curr - prev, 4) if prev != "N/A" else "N/A"
                f.write(f"| {key} | {prev} | {curr} | {delta} |\n")
    logger.info(f"Markdown saved to {md_path}")


def save_comparison(current, previous, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if previous is None:
        logger.info("No previous metrics — writing empty comparison.")
        with open(path, "w") as f:
            json.dump({}, f, indent=4)
        return
    comparison = {}
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        curr_val = current[key]
        prev_val = previous.get(key)
        if prev_val is not None:
            delta = round(curr_val - prev_val, 4)
            comparison[key] = {
                "previous": prev_val,
                "current": curr_val,
                "delta": delta,
                "direction": "up" if delta > 0 else ("down" if delta < 0 else "same")
            }
    with open(path, "w") as f:
        json.dump(comparison, f, indent=4)
    logger.info(f"Comparison saved to {path}")


def run_validation():
    args = get_validate_args()
    params = load_params(args.params)

    paths = params["paths"]
    PROCESSED_PATH  = Path(paths["processed_data"])
    ARTIFACTS_PATH  = Path(paths["artifacts"])
    METRICS_PATH    = Path(paths["metrics"])
    COMPARISON_PATH = Path(paths["comparison"])

    X_train, X_test, y_train, y_test = load_splits(PROCESSED_PATH)
    pipeline, le_target = load_pipeline(ARTIFACTS_PATH)

    previous_metrics = load_previous_metrics(METRICS_PATH)
    current_metrics  = evaluate(pipeline, X_test, y_test)
    compare_metrics(current_metrics, previous_metrics)
    save_comparison(current_metrics, previous_metrics, path=COMPARISON_PATH)
    save_metrics(current_metrics, path=METRICS_PATH, previous_metrics=previous_metrics)


if __name__ == "__main__":
    run_validation()