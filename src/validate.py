from pathlib import Path
from data_loader import load_splits
from logger import logger
import pickle
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

import yaml
from utils import get_validate_args, load_params


# PROJECT_ROOT = Path(__file__).parent.parent


# def load_params(params_path=None):
#     if params_path is None:
#         params_path = PROJECT_ROOT / "config" / "params.yaml"
#     with open(params_path, "r") as f:
#         return yaml.safe_load(f)
    


def load_model(path):
    # if path is None:
    model_path = Path(path / "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate(model, X_test, y_test):
    logger.info("Starting evaluation.")
    y_test = np.array(y_test, dtype=int)  # ← move to top
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
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
    

def save_metrics(metrics,previous_metrics=None,path=None):
    if path is None:
        path = Path("reports/metrics.json")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {path}")

    
    # Also save as markdown
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

        # At the end of save_metrics, after confusion matrix
        if previous_metrics:
            f.write("\n## Metrics Comparison vs Previous Run\n\n")
            f.write("| Metric | Previous | Current | Delta |\n")
            f.write("|--------|----------|---------|-------|\n")
            for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                prev = previous_metrics.get(key, "N/A")
                curr = metrics[key]
                delta = round(curr - prev, 4) if prev != "N/A" else "N/A"
                f.write(f"| {key} | {prev} | {curr} | {delta} |\n")

    logger.info(f"Metrics saved to {path}")

def save_comparison(current, previous, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if previous is None:
        logger.info("No previous metrics — writing empty comparison.")
        with open(path, "w") as f:
            json.dump({}, f, indent=4)
        return

    comparison = {}
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    for key in keys:
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

if __name__ == "__main__":
    args = get_validate_args()

    params = load_params(args.params)
    PROCESSED_PATH = params["validation_data"]["path"]
    METRICS_PATH =  Path(params["reports"]["metrics_path"])
    PREVIOUS_METRIC_PATH = params["reports"]["comparison_path"]
    MODEL_PATH = Path(params["model"]["artifacts_path"]) 
    X_train, X_test, y_train, y_test = load_splits(path=PROCESSED_PATH)
    model = load_model(MODEL_PATH)
    previous_metrics = load_previous_metrics(METRICS_PATH)
    current_metrics = evaluate(model, X_test, y_test)
    compare_metrics(current_metrics, previous_metrics)
    save_comparison(current_metrics, previous_metrics,path=PREVIOUS_METRIC_PATH)   # ← add this
    save_metrics(current_metrics, previous_metrics, path=METRICS_PATH )