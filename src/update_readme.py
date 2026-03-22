import json
import re
from pathlib import Path

from utils import load_params
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True)
    return parser.parse_args()

# PROJECT_ROOT = Path(__file__).parent.parent
# METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
# COMPARISON_PATH = PROJECT_ROOT / "reports" / "metrics_comparison.json"
# README_PATH = PROJECT_ROOT / "README.md"

args = get_args()
params = load_params(args.params)

METRICS_PATH    = Path(params["paths"]["metrics"])
COMPARISON_PATH = Path(params["paths"]["comparison"])
README_PATH     = Path("README.md")
# Load current metrics
with open(METRICS_PATH) as f:
    m = json.load(f)

# Build metrics table
table = "## Latest Metrics\n"
table += "<!-- METRICS_START -->\n"
table += "| Metric | Value |\n"
table += "|--------|-------|\n"
for k, v in m.items():
    if k != "confusion_matrix":
        table += f"| {k} | {v} |\n"

# Add confusion matrix
cm = m["confusion_matrix"]
table += "\n### Confusion Matrix\n"
table += "| | Predicted 0 | Predicted 1 |\n"
table += "|---|---|---|\n"
table += f"| Actual 0 | {cm[0][0]} | {cm[0][1]} |\n"
table += f"| Actual 1 | {cm[1][0]} | {cm[1][1]} |\n"

# Add comparison if exists
if COMPARISON_PATH.exists():
    with open(COMPARISON_PATH) as f:
        comp = json.load(f)

    table += "\n### Comparison vs Previous Run\n"
    table += "| Metric | Previous | Current | Delta | Direction |\n"
    table += "|--------|----------|---------|-------|-----------|\n"
    for key, val in comp.items():
        direction = "⬆️" if val["direction"] == "up" else ("⬇️" if val["direction"] == "down" else "➡️")
        table += f"| {key} | {val['previous']} | {val['current']} | {val['delta']:+.4f} | {direction} |\n"

table += "<!-- METRICS_END -->"

# Read current README
readme = README_PATH.read_text()

# Replace between markers
updated = re.sub(
    r"## Latest Metrics\n<!-- METRICS_START -->.*<!-- METRICS_END -->",
    table,
    readme,
    flags=re.DOTALL
)

README_PATH.write_text(updated)
print("README updated with latest metrics and comparison")