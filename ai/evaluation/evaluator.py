import json
from pathlib import Path


def evaluate(params: dict):
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    # TODO: implement evaluation logic

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {results_dir / 'metrics.json'}")
