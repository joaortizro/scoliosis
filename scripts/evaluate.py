"""DVC stage — Dice + Cobb MAE for the val split."""

from __future__ import annotations

import logging

import yaml

from ai.evaluation.evaluator import evaluate


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    evaluate(params)


if __name__ == "__main__":
    main()
