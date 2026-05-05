"""DVC stage entry point — delegates to ai.training.trainer.run."""

from __future__ import annotations

import logging

import yaml

from ai.training.trainer import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    run(params)


if __name__ == "__main__":
    main()
