import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def preprocess(params: dict):
    raw_dir = Path(params["data"]["raw_dir"])
    processed_dir = Path(params["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implement preprocessing logic
    log.info("processing data from %s -> %s", raw_dir, processed_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = load_params()
    preprocess(params)
