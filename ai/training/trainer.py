import mlflow
import yaml
from pathlib import Path


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def train(params: dict):
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(params["train"])
        mlflow.log_params(params["model"])

        # TODO: implement training loop
        print("Training started...")


if __name__ == "__main__":
    params = load_params()
    train(params)
