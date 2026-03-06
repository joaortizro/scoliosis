import yaml
from ai.preprocessing.pipeline import preprocess

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    preprocess(params)
