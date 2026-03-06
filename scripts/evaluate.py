import yaml
from ai.evaluation.evaluator import evaluate

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    evaluate(params)
