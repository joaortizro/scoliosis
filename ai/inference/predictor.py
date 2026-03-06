from pathlib import Path


class Predictor:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        # TODO: load model from checkpoint
        pass

    def predict(self, input_data):
        # TODO: run inference
        raise NotImplementedError
