"""Portable NumPy inference for an MLP trained from scratch in PyTorch."""
import json
from pathlib import Path
import numpy as np
from .features import BASE_FEATURES, make_features

class Predictor:
    def __init__(self, path, allow_provisional=False):
        self.bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        b = self.bundle
        if b.get("schema_version") != 1 or b.get("architecture") != "mlp_relu_sigmoid":
            raise ValueError("Unsupported model artifact")
        if not b["dataset_metadata"].get("alignment_confirmed") and not allow_provisional:
            raise ValueError("Alignment is provisional; confirm data alignment or explicitly allow the development artifact")
        self.turbines = b["turbines"]
        expected = BASE_FEATURES + ["turbine:" + t for t in self.turbines]
        if b["feature_names"] != expected:
            raise ValueError("Artifact feature schema does not match this application")
        self.mean = np.asarray(b["mean"], dtype=np.float32)
        self.scale = np.asarray(b["scale"], dtype=np.float32)
        if self.mean.shape != (len(expected),) or self.scale.shape != self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid feature scaling")
        self.layers = []
        width = len(expected)
        for layer in b["layers"]:
            weight, bias = np.asarray(layer["weight"], dtype=np.float32), np.asarray(layer["bias"], dtype=np.float32)
            if weight.ndim != 2 or weight.shape[1] != width or bias.shape != (weight.shape[0],) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("Invalid network weights")
            self.layers.append((weight, bias)); width = weight.shape[0]
        if not self.layers or width != 1:
            raise ValueError("Network must produce one normalized power value per hour")

    def predict(self, frame, weather_model):
        if weather_model != self.bundle["weather_model"]:
            raise ValueError(f"Model trained for {self.bundle['weather_model']}, received {weather_model}")
        x = (make_features(frame, self.turbines) - self.mean) / self.scale
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        return (1 / (1 + np.exp(-np.clip(x[:, 0], -60, 60)))).astype(float)
