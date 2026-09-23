"""Train a small weather-to-power neural model from random initialization."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import platform
import numpy as np
import pandas as pd
from .dataset import sha256
from .features import BASE_FEATURES, make_features

def split_masks(frame, validation_start, test_start, test_end):
    t = pd.to_datetime(frame.valid_time_utc, utc=True)
    a, b, c = [pd.Timestamp(x, tz="UTC") for x in (validation_start, test_start, test_end)]
    if not a < b < c:
        raise ValueError("Split dates must increase")
    if (t >= c).any():
        raise ValueError("Dataset includes rows on/after the test end; remove forbidden labels")
    masks = {"train": t < a, "validation": (t >= a) & (t < b), "test": (t >= b) & (t < c)}
    if any(not m.any() for m in masks.values()):
        raise ValueError("Every chronological split needs data")
    return masks

def metrics(y, pred):
    error = np.asarray(pred) - np.asarray(y)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean()), "rows": len(error)}

def train(dataset, output_dir, device="cuda", epochs=80, batch_size=512, seed=42, validation_start="2025-10-01", test_start="2026-01-01", test_end="2026-02-01", patience=10):
    import torch
    from torch import nn
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch size and patience must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Run training on the remote GPU worker; --device cpu is only an explicit alternative.")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    data_path = Path(dataset)
    metadata = json.loads(data_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if sha256(data_path) != metadata["dataset_sha256"]:
        raise ValueError("Dataset checksum does not match assembly metadata")
    frame = pd.read_csv(data_path)
    if frame.duplicated(["turbine_id", "valid_time_utc", "provider_offset_days"]).any():
        raise ValueError("Duplicate training examples")
    if set(frame.model) != {metadata["weather_model"]}:
        raise ValueError("Weather-model metadata mismatch")
    y = frame.target_power.to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError("Targets must be finite normalized power in 0..1")
    masks = split_masks(frame, validation_start, test_start, test_end)
    turbines = sorted(frame.loc[masks["train"], "turbine_id"].unique().tolist())
    x = make_features(frame, turbines)
    mean, scale = x[masks["train"]].mean(axis=0), x[masks["train"]].std(axis=0)
    scale = np.where(scale < 1e-6, 1, scale)
    x = (x - mean) / scale
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(x[masks["train"]], device=device)
    yt = torch.tensor(y[masks["train"]], device=device).reshape(-1, 1)
    xv = torch.tensor(x[masks["validation"]], device=device)
    yv = torch.tensor(y[masks["validation"]], device=device).reshape(-1, 1)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); order = torch.randperm(len(xt), device=device)
        total = 0.0
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[indices]), yt[indices]); loss.backward(); optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode(): val_mae = (model(xv) - yv).abs().mean().item()
        record = {"epoch": epoch, "training_mse": total / len(xt), "validation_mae": val_mae}
        history.append(record); print(json.dumps(record), flush=True)
        if val_mae < best - 1e-6:
            best, best_epoch, best_state, stale = val_mae, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience: break
    if best_state is None: raise RuntimeError("Training did not produce finite validation loss")
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode(): predictions = model(torch.tensor(x, device=device)).cpu().numpy()[:, 0]
    # Both baselines use training labels only. Offset examples remain together in splits.
    train_frame = frame.loc[masks["train"]]
    constant = np.full(len(frame), y[masks["train"]].mean())
    curve = np.empty(len(frame))
    for turbine in turbines:
        subset = train_frame.loc[train_frame.turbine_id == turbine].copy()
        subset["bin"] = np.floor(subset.wind_speed_10m_ms / 0.5) * 0.5 + 0.25
        grouped = subset.groupby("bin").target_power.mean().sort_index()
        selected = frame.turbine_id == turbine
        curve[selected] = np.interp(frame.loc[selected, "wind_speed_10m_ms"], grouped.index, grouped.to_numpy())
        constant[selected] = subset.target_power.mean()
    report = {"schema_version": 1, "dataset_metadata": metadata, "device": device, "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}, "seed": seed, "best_epoch": best_epoch, "parameter_count": sum(p.numel() for p in model.parameters()), "split_boundaries_utc": {"validation_start": validation_start, "test_start": test_start, "test_end": test_end}, "metrics": {}, "metrics_by_offset": {}, "history": history}
    report["metrics_by_turbine"] = {}
    for name, mask in masks.items():
        report["metrics"][name] = {label: metrics(y[mask], pred[mask]) for label, pred in [("neural", predictions), ("constant", constant), ("wind_curve", curve)]}
        report["metrics_by_offset"][name] = {str(offset): metrics(y[mask & (frame.provider_offset_days == offset)], predictions[mask & (frame.provider_offset_days == offset)]) for offset in sorted(frame.loc[mask, "provider_offset_days"].unique())}
        report["metrics_by_turbine"][name] = {}
        for turbine in turbines:
            selected = mask & frame.turbine_id.eq(turbine)
            if not selected.any():
                report["metrics_by_turbine"][name][turbine] = {"rows": 0, "status": "no_data"}
                continue
            report["metrics_by_turbine"][name][turbine] = {
                "unique_target_hours": int(frame.loc[selected, "valid_time_utc"].nunique()),
                "models": {label: metrics(y[selected], pred[selected]) for label, pred in [("neural", predictions), ("constant", constant), ("wind_curve", curve)]},
                "neural_by_offset": {str(offset): metrics(y[selected & frame.provider_offset_days.eq(offset)], predictions[selected & frame.provider_offset_days.eq(offset)]) for offset in sorted(frame.loc[selected, "provider_offset_days"].unique())},
            }
    layers = [{"weight": layer.weight.detach().cpu().tolist(), "bias": layer.bias.detach().cpu().tolist()} for layer in model if isinstance(layer, nn.Linear)]
    version = "mlp-" + hashlib.sha256(json.dumps(layers, sort_keys=True).encode()).hexdigest()[:12]
    bundle = {"schema_version": 1, "architecture": "mlp_relu_sigmoid", "model_version": version, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "weather_model": metadata["weather_model"], "turbines": turbines, "feature_names": BASE_FEATURES + ["turbine:" + t for t in turbines], "mean": mean.tolist(), "scale": scale.tolist(), "layers": layers, "dataset_metadata": metadata, "training_seed": seed, "best_epoch": best_epoch}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(bundle, allow_nan=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    held = frame.loc[masks["test"], ["turbine_id", "valid_time_utc", "provider_offset_days", "target_power"]].copy()
    held["predicted_power"] = predictions[masks["test"]]
    held.to_csv(out / "evaluation_predictions.csv", index=False)
    # Check portable inference against the trained framework before exporting as usable.
    from .model import Predictor
    portable = Predictor(out / "model.json", allow_provisional=True)
    for turbine in turbines:
        indices = np.flatnonzero(frame.turbine_id.eq(turbine))[:128]
        np.testing.assert_allclose(portable.predict(frame.iloc[indices], metadata["weather_model"]), predictions[indices], atol=2e-6, rtol=1e-5)
    print(json.dumps({"model_version": version, "test": report["metrics"]["test"]}), flush=True)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-start", default="2025-10-01")
    p.add_argument("--test-start", default="2026-01-01")
    p.add_argument("--test-end", default="2026-02-01")
    p.add_argument("--patience", type=int, default=10)
    args = vars(p.parse_args())
    train(**args)

if __name__ == "__main__": main()
