"""Join archived forecast predictors with hourly measured power, with explicit alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import WEATHER_COLUMNS

COLUMNS = {
    "Статистическое время": "time",
    "Нормализованная активная мощность": "power",
}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def hourly_targets(path, turbine_id, timezone, timestamp_position):
    frame = pd.read_csv(path).rename(columns=COLUMNS)
    if not {"time", "power"}.issubset(frame):
        raise ValueError("Turbine CSV requires Статистическое время and Нормализованная активная мощность")
    times = pd.to_datetime(frame.time, errors="raise")
    if times.duplicated().any():
        raise ValueError("Duplicate measurement timestamps: resolve before aggregation")
    if ((times.dt.minute % 10 != 0) | (times.dt.second != 0)).any():
        raise ValueError("Expected measurements on a 10-minute grid")
    # Ambiguous clock-change readings are excluded and counted, not guessed.
    localized = times.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    if timestamp_position == "end":
        localized = localized - pd.Timedelta(minutes=10)
    elif timestamp_position != "start":
        raise ValueError("timestamp_position must be start or end")
    power = pd.to_numeric(frame.power, errors="coerce")
    good = np.isfinite(power) & power.between(0, 1) & localized.notna()
    values = pd.DataFrame({"valid_time_utc": localized[good].dt.floor("h"), "power": power[good]})
    hourly = values.groupby("valid_time_utc").power.agg(["mean", "count"])
    valid = hourly.loc[hourly["count"] == 6].reset_index().rename(columns={"mean": "target_power"})
    valid["turbine_id"] = turbine_id
    audit = dict(source_rows=len(frame), invalid_or_ambiguous_rows=int((~good).sum()), ambiguous_or_nonexistent_timestamps=int(localized.isna().sum()), complete_hours=len(valid), incomplete_hours=int((hourly["count"] != 6).sum()), source_sha256=sha256(path))
    return valid[["turbine_id", "valid_time_utc", "target_power"]], audit

def assemble(weather_path, turbine_files, timezone, timestamp_position, alignment_confirmed=False, model="jma_gsm"):
    weather = pd.read_csv(weather_path)
    required = ["model", "turbine_id", "valid_time_utc", "provider_offset_days"] + WEATHER_COLUMNS
    if not set(required).issubset(weather):
        raise ValueError(f"Forecast CSV requires {required}")
    weather = weather.loc[(weather.model == model) & weather.turbine_id.isin(turbine_files)].copy()
    weather["valid_time_utc"] = pd.to_datetime(weather.valid_time_utc, utc=True, errors="raise")
    keys = ["turbine_id", "valid_time_utc", "provider_offset_days"]
    if weather.duplicated(keys).any():
        raise ValueError("Duplicate weather keys; do not mix archive revisions")
    if not weather.provider_offset_days.isin([1, 2]).all():
        raise ValueError("This baseline expects provider offsets 1 and 2")
    before = len(weather)
    finite = np.isfinite(weather[WEATHER_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    weather = weather.loc[finite]
    targets, audits = [], {}
    for turbine, path in turbine_files.items():
        target, audit = hourly_targets(path, turbine, timezone, timestamp_position)
        targets.append(target); audits[turbine] = audit
    if not targets:
        raise ValueError("Provide at least one turbine CSV")
    all_targets = pd.concat(targets, ignore_index=True)
    joined = weather.merge(all_targets, on=["turbine_id", "valid_time_utc"], how="inner", validate="many_to_one")
    joined = joined.loc[joined.valid_time_utc < pd.Timestamp("2026-02-01", tz="UTC")].sort_values(keys).reset_index(drop=True)
    if joined.empty:
        raise ValueError("No matched training rows; check timezone and weather coverage")
    metadata = dict(schema_version=1, weather_model=model, measurement_timezone=timezone, timestamp_position=timestamp_position, alignment_confirmed=alignment_confirmed, archive_semantics="provider_fixed_offset_not_exact_issue_time", weather_sha256=sha256(weather_path), weather_rows_selected=before, missing_weather_rows_dropped=before-len(weather), joined_rows=len(joined), unique_target_hours=int(joined[["turbine_id", "valid_time_utc"]].drop_duplicates().shape[0]), first_valid_time=str(joined.valid_time_utc.min()), last_valid_time=str(joined.valid_time_utc.max()), turbine_audits=audits)
    return joined, metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weather", required=True)
    p.add_argument("--turbine", action="append", required=True, help="turbine_id=path.csv (repeatable)")
    p.add_argument("--measurement-timezone", required=True, help="IANA zone or explicit UTC; no inferred default")
    p.add_argument("--timestamp-position", required=True, choices=["start", "end"])
    p.add_argument("--alignment-confirmed", action="store_true")
    p.add_argument("--model", default="jma_gsm")
    p.add_argument("--output", default="data/training.csv")
    args = p.parse_args()
    pairs = [item.split("=", 1) for item in args.turbine]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        p.error("Each turbine must have a unique id and CSV path")
    frame, metadata = assemble(args.weather, dict(pairs), args.measurement_timezone, args.timestamp_position, args.alignment_confirmed, args.model)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    metadata["dataset_sha256"] = sha256(path)
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
