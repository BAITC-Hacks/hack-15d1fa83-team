"""Reproducible JMA fixed-offset archive downloader for the two turbine pins."""
import argparse
import calendar
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

SITES = [("turbine_1", 43.645150, 78.535604), ("turbine_2", 43.643198, 78.538828)]
VARIABLES = {"wind_speed_10m": "wind_speed_10m_ms", "wind_direction_10m": "wind_direction_10m_deg", "temperature_2m": "temperature_2m_c"}

def fetch_archive(start, end, output):
    start, end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if end < start: raise ValueError("End must not precede start")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "archive_cache"; cache.mkdir(exist_ok=True)
    fields = ["turbine_id", "model", "valid_time_utc", "provider_offset_days", "requested_latitude", "requested_longitude", "provider_grid_latitude", "provider_grid_longitude"] + list(VARIABLES.values()) + ["source_key"]
    manifest, count, missing = [], 0, 0
    # Write atomically: failed downloads cannot leave a seemingly complete CSV.
    temporary = output.with_suffix(".partial.csv")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        cursor = start
        while cursor <= end:
            last = min(dt.date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1]), end)
            params = dict(latitude=",".join(str(s[1]) for s in SITES), longitude=",".join(str(s[2]) for s in SITES), start_date=str(cursor), end_date=str(last), models="jma_gsm", hourly=",".join(f"{v}_previous_day{o}" for o in [1, 2] for v in VARIABLES), wind_speed_unit="ms", temperature_unit="celsius", timezone="GMT")
            url = "https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
            key = hashlib.sha256(url.encode()).hexdigest()[:20]
            rawpath, metapath = cache / (key + ".json.gz"), cache / (key + ".meta.json")
            if rawpath.exists() and metapath.exists():
                raw = gzip.decompress(rawpath.read_bytes()); meta = json.loads(metapath.read_text())
                if meta["url"] != url or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                    raise ValueError("Archive cache failed provenance check")
            else:
                for attempt in range(4):
                    try:
                        with urllib.request.urlopen(url, timeout=90) as response: raw = response.read()
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if attempt == 3: raise
                        time.sleep(2 ** attempt)
                meta = dict(url=url, sha256=hashlib.sha256(raw).hexdigest(), retrieved_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_key=key)
                rawpath.write_bytes(gzip.compress(raw)); metapath.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, list) or len(data) != len(SITES): raise ValueError("Unexpected archive site response")
            expected_times = [(dt.datetime.combine(cursor, dt.time()) + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(((last-cursor).days + 1) * 24)]
            for site, response in zip(SITES, data):
                hourly = response["hourly"]
                if hourly["time"] != expected_times: raise ValueError("Archive has missing or unexpected timestamps")
                for variable in VARIABLES:
                    for offset in [1, 2]:
                        if len(hourly[f"{variable}_previous_day{offset}"]) != len(expected_times): raise ValueError("Malformed hourly array")
                for i, valid_time in enumerate(expected_times):
                    for offset in [1, 2]:
                        row = dict(turbine_id=site[0], model="jma_gsm", valid_time_utc=valid_time+":00Z", provider_offset_days=offset, requested_latitude=site[1], requested_longitude=site[2], provider_grid_latitude=response["latitude"], provider_grid_longitude=response["longitude"], source_key=key)
                        for variable, column in VARIABLES.items(): row[column] = hourly[f"{variable}_previous_day{offset}"][i]
                        missing += int(any(row[c] is None for c in VARIABLES.values()))
                        writer.writerow(row); count += 1
            manifest.append(meta)
            print(f"Downloaded {cursor} through {last}", flush=True)
            cursor = last + dt.timedelta(days=1)
    temporary.replace(output)
    summary = dict(model="jma_gsm", archive_semantics="provider_fixed_offset_not_exact_issue_time", native_source_interval_hours=6, output_interval_hours=1, rows=count, rows_with_missing_weather=missing, requests=manifest)
    output.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2023-03-10", help="Includes a UTC boundary day for local measurements")
    p.add_argument("--end", default="2026-01-31")
    p.add_argument("--output", default="data/weather.csv")
    a = p.parse_args(); result = fetch_archive(a.start, a.end, a.output)
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}))

if __name__ == "__main__": main()

