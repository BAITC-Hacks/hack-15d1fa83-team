import json
import numpy as np
import pandas as pd
import pytest
from windpower.dataset import sha256
from windpower.train import train

@pytest.fixture(scope="session")
def trained_fixture(tmp_path_factory):
    """Small artificial test data verifies plumbing, never reported as model skill."""
    out = tmp_path_factory.mktemp("training")
    rows = []
    for date in ["2025-09-01", "2025-10-01", "2026-01-01"]:
        for i, timestamp in enumerate(pd.date_range(date, periods=24, freq="h", tz="UTC")):
            for offset in [1, 2]:
                speed = 2 + (i % 10) * 0.7 + offset * 0.05
                rows.append(dict(turbine_id="turbine_2", model="jma_gsm", valid_time_utc=timestamp.isoformat(), provider_offset_days=offset, wind_speed_10m_ms=speed, wind_direction_10m_deg=(i*25)%360, temperature_2m_c=5+i/10, target_power=float(np.clip(speed/12, 0, 1))))
    path = out / "training.csv"; pd.DataFrame(rows).to_csv(path, index=False)
    metadata = dict(dataset_sha256=sha256(path), weather_model="jma_gsm", alignment_confirmed=True, archive_semantics="synthetic_test_fixture")
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    report = train(path, out / "model", device="cpu", epochs=3, batch_size=32)
    return out, report

