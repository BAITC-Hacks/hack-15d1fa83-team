# HackAlemAI wind-power predictor

Hackathon team repository for Арыс.

This component requests hourly weather from the team's weather module and predicts normalized turbine power. The prediction engine is a small neural network trained from random weights on archived forecasts and measured power. It needs no LLM and no manually supplied forecast issue time.

**Current state:** dataset assembly and the training/inference pipeline are implemented and tested. A real turbine-2 dataset is assembled locally. Full training is intended for an NVIDIA Brev GPU and has not been run there yet. No production weights, invented accuracy results, private measurements or credentials are included in Git.

## Run on NVIDIA Brev

The local PC controls the job; the Brev instance performs training. See [Brev setup and commands](docs/BREV.md). The controller uses only Python's standard library and the official Brev CLI locally. Once the dataset is assembled and you have a running, authenticated instance:

```bash
python scripts/brev_train.py submit --instance YOUR_INSTANCE --dataset data/training.csv
# The command prints a job receipt path. Use that exact path below.
python scripts/brev_train.py status --job artifacts/brev-jobs/JOB_ID/job.json
python scripts/brev_train.py download --job artifacts/brev-jobs/JOB_ID/job.json
```

`YOUR_INSTANCE` and `JOB_ID` identify your real instance and submitted job. They are not prediction inputs. The launcher uploads code and the selected dataset, starts a detached CUDA training job, and retrieves model/evaluation files. It never provisions an instance or redeems a coupon. On Windows, run the Brev CLI/controller in WSL.

## Install for data preparation or development

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

PowerShell activation: `.venv\Scripts\Activate.ps1`. Installing the package without `[train]` does not install PyTorch. Serving the exported model uses NumPy, so inference does not require a GPU.

## Assemble the training dataset

```bash
windpower-fetch --start 2023-03-10 --end 2026-01-31 --output data/weather.csv
windpower-assemble --weather data/weather.csv --turbine 'turbine_2=data/turbine_2.csv' --measurement-timezone Asia/Almaty --timestamp-position start --output data/training.csv
```

The extra March 10 UTC day covers local measurements starting March 11. Repeat `--turbine 'turbine_1=data/turbine_1.csv'` when that CSV is available. Only turbine 2 has been supplied so far; an artifact trained on turbine 2 refuses turbine 1 requests.

`--timestamp-position start` is an explicit provisional assumption, not a confirmed property of the source. Confirm interval start versus end with the organizers. The user's local-time interpretation is represented with `Asia/Almaty`, including the 2024 offset change. Rows with ambiguous clock-change times are excluded and counted. Add `--alignment-confirmed` only once these conventions are established; until then artifacts carry a provisional status.

The assembler requires all six valid ten-minute readings for an hourly target. Missing hours are not filled. Measured wind/temperature are not substituted for forecast inputs. It writes a CSV plus metadata with source hashes and coverage counts. [Dataset details and limitations](docs/DATA.md).

## Training on the GPU worker

The launcher performs these steps remotely; they can also be run from a Brev shell:

```bash
python -m pip install -e '.[train]'
windpower-train --dataset data/training.csv --output-dir artifacts --device cuda --epochs 80
```

- Inputs: forecast 10 m wind speed, wind direction, 2 m temperature; cyclic calendar features and turbine identity.
- Network: dense layers 64 → 32 → 1, ReLU hidden activations and sigmoid output in 0..1. Random initialization, AdamW, MSE training loss, validation-MAE early stopping.
- No pretrained weights, recent actual power, observed future weather, provider issue timestamp, or fabricated horizon values are model inputs.
- Targets before October 2025 train the model. October–December 2025 select the checkpoint. January 2026 is held out for evaluation. February labels are never included.
- All offset versions of one target hour stay in the same split. Feature scaling is fitted on the training split only.
- Reports include MAE, RMSE, bias and comparisons with training-only constant and empirical wind-curve baselines. Offset groups are reported separately but are not independent target observations.

Outputs: `model.json` (weights, preprocessing and provenance), `metrics.json`, and `evaluation_predictions.csv`. Exported NumPy predictions are checked against PyTorch predictions before training reports success. The training code defaults to CUDA and fails if no GPU is available; it does not silently run a full training job on the PC. `--device cpu` is available for explicit small development checks.

## Serve predictions

After downloading and extracting the trained artifact:

```bash
export MODEL_PATH=artifacts/model.json
export WEATHER_PROVIDER_URL=http://localhost:8001/weather/forecast
uvicorn windpower.api:app --host 0.0.0.0 --port 8000
```

Use environment variables directly or your deployment's environment-file mechanism; the application does not automatically load `.env`. PowerShell example: `$env:MODEL_PATH='artifacts/model.json'`.

```bash
curl -X POST http://localhost:8000/power/forecast \
  -H 'Content-Type: application/json' \
  -d '{"turbine_id":"turbine_2","hours":48}'
```

The service fetches the latest weather, validates the series and returns one normalized-power estimate per hour. `GET /health` checks that the artifact loaded. Interactive API schema: `/docs`.

The weather provider must follow [the integration contract](docs/WEATHER_CONTRACT.md), including the weather model identity. Retries are bounded. Missing/invalid/stale weather returns HTTP 502; unknown turbines return 422. A missing model prevents startup. No dummy forecast or fake power fallback exists inside this component.

For your teammate's historical/simulated forecast module, explicitly set `ALLOW_HISTORICAL_FORECASTS=1`. For a development model with unconfirmed measurement alignment, explicitly set `ALLOW_PROVISIONAL_MODEL=1`. These settings are separate and disabled by default. Simulation changes the accepted dates, not the prediction algorithm.

Container serving is also available via the included `Dockerfile`. Mount the real artifact read-only at `/model/model.json` and set the weather endpoint. The container does not include training data or weights.

## Verification

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
pytest -q
```

Tests cover timezone conversion, incomplete intervals, split isolation, training/export equivalence, bad-weather rejection, remote bundle contents and a full service request through a simulated upstream. Tiny artificial data is used only in tests to verify code; its errors are not claimed as real forecasting performance. GitHub Actions runs these checks on pushes and pull requests.

## Scope

This is the team's weather-to-power component, not the entire agent. The archive downloader is a training-data tool; live forecast acquisition belongs to the separate weather module. Archive coverage, exact-run replay compliance and prediction skill are different properties. The current fixed-offset baseline does not yet establish an auditable daily issue-time replay. See [data provenance](docs/DATA.md) before using it for the final competition evaluation.
