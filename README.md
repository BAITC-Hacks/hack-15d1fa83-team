# HackAlemAI wind-power predictor

Hackathon team repository for Арыс.

This component accepts 48 hourly weather rows from the team's Django orchestrator and predicts normalized turbine power at `POST /v1/predict`. The prediction engine is a small neural network trained from random weights on archived forecasts and measured power. It needs no LLM and no manually supplied forecast issue time. The optional `/power/forecast` endpoint can fetch weather for standalone use.

**Current state:** the turbine-2 model has been trained on Brev and independently verified. Both turbine datasets are assembled and joint training is supported. The direct-input HTTP contract is implemented and tested; see [the exact integration contract](docs/ML_SERVICE_CONTRACT.md) and [current verification status](docs/STATUS.md). No trained weights, private measurements or credentials are included in Git. Deploy the downloaded artifact and check `/v1/metadata` for its actual supported turbines.

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

The extra March 10 UTC day covers local measurements starting March 11. Repeat `--turbine 'turbine_1=data/turbine_1.csv'` to train both turbines; both source CSVs have now been supplied locally. An artifact trained only on turbine 2 still refuses turbine 1 requests until the joint weights are deployed.

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
# Required for current development weights whose measurement alignment is provisional.
export ALLOW_PROVISIONAL_MODEL=1
# Set ML_SERVICE_TOKEN to the same shared service token used by Django.
uvicorn windpower.api:app --host 0.0.0.0 --port 8000
```

Use environment variables directly or your deployment's environment-file mechanism; the application does not automatically load `.env`. PowerShell example: `$env:MODEL_PATH='artifacts/model.json'`.

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ML_SERVICE_TOKEN" \
  --data-binary @docs/examples/direct-input.json
```

The service validates the supplied series and returns 48 normalized-power estimates without fetching weather. The example contains synthetic weather for contract testing, not performance evaluation. `GET /health` checks artifact loading; authenticated `GET /v1/metadata` gives supported turbines/source, model version and schema. Interactive API schema: `/docs`.

Follow [the direct-input integration contract](docs/ML_SERVICE_CONTRACT.md), including model identity, units and timestamp semantics. Schema/source/turbine errors return 422, token failures 401. A missing model prevents startup. No dummy power fallback exists. The optional weather-fetching endpoint instead follows [the standalone weather contract](docs/WEATHER_CONTRACT.md) and requires `WEATHER_PROVIDER_URL`.

The direct-input endpoint accepts live or historical timestamps; Django owns forecast selection and historical availability checks. `ALLOW_HISTORICAL_FORECASTS` applies only to the optional weather-fetching endpoint. `ALLOW_PROVISIONAL_MODEL=1` explicitly permits a development artifact with unconfirmed measurement alignment; it is disabled by default and the response exposes the alignment flag.

Container serving is available via the included `Dockerfile`. Mount the real artifact read-only at `/model/model.json` and configure the shared token and provisional setting. A weather endpoint is unnecessary for direct inference. The container does not include training data or weights.

## Verification

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
pytest -q
```

Tests cover timezone conversion, incomplete intervals, split isolation, training/export equivalence, bad-weather rejection, remote bundle contents and a full service request through a simulated upstream. Tiny artificial data is used only in tests to verify code; its errors are not claimed as real forecasting performance. GitHub Actions runs these checks on pushes and pull requests.

## Scope

This is the team's weather-to-power component, not the entire agent. The archive downloader is a training-data tool; live forecast acquisition belongs to the separate weather module. Archive coverage, exact-run replay compliance and prediction skill are different properties. The current fixed-offset baseline does not yet establish an auditable daily issue-time replay. See [data provenance](docs/DATA.md) before using it for the final competition evaluation.
