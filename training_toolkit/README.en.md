# Model creation toolkit

[Русская версия](README.ru.md) · [Inference and integration](../README.en.md)

## Purpose and separation

This directory is the dedicated entry point for creating and evaluating future models. Running the prediction server never invokes it. It dispatches to shared data/feature/training modules and existing controllers; there is one implementation of preprocessing rather than separate training and serving copies.

All commands below run from the **repository root**. Relative dataset/output paths are relative to your current directory. Paths containing spaces must be quoted. `YOUR_INSTANCE` and `JOB_ID` are placeholders for your real Brev instance and the receipt printed by submission; they are not weather-model inputs.

```bash
python training_toolkit/run.py --help
python training_toolkit/run.py train --help
python training_toolkit/run.py auto --help
python training_toolkit/run.py brev --help
```

| Command | Executes | Local requirements |
|---|---|---|
| `fetch` | Download JMA archive for both coordinates | Installed root package |
| `assemble` | Join weather to hourly turbine targets | Installed root package |
| `train` | Fit/export the MSE neural baseline | Root package with `train`; CUDA if requested |
| `auto` | Submit/resume, monitor, download/verify, optionally start/stop Brev | Standard-library Python + authenticated Brev CLI |
| `brev` | Manual submit/status/download | Standard-library Python + authenticated Brev CLI |
| `benchmark` | Fixed CPU comparison of losses/models | Root package with `train,benchmark` |
| `verify` | Independently reproduce exports and exercise direct API | Root package; no GPU/PyTorch required |

Use `inference/smoke_test.py` to verify a running service. That test needs only standard-library Python and the committed examples.

## 1. Environments

For data preparation and artifact verification, use the runtime setup in the main README. For training/development on a machine that will perform fitting:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[train,test,benchmark]'
```

For Windows CPU work, use `py -3.12 -m venv .venv` and `.\.venv\Scripts\python.exe` in place of `python`; activation is optional. For Brev control on Windows, run the standard-library controller inside WSL with the user's existing Brev login. PyTorch does not need to be installed on the local controller PC; it is installed/used on the remote worker.

The selected release was trained with Python 3.12.3, PyTorch 2.14.0+cu130, NumPy 2.5.3 and pandas 2.3.3 on a Brev H200. Full versions are in `models/production/metrics.json`. CUDA compatibility depends on the worker's driver/build; verify it on that machine. Retraining with different hardware/library versions can produce different weights despite the same seed. The committed artifact is the exact integration reference.

## 2. Required inputs and exclusions

Obtain organizer-provided turbine 1 and turbine 2 measurement CSVs. Put working data under ignored `data/`, for example:

```text
data/turbine_1.csv
data/turbine_2.csv
data/weather.csv
data/training.csv
data/training.metadata.json
```

The assembled dataset and metadata must remain together. The metadata contains a SHA-256 of the exact CSV bytes. Editing the CSV without reassembling invalidates this check.

Required source columns include `Статистическое время` and `Нормализованная активная мощность`. The source also contains measured wind and temperature, but those are not substituted for forecast predictors. Raw measurement files are not in Git. Although the original filenames mention February, the supplied files actually end January 31, 2026.

Do not train on February labels. Do not substitute reanalysis/observed future weather for archived forecasts. Fixed-offset weather archives and exact issue-time replay are different data products; retaining timestamps does not invent unavailable provenance.

## 3. Fetch weather archives

```bash
python training_toolkit/run.py fetch --start 2023-03-10 --end 2026-01-31 --output data/weather.csv
```

The downloader requests both stored coordinates, JMA GSM, 10 m wind speed/direction and 2 m temperature, with provider offsets 1 and 2. Units are m/s and Celsius, timestamps UTC. March 10 is an extra UTC boundary day covering measurements starting March 11 local time. Requests run month by month; compressed raw responses and request/hash metadata are cached beside the weather output in its cache directory. Cached response identity is checked before reuse. `weather.manifest.json` records request provenance and missing-weather counts.

This command accesses the network and provider limits/availability apply. Missing weather is counted, not changed to zero. Native JMA GSM data are coarser than the hourly interpolated output. The archive semantics are stored as `provider_fixed_offset_not_exact_issue_time`; this dataset alone does not prove which original model run was available at an exact historical decision time.

## 4. Assemble both turbines

```bash
python training_toolkit/run.py assemble \
  --weather data/weather.csv \
  --turbine turbine_1=data/turbine_1.csv \
  --turbine turbine_2=data/turbine_2.csv \
  --measurement-timezone Asia/Almaty \
  --timestamp-position start \
  --output data/training.csv
```

For PowerShell, put the command on one line or use PowerShell continuation syntax; Bash backslashes are not PowerShell continuations. Quote the entire `turbine_1=PATH WITH SPACES.csv` argument when needed.

`Asia/Almaty` includes the historical UTC+6 to UTC+5 change. A constant UTC+5 over the whole dataset is incorrect for this assumption. Ambiguous timestamps are removed and counted. The assembler rejects duplicate timestamps, requires all six ten-minute readings for each hourly mean, joins each turbine to its own coordinate request, drops missing required weather and excludes targets on/after 2026-02-01 UTC.

`--timestamp-position start` is currently provisional. If organizers confirm interval end, reassemble with `end`, retrain and re-evaluate. Only add `--alignment-confirmed` after confirming the time conventions. Never flip the artifact's flag without correcting and verifying its data.

The current assembled dataset has 96,900 rows and 48,450 turbine-hour targets. Different downloads/source corrections can change these counts and the dataset hash. Keep the new audit instead of claiming byte-identical reproduction.

## 5. Train directly on the chosen machine

### CUDA worker

```bash
python -m pip install -e '.[train]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python training_toolkit/run.py train --dataset data/training.csv --output-dir artifacts/candidate-01 --device cuda --epochs 80
```

The CUDA mode fails if no GPU is available. It does not silently fall back to CPU.

### Explicit local CPU training

```bash
python training_toolkit/run.py train --dataset data/training.csv --output-dir artifacts/candidate-cpu --device cpu --epochs 80
```

The baseline network is small; real local comparison fits were feasible with two CPU threads. Use a GPU for workloads that justify it, not for serving this model. The `train` command limits PyTorch CPU threads to two. Do not overwrite `models/production/` with an experiment.

Default training: seed 42, batch 512, AdamW learning rate 0.002, weight decay 0.001, 64/32 ReLU layers, sigmoid output, MSE loss. Maximum 80 epochs, patience 10, best checkpoint selected by validation MAE. Train before October 2025; validate October–December; score January. Split boundaries can be supplied using the trainer's help options, but keep February labels excluded and preserve chronological separation.

Outputs:

| File | Content |
|---|---|
| `model.json` | Weights, normalization, feature schema, supported turbines/source and dataset provenance |
| `metrics.json` | Versions, seed, splits, history, MAE/RMSE/bias, baselines and per-turbine/offset metrics |
| `evaluation_predictions.csv` | January targets/predictions for independent verification |

The trainer reloads the exported portable model and compares it with PyTorch before reporting success. Constant and wind-curve baselines use training targets only. Do not interpret duplicate archive-offset rows as independent observed target hours.

## 6. NVIDIA Brev: automatic workflow

Brev is a GPU-instance service. An account credit/coupon is not a model-inference API or an authentication key. Use an existing permitted instance. The controller does not provision a new instance or redeem a coupon.

Install the official Brev CLI using NVIDIA's current instructions, then run inside the authenticated Linux/WSL user session:

```bash
brev login
brev list
brev refresh
```

Select the **existing exact instance name** from the list. For this project's earlier runs it was `wind-training`. The normal future command is:

```bash
python3 training_toolkit/run.py auto \
  --instance YOUR_INSTANCE \
  --dataset data/training.csv \
  --output-dir artifacts/brev-candidate-01 \
  --epochs 80 --start-instance --stop-instance
```

Keep the controller terminal open. Starting a stopped GPU resumes compute billing. The controller checks completed results before startup, checks existing receipts, waits for GPU SSH, submits once, monitors training, downloads and verifies results, and only then requests instance stop. No commands/passwords should be required during a healthy run after authentication is configured.

If the dataset, instance and requested epochs match an existing submitted receipt, a relaunch resumes monitoring. A completed run is not trained again. Changing training settings requires a new output directory; do not erase receipts to force a rerun. The controller uses `progress.log`, a persistent OS-managed `automation.guard`, job receipts under `jobs/`, `results.zip`, extracted artifacts and `complete.json`. Guard-file existence does not mean it is locked; the OS releases ownership after process exit. Do not manually delete an active guard.

An old empty `automation.lock` is retired only after checking that a matching older controller is not alive. If another training window is active, close that old controller window and relaunch. Detached remote training is not deliberately cancelled by closing the local monitor. An interrupted submission marked `submitted: false` requires inspection because a remote operation may have partially succeeded; automatic duplication is refused.

On failures/timeouts, inspect `progress.log` and the Brev console. The controller does not automatically stop the GPU on every failure, because a training job may still be running. Check billing/state before leaving it idle. The default monitoring limit is 90 minutes; a later launch can resume an existing job. CLI status/start/stop waits are bounded. Initial instance startup uses `--detached`; its stopped-to-running recovery path still needs a live check even though the joint submit/download/stop flow succeeded.

### Manual operations

```bash
# Only build a bundle; execute no remote command:
python3 training_toolkit/run.py brev submit --instance YOUR_INSTANCE --dataset data/training.csv --prepare-only

# Submit a new job to an already-running GPU:
python3 training_toolkit/run.py brev submit --instance YOUR_INSTANCE --dataset data/training.csv --epochs 80

# Use the exact receipt printed by submission:
python3 training_toolkit/run.py brev status --job artifacts/brev-jobs/JOB_ID/job.json
python3 training_toolkit/run.py brev download --job artifacts/brev-jobs/JOB_ID/job.json --output artifacts/download/results.zip
```

Manual download returns a ZIP; it does not replace the serving artifact or automatically stop the instance. The automatic workflow performs verification/extraction and optional stop. `--prepare-only` receipts are not submitted jobs; do not place them in an auto-resume output folder and pretend they completed.

### Remote venv failure

If the worker says `ensurepip` is unavailable on the observed Ubuntu/Python 3.12 image:

```bash
brev exec YOUR_INSTANCE "sudo apt-get update && sudo apt-get install -y --no-install-recommends python3.12-venv"
```

This installs the package on the remote worker, not on the Windows PC. Match the venv package to the actual Python version if the image differs. Authentication/sudo setup may require the account owner once; never put passwords in source or receipts.

## 7. Compare candidate models

```bash
python -m pip install -e '.[train,benchmark]'
python training_toolkit/run.py benchmark --dataset data/training.csv --baseline models/production/model.json --output artifacts/comparison-01
```

The output directory must be new. The script fixes its candidate list before training, runs CPU MSE/MAE/Huber networks, a larger network, another seed, pooled/separate CatBoost variants and two fixed ensembles. It selects on October–December validation, writes `selection.json`, and then scores January. The January set was previously inspected in this project; it is excluded from fitting and selection in this experiment but is not a pristine external blind test.

Outputs retain protocol, package versions, candidate artifacts, validation predictions, full January predictions, per-turbine/per-offset scores, comparison JSON/CSV and a paired day-bootstrap estimate. The bootstrap is approximate within that month; it does not prove future seasonal improvement. The existing release is never replaced by a benchmark run.

Measured results are in `docs/MODEL_COMPARISON.md`: the MAE-selected ensemble lowers average absolute error but raises RMSE. This release keeps the original MSE network following the user's preference for smaller large-error contribution. Future selection must explicitly state the chosen metric and use validation data rather than cherry-picking January.

## 8. Verify and promote a candidate

```bash
python training_toolkit/run.py verify --dataset data/training.csv --results artifacts/candidate-01
```

Verification checks dataset identity, complete test coverage, target correspondence, portable predictions, aggregate/per-turbine scores and both 48-hour API paths, and writes independent results/reference cases. It deliberately allows provisional artifacts but reports their status. It currently verifies the MLP export format; a CatBoost/ensemble candidate requires an inference adapter before it can replace this service artifact.

Promotion is a separate, reviewable action:

1. Freeze the selection rule using validation metrics; record MAE, RMSE, bias and both turbine results.
2. Confirm source/units/coordinates and time alignment. Keep genuine limitations in metadata.
3. Independently verify the candidate and retain the previous artifact for rollback.
4. Copy the chosen model and reports into `models/production/`; update manifest hash/version and reference responses together.
5. Run the full regression suite and the real HTTP smoke test in a runtime-only environment.
6. Commit the release, restart deployment, confirm metadata, and invalidate caches for the old model version.

Do not promote simply by renaming an unrelated artifact to the old model version. Do not set `alignment_confirmed: true` merely to silence startup checks. CUDA training, local verification and competition replay compliance are different checks.

## 9. Reproducibility and references

- Exact current model and dataset hashes: `models/production/manifest.json`.
- Current training parameters, package versions, split and all scores: `models/production/metrics.json`.
- Raw source audit: `reports/dataset_assembly_both.json`.
- Selected model verification: `models/production/independent_verification.json`.
- Model comparison: `reports/model_comparison_v1.json`.
- [NVIDIA Brev instance management](https://docs.nvidia.com/brev/cli/instance-management).
- [Open-Meteo previous runs](https://open-meteo.com/en/docs/previous-runs-api) and [JMA data](https://open-meteo.com/en/docs/jma-api).

Large raw CSVs, cached weather, GPU bundles and experimental weights belong in ignored data/artifact directories or agreed external storage. Commit source code, chosen integration weights, provenance summaries and measured reports. Do not commit credentials or billing links.
