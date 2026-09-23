# Remote training with NVIDIA Brev

The supplied credit link belongs to NVIDIA Brev, a GPU-instance service. This is suitable for custom PyTorch training. A billing coupon is not an API credential; this repository does not store or redeem it. Available credit, GPU inventory, permissions and instance cost must be checked in the account.

The workflow is:

1. Prepare the small tabular dataset, or use the assembled CSV and its matching metadata file.
2. Authenticate the official Brev CLI and select the account's organization.
3. Start/select a GPU instance in the Brev console. Use a current Python 3.11/3.12 environment with a working CUDA-enabled PyTorch installation, or allow the launcher to install the training extra remotely.
4. Submit the job from the local PC. Upload, job monitoring and result download use Brev's authenticated CLI.
5. Download the model and evaluation report, then stop the instance when no longer needed.

The first dense network has only a few thousand parameters; select a modest single GPU rather than defaulting to multiple expensive GPUs. Check the actual current hourly price and remaining credit in Brev before starting an instance. The basic submission command does not manage instance lifetime; the automated controller supports explicit start/stop options for an existing instance.

## Authentication and instance selection

Follow the [official quickstart](https://docs.nvidia.com/brev/getting-started/quickstart). On Windows use WSL, as documented by Brev. Your Windows folders are accessible under `/mnt/c/` from WSL.

```bash
brev login
brev refresh
brev list
brev exec YOUR_INSTANCE "nvidia-smi"
```

For API-token authentication, use the exact login command shown in your Brev CLI/API-key settings. Keep the token in Brev's credential store, not in source files or training arguments. The local launcher calls the official CLI, which handles the service authentication and SSH configuration. It does not call the hosted NIM inference API.

## Local controller

### Automated training and retrieval

With an authenticated Brev CLI and an existing running GPU, this standard-library controller submits once, polls every 30 seconds, downloads and verifies the dataset identity and CUDA metadata in the result archive, saves the three model artifacts, and optionally stops the selected instance:

```bash
python3 scripts/auto_train.py --instance YOUR_INSTANCE --dataset data/training.csv --output-dir artifacts/full-run --epochs 80 --stop-instance
```

Keep its terminal open. Progress is also saved to `progress.log` in the output directory. A repeat invocation reuses a matching 80-epoch receipt from the output directory or `artifacts/brev-jobs`, and completed runs are not resubmitted. An interrupted submission requires inspection rather than automatically risking a second billable job. Monitoring times out after 90 minutes by default; a later invocation can resume a submitted job. On failure the controller reports the error and leaves the GPU available for diagnosis; check its state before leaving it idle. On success it downloads results before invoking `brev stop` when requested. No cloud instances are created by this controller.

The controller holds an operating-system lock on `automation.guard`. The file stays in place; its existence does not block relaunching, and the OS releases ownership when the controller exits or crashes. Do not delete it while a controller is running. For migration from the old empty `automation.lock`, the WSL launcher inspects local processes and removes the legacy file only when no matching controller is alive. If it reports an older active training window, close that old window and reopen the launcher. This does not intentionally cancel detached remote training; a submitted receipt is resumed. Never delete job receipts to bypass an interrupted-submission warning.

Startup uses the documented [`brev start INSTANCE --detached`](https://docs.nvidia.com/brev/cli/instance-management) option, then refreshes SSH configuration and checks GPU access. CLI status/start/stop calls have bounded waits, emit a waiting message every 30 seconds, and preserve diagnostic output on failure. If the start command times out, the controller checks the instance's fresh state before proceeding rather than blindly repeating startup. Live WSL verification is still required for this recovery change.

If the worker reports that `ensurepip` is unavailable, install its matching virtual-environment package remotely before submitting. For the observed Python 3.12 Ubuntu worker:

```bash
brev exec YOUR_INSTANCE "sudo apt-get update && sudo apt-get install -y --no-install-recommends python3.12-venv"
```

### Manual commands

Add `--start-instance` to the automated command to start the selected existing stopped instance. This resumes compute billing. The controller requires the exact name in `brev list`, refuses unknown instances, and waits for GPU SSH readiness before creating a job. Completed results are checked before startup, so reopening a finished run does not restart its GPU. Use separate output directories for different datasets.

No PyTorch, CUDA or GPU is needed on the local PC for these commands. Use standard Python and the Brev CLI:

```bash
# Optional: build and inspect the upload package without remote execution.
python scripts/brev_train.py submit --instance YOUR_INSTANCE --dataset data/training.csv --prepare-only

# Submit to the selected, already-running GPU instance.
python scripts/brev_train.py submit --instance YOUR_INSTANCE --dataset data/training.csv --epochs 80
```

The CSV must be accompanied by `training.metadata.json` from the assembler. Its checksum is verified. Each command creates an isolated local job directory and prints its receipt path. `--prepare-only` does not submit the prepared job; run `submit` without that flag to create and submit a fresh job.

The upload bundle contains only Python package code, package metadata, and the explicitly selected dataset/metadata pair. It does not include `.env`, credentials, billing links, unrelated files or the source turbine CSV. It is sent to the selected GPU instance, not GitHub.

The remote worker creates its own environment, trains with `--device cuda`, records its exit code and bundles outputs. It runs detached so closing the local terminal does not deliberately terminate training. The remote job directory is under `/tmp`; retrieve outputs before rebooting, stopping or deleting the instance. Provider lifecycle behavior varies.

```bash
python scripts/brev_train.py status --job artifacts/brev-jobs/JOB_ID/job.json
python scripts/brev_train.py download --job artifacts/brev-jobs/JOB_ID/job.json --output artifacts/brev-results.zip
python -m zipfile -e artifacts/brev-results.zip .
```

Download checks that the worker recorded exit code zero. A failed job leaves its log for inspection. Keep the receipt for troubleshooting or reconnecting from another terminal. After a successful download, verify the JSON artifacts and stop the instance if it is no longer serving work:

```bash
brev stop YOUR_INSTANCE
```

The result archive contains `artifacts/model.json`, `artifacts/metrics.json` and `artifacts/evaluation_predictions.csv`. Inference can run on an ordinary CPU server using the included container; retaining a GPU solely for this small model's inference is optional.

## What is and is not verified

The package builder, dataset checksum checks, training/export path and service integration are tested locally. The CUDA training path and Brev submission require an authenticated instance; no cloud training or spending has been performed by this draft. First run `nvidia-smi` and a short remote job if you need to check drivers before a longer run.

References: [Brev connectivity](https://docs.nvidia.com/brev/cli/connectivity), [file transfers](https://docs.nvidia.com/brev/guides/development-tools/file-transfer-scp), [GPU instances](https://docs.nvidia.com/brev/concepts/gpu-instances).
