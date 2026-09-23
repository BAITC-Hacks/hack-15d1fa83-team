# Measured model comparison

Selected using 2025 validation: **Average of two MAE networks**. January MAE fell from 0.178525 to 0.166403 (6.8% lower). January RMSE increased from 0.238635 to 0.245725 (3.0% worse). The current service artifact was not replaced.

| Candidate | OctвЂ“Dec validation MAE | January MAE | January RMSE | January bias |
|---|---:|---:|---:|---:|
| Current GPU-trained network | 0.194980 | 0.178525 | 0.238635 | +0.066447 |
| Same network, CPU control | 0.196616 | 0.175713 | 0.235943 | +0.065154 |
| Network trained with MAE | 0.179754 | 0.168336 | 0.248435 | +0.010660 |
| Network trained with Huber | 0.182389 | 0.169017 | 0.243618 | +0.018803 |
| Larger MAE network | 0.179784 | 0.163725 | 0.247235 | +0.016949 |
| MAE network, second seed | 0.179567 | 0.169407 | 0.251608 | +0.048563 |
| CatBoost, RMSE objective | 0.209629 | 0.200455 | 0.255481 | +0.091875 |
| CatBoost, MAE objective | 0.190824 | 0.179764 | 0.248035 | +0.045918 |
| CatBoost, deeper MAE trees | 0.189565 | 0.175397 | 0.247521 | +0.048031 |
| Separate CatBoost per turbine | 0.190443 | 0.178254 | 0.247590 | +0.047372 |
| Average of two MAE networks **(selected)** | 0.178645 | 0.166403 | 0.245725 | +0.029611 |
| Current network + best validation tree | 0.190492 | 0.174566 | 0.239338 | +0.057239 |

## Both turbines

| Turbine | Current MAE | Selected MAE | Reduction |
|---|---:|---:|---:|
| turbine_1 | 0.173897 | 0.165504 | 4.8% |
| turbine_2 | 0.183153 | 0.167302 | 8.7% |

## What was actually run

- Nine new fits/configurations plus the existing network and two fixed equal-weight ensembles; separate turbine configuration fits two trees. All new training ran locally on two CPU threads, using real data and random initialization. No Brev instance was started.
- Same weather inputs, provider, turbine IDs and chronological split for all models. Train before October 2025; validate OctoberвЂ“December 2025; evaluate January 2026. Checkpoints and winner were selected on validation MAE; selection.json was saved before January scoring.
- January contains 2,956 forecast/target rows: 739 distinct target hours per turbine, each at two weather archive offsets. Both turbines and offsets stay together across time splits. This is not 2,956 independent hours.
- All model files were reloaded and checked against their validation predictions. January metrics were independently recomputed from saved predictions. No raw future measurements were added to predictors.
- Larger MAE network had the lowest observed January MAE, but was not the validation-selected winner. We did not change the selection after seeing January.
- January had already been inspected earlier in the project. It was excluded from fitting and selection in this experiment, but is not a completely blind external test. No February labels were supplied.
- Day-block bootstrap of the selected MAE reduction: [0.00024557280753856027, 0.023562510873731363]. This approximate within-January interval does not capture seasonal drift or dependence across days.
- Timing alignment is still provisional, and fixed-offset archives do not establish exact forecast issue-time availability. All experiments share these limitations.
- No alternative weather providers or sequence models were tested in this run.

## Reproduce

Install the repository with its train and benchmark extras, then run:

```sh
python scripts/benchmark_models.py --dataset PATH/training.csv --baseline PATH/current/model.json --output PATH/new-comparison-folder
```

The output folder must not already exist. protocol.json records the fixed candidate list, versions, dataset hash and selection rules. All candidate artifacts, January predictions and detailed scores are retained locally.
