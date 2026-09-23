# Implementation status

The branch contains reproducible archive download, explicit-timezone dataset assembly, a PyTorch MLP training module, portable NumPy inference, weather-provider HTTP integration, and a local NVIDIA Brev job controller.

35 local regression checks pass. A real Brev H200 training run for turbine 2 completed after 35 epochs, selecting epoch 25. Downloaded model `mlp-4163decdcbe5` reproduces all 1,478 exported January predictions (739 distinct hours); MAE is 0.18600 versus 0.25300 for the empirical wind curve. The 24- and 48-hour API paths were verified with real archived weather supplied through a simulated HTTP provider. Results remain provisional because measurement interval alignment is unconfirmed. Raw measurements, trained weights and credentials are not committed.

The automatic controller submits or resumes training, monitors completion, verifies downloaded results and optionally stops the selected GPU. The launcher recorded a successful GPU stop after local recovery of a ZIP-layout bug, now fixed. It can now optionally start an existing stopped instance without provisioning a new GPU; that startup option still needs live verification. This desktop agent session cannot directly access the authenticated WSL environment, so the user launches a Windows wrapper once.

Both real turbine CSVs are now available. The joint assembly contains 96,900 predictor/target rows for 48,450 turbine-hour targets: 23,666 hours for turbine 1 and 24,784 for turbine 2, each represented at two archive offsets. Both source CSVs end January 31, 2026; no February measurements were supplied. Separate coordinate requests and turbine-ID joins are preserved even though all overlapping JMA weather values are identical at the nearby locations. Missing/ambiguous-row counts and hashes are in `reports/dataset_assembly_both.json`.

The shared network learns turbine-ID features. Its export check covers every trained turbine, and evaluation reports separate metrics and baselines per turbine and offset. A tiny two-turbine fixture verifies this path locally. The joint real-data model is prepared for cloud training but has not been trained yet; the current saved model supports turbine 2 only.

The `yevgeniy` branch was inspected read-only. `docs/TEAM_INTEGRATION_PROPOSAL.md` proposes JMA GSM 10 m weather and Django posting prepared rows to a pure inference endpoint. That adapter is proposed, not implemented or agreed with the other developer. No other branch was modified.

Remaining: train and evaluate both turbines in Brev, confirm measurement alignment, agree and implement the shared direct-input contract, connect the actual team weather endpoint, and reproduce competition decisions using original archived runs with verified availability. Fixed-offset training is not proof of exact historical forecast availability.
