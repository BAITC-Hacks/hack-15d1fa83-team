# Implementation status

The branch contains reproducible archive download, explicit-timezone dataset assembly, a PyTorch MLP training module, portable NumPy inference, weather-provider HTTP integration, and a local NVIDIA Brev job controller.

Local regression checks pass. Tiny artificial data is used only for testing the training/export/service path. No real-data performance claim or production-trained artifact is made. Training defaults to CUDA and is intended for a remote Brev GPU instance. Raw turbine measurements, API credentials and billing information are not committed.

The real local turbine-2 assembly contains 49,568 predictor/target rows representing 24,784 complete target hours at two archive offsets, with no missing required JMA weather values among selected weather rows. Measurement interval-start semantics remain provisional. The source's six ambiguous clock-change readings and 219 incomplete hours were excluded and counted.

Still required before a production claim: confirm measurement alignment, run and evaluate on the authenticated Brev GPU, supply turbine-1 labels if needed, and implement the full original-run issue-time replay for the competition. The current archive product is explicitly a fixed-offset baseline.
