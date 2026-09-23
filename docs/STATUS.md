# First implementation checkpoint

Work in progress. The initial branch contains archive download, explicit-timezone dataset assembly, a PyTorch MLP training module, portable inference, and the weather-provider HTTP integration.

This checkpoint has not yet passed the test suite and does not contain production-trained weights. Training defaults to CUDA and is intended for a remote GPU instance. Raw turbine measurements, API credentials and billing information are not committed.

Next: add NVIDIA Brev remote training instructions/launcher, regression tests, dataset assembly report and full setup documentation.
